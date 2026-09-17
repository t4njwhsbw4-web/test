"""Cross-Sectional-Momentum: rotierendes Long-only-Portfolio über N Krypto-Assets.

These
-----
Relative Stärke zwischen Assets ist persistenter als die absolute Richtung
eines einzelnen Assets. Statt "steigt BTC morgen?" wird gefragt "welcher der
N Coins ist gerade der stärkste?" - und in die Top-k rotiert. Ökonomische
Begründung: Cross-sectional Momentum (Jegadeesh/Titman 1993, vielfach auf
andere Assetklassen repliziert) beruht darauf, dass Kapitalflüsse relativer
Performance mit Verzögerung folgen.

Kein Look-Ahead
---------------
Die Gewichts-Zeile zu Datum t wird ausschliesslich aus Daten bis
einschliesslich Bar t berechnet (Returns/Vola über abgeschlossene rolling
windows, kein shift(-n), kein center=True). Die Ausführung im Portfolio-
Backtest erfolgt - analog zu src/backtest/engine.py - mit einem Bar
Verzögerung: executed = weights.shift(1).

Die vorhandene evaluate() im Harness bewertet eine Einzel-Asset-Signalserie
und ist für ein rotierendes Portfolio nicht verwendbar. Die Portfolio-
Auswertung (backtest_portfolio) ist daher hier eigenständig implementiert;
bestehende Module werden nicht verändert.
"""
from __future__ import annotations

import dataclasses
import itertools
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from src.research.harness import DEFAULT_COST_BPS, RESEARCH_SYMBOLS, load_data

TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------- Datenpanel

def build_price_panel(
    symbols: list[str] | None = None,
    period: str = "dev",
    field: str = "close",
) -> pd.DataFrame:
    """Close-Preis-Panel: Zeilen = Datum, Spalten = Symbol.

    period ist auf "dev" festgenagelt - der Hold-out-Zeitraum ist versiegelt
    und wird in der Strategie-Entwicklung nie angefasst.
    """
    if period != "dev":
        raise ValueError(
            "Strategie-Entwicklung läuft ausschliesslich auf period='dev'. "
            "Der Hold-out wird genau einmal, in einem separaten Schritt, bewertet."
        )
    symbols = list(symbols or RESEARCH_SYMBOLS)
    cols = {s: load_data(s, period=period)[field] for s in symbols}
    panel = pd.DataFrame(cols).sort_index()
    return panel.dropna(how="all")


# ------------------------------------------------------------- Gewichtslogik

def generate_weights(
    price_panel: pd.DataFrame,
    lookback: int = 30,
    top_k: int = 2,
    rebalance_days: int = 7,
    vol_normalize: bool = False,
    vol_window: int = 30,
    absolute_filter: bool = False,
    **_ignored,
) -> pd.DataFrame:
    """Portfoliogewichte (Zeilen = Datum, Spalten = Symbol, Zeilensumme <= 1).

    lookback        : Fenster für den Momentum-Score (Rendite über lookback Bars)
    top_k           : Anzahl gehaltener Coins, gleichgewichtet mit 1/top_k
    rebalance_days  : nur jeden n-ten Bar wird neu geranked, sonst Gewichte halten
    vol_normalize   : Score = Rendite / realisierte Vola (risk-adjusted Momentum)
    absolute_filter : Coin wird nur gehalten, wenn sein eigenes Momentum > 0 ist;
                      sonst bleibt der Slot in Cash (Zeilensumme < 1)

    Long-only, kein Hebel: Gewichte in [0, 1/top_k], Zeilensumme <= 1.
    """
    prices = price_panel.sort_index()

    # Momentum-Score: Rendite über die letzten `lookback` Bars, bekannt zu t.
    score = prices / prices.shift(lookback) - 1.0

    if vol_normalize:
        daily = prices.pct_change()
        vol = daily.rolling(vol_window, min_periods=vol_window).std()
        score = score / vol.replace(0.0, np.nan)

    # Ein Coin ist nur handelbar, wenn sein Score zu t definiert ist.
    valid = score.notna() & prices.notna()
    score = score.where(valid)

    ranks = score.rank(axis=1, ascending=False, method="first")
    selected = (ranks <= top_k) & valid

    if absolute_filter:
        # Absolut-Filter immer auf der reinen Rendite, nicht auf dem
        # vol-normierten Score (Vorzeichen ist identisch, aber explizit ist besser).
        raw_mom = prices / prices.shift(lookback) - 1.0
        selected = selected & (raw_mom > 0)

    targets = selected.astype(float) / float(top_k)

    # Rebalancing: nur an jedem n-ten Bar neu setzen, dazwischen halten.
    first_valid = valid.any(axis=1).idxmax() if valid.any(axis=1).any() else None
    if first_valid is None:
        return pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

    positions = np.arange(len(prices.index))
    offset = int(prices.index.get_loc(first_valid))
    is_rebalance = ((positions - offset) % max(int(rebalance_days), 1) == 0) & (
        positions >= offset
    )

    weights = targets.where(pd.Series(is_rebalance, index=prices.index), other=np.nan)
    weights = weights.ffill().fillna(0.0)

    # Absicherung gegen Hebel/Shorts.
    weights = weights.clip(lower=0.0)
    row_sum = weights.sum(axis=1)
    over = row_sum > 1.0 + 1e-12
    if over.any():
        weights.loc[over] = weights.loc[over].div(row_sum[over], axis=0)
    return weights


PARAM_GRID: dict[str, list] = {
    "lookback": [7, 14, 30, 60, 90, 180],
    "top_k": [1, 2, 3],
    "rebalance_days": [1, 3, 7, 14, 30],
    "vol_normalize": [False, True],
    "absolute_filter": [False, True],
}


# ------------------------------------------------------- Portfolio-Auswertung

@dataclasses.dataclass
class PortfolioResult:
    label: str
    params: dict
    equity_curve: pd.Series
    returns: pd.Series
    total_return: float
    cagr: float
    sharpe: float
    max_drawdown: float
    n_rebalances: int
    total_turnover: float
    exposure: float
    yearly_sharpe: dict[str, float]
    top3_day_concentration: float

    def summary(self) -> dict:
        return {
            "label": self.label,
            **{f"p_{k}": v for k, v in self.params.items()},
            "sharpe": round(self.sharpe, 3),
            "cagr": round(self.cagr, 4),
            "max_drawdown": round(self.max_drawdown, 4),
            "total_return": round(self.total_return, 3),
            "n_rebalances": self.n_rebalances,
            "turnover": round(self.total_turnover, 1),
            "exposure": round(self.exposure, 3),
            "yearly_sharpe": {k: round(v, 2) for k, v in self.yearly_sharpe.items()},
            "top3_day_conc": round(self.top3_day_concentration, 3),
        }


def _yearly_sharpe(returns: pd.Series) -> dict[str, float]:
    out = {}
    for year, group in returns.groupby(returns.index.year):
        std = group.std()
        out[str(year)] = float((group.mean() / std) * np.sqrt(TRADING_DAYS_PER_YEAR)) if std > 0 else 0.0
    return out


def _top3_day_concentration(returns: pd.Series) -> float:
    positive = returns[returns > 0]
    if positive.empty:
        return 0.0
    return float(positive.nlargest(3).sum() / positive.sum())


def backtest_portfolio(
    price_panel: pd.DataFrame,
    weights: pd.DataFrame,
    label: str = "portfolio",
    params: dict | None = None,
    starting_cash: float = 10_000.0,
    transaction_cost_bps: float = DEFAULT_COST_BPS,
    warmup_from: pd.Timestamp | None = None,
) -> PortfolioResult:
    """Gewichtete Tagesrendite über alle gehaltenen Positionen, Kosten auf Turnover.

    Ausführung mit einem Bar Verzögerung (weights.shift(1)) - identisch zur
    Konvention in src/backtest/engine.py. Kosten: transaction_cost_bps pro
    Seite auf den umgeschichteten Anteil, d.h. sum(|Delta w|) * bps.
    """
    prices = price_panel.sort_index()
    weights = weights.reindex(index=prices.index, columns=prices.columns).fillna(0.0)

    executed = weights.shift(1).fillna(0.0)
    asset_returns = prices.pct_change().fillna(0.0)

    gross = (executed * asset_returns).sum(axis=1)

    delta = executed.diff()
    delta.iloc[0] = executed.iloc[0]
    turnover = delta.abs().sum(axis=1)
    costs = turnover * (transaction_cost_bps / 10_000.0)
    net = gross - costs

    if warmup_from is not None:
        net = net.loc[net.index >= warmup_from]
        turnover = turnover.loc[turnover.index >= warmup_from]
        executed = executed.loc[executed.index >= warmup_from]

    equity_curve = starting_cash * (1 + net).cumprod()
    total_return = float(equity_curve.iloc[-1] / starting_cash - 1.0)
    years = max(len(net) / TRADING_DAYS_PER_YEAR, 1e-9)
    cagr = (1 + total_return) ** (1 / years) - 1 if total_return > -1 else -1.0

    std = net.std()
    sharpe = float((net.mean() / std) * np.sqrt(TRADING_DAYS_PER_YEAR)) if std > 0 else 0.0

    running_max = equity_curve.cummax()
    max_drawdown = float((equity_curve / running_max - 1.0).min())

    return PortfolioResult(
        label=label,
        params=params or {},
        equity_curve=equity_curve,
        returns=net,
        total_return=total_return,
        cagr=float(cagr),
        sharpe=sharpe,
        max_drawdown=max_drawdown,
        n_rebalances=int((turnover > 1e-9).sum()),
        total_turnover=float(turnover.sum()),
        exposure=float(executed.sum(axis=1).mean()),
        yearly_sharpe=_yearly_sharpe(net),
        top3_day_concentration=_top3_day_concentration(net),
    )


# ------------------------------------------------------------- Benchmarks

def benchmark_buy_and_hold(
    price_panel: pd.DataFrame,
    symbol: str = "BTC/USD",
    warmup_from: pd.Timestamp | None = None,
) -> PortfolioResult:
    w = pd.DataFrame(0.0, index=price_panel.index, columns=price_panel.columns)
    w[symbol] = 1.0
    return backtest_portfolio(
        price_panel, w, label=f"buy_and_hold_{symbol}",
        transaction_cost_bps=0.0, warmup_from=warmup_from,
    )


def benchmark_equal_weight(
    price_panel: pd.DataFrame,
    warmup_from: pd.Timestamp | None = None,
) -> PortfolioResult:
    n = price_panel.shape[1]
    w = pd.DataFrame(1.0 / n, index=price_panel.index, columns=price_panel.columns)
    return backtest_portfolio(
        price_panel, w, label="equal_weight_all",
        transaction_cost_bps=0.0, warmup_from=warmup_from,
    )


# ------------------------------------------------------------- Grid-Search

def param_combinations(grid: dict[str, list] | None = None) -> list[dict]:
    grid = grid or PARAM_GRID
    keys = list(grid)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(grid[k] for k in keys))]


def run_grid(
    price_panel: pd.DataFrame | None = None,
    grid: dict[str, list] | None = None,
    cost_bps: float = DEFAULT_COST_BPS,
) -> tuple[list[PortfolioResult], dict[str, PortfolioResult]]:
    panel = build_price_panel() if price_panel is None else price_panel
    combos = param_combinations(grid)

    # Gemeinsamer Startpunkt für ALLE Varianten und Benchmarks, damit der
    # längste Lookback nicht gegen einen anderen Zeitraum verglichen wird.
    max_warmup = max(
        max(c["lookback"] for c in combos),
        max(PARAM_GRID.get("lookback", [0])),
        30,
    )
    warmup_from = panel.index[max_warmup]

    results = []
    for params in combos:
        w = generate_weights(panel, **params)
        results.append(
            backtest_portfolio(
                panel, w, label="xsec_mom", params=params,
                transaction_cost_bps=cost_bps, warmup_from=warmup_from,
            )
        )

    benchmarks = {
        "btc": benchmark_buy_and_hold(panel, "BTC/USD", warmup_from=warmup_from),
        "eq": benchmark_equal_weight(panel, warmup_from=warmup_from),
    }
    return results, benchmarks


def main() -> None:
    panel = build_price_panel()
    results, benchmarks = run_grid(panel)
    results.sort(key=lambda r: r.sharpe, reverse=True)

    print(f"Panel: {panel.shape[1]} Symbole, {panel.index.min().date()} .. "
          f"{panel.index.max().date()} ({len(panel)} Bars)\n")

    print("=== Benchmarks (kostenfrei, gleicher Zeitraum) ===")
    for b in benchmarks.values():
        print(" ", b.summary())

    print(f"\n=== Grid: {len(results)} Kombinationen, Top 10 nach Sharpe ===")
    for r in results[:10]:
        print(" ", r.summary())

    sharpes = np.array([r.sharpe for r in results])
    print(f"\nGrid-Verteilung Sharpe: median={np.median(sharpes):.3f} "
          f"mean={sharpes.mean():.3f} p25={np.percentile(sharpes, 25):.3f} "
          f"p75={np.percentile(sharpes, 75):.3f} max={sharpes.max():.3f}")
    for name, b in benchmarks.items():
        frac = float((sharpes > b.sharpe).mean())
        print(f"Anteil des Grids mit Sharpe > {b.label} ({b.sharpe:.3f}): {frac:.1%}")


if __name__ == "__main__":
    main()
