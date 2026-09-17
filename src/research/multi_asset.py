"""Multi-Asset-Allokation mit Volatilitaets-Steuerung.

Frage: Schlaegt eine diversifizierte Allokation (statisch oder vol-gesteuert)
Einzelasset-Buy-and-Hold RISIKOADJUSTIERT - und haelt das out-of-sample?

Diese Datei ist bewusst selbst-genuegsam: `src/backtest/engine.py` bewertet nur
Einzelasset-Signalserien, hier brauchen wir gewichtete Mehr-Asset-Portfolios mit
Drift zwischen den Rebalancing-Terminen und Turnover-Kosten pro Asset.

Zeitraum-Disziplin
------------------
Jede Parameterwahl (Lookback, Vol-Ziel, Rebalancing-Frequenz) passiert
AUSSCHLIESSLICH auf dem Dev-Zeitraum (bis 2018-12-31). Der Hold-out
(2019-01-01 bis heute) wird genau einmal am Ende mit fixierten Parametern
bewertet.

Kein Look-Ahead
---------------
- Gewichte zu t verwenden nur Daten bis einschliesslich t.
- Ausfuehrung mit einem Bar Verzoegerung (Gewichte aus Schluss t wirken auf
  die Rendite von t+1) - implementiert ueber `_shift_rebalance_dates`.
- Keine zentrierten Fenster, kein shift(-n).
"""
from __future__ import annotations

import dataclasses
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from src.data.fetch import fetch_ohlcv

TRADING_DAYS_PER_YEAR = 252

DEV_END = pd.Timestamp("2018-12-31")
HOLDOUT_START = pd.Timestamp("2019-01-01")

# Transaktionskosten pro Seite, in Basispunkten des gehandelten Volumens.
COST_BPS = {
    "SPY": 3.0,
    "QQQ": 3.0,
    "TLT": 3.0,
    "IEF": 3.0,
    "GLD": 3.0,
    "BTC-USD": 10.0,
}
DEFAULT_COST_BPS = 5.0

CLASSIC_UNIVERSE = ["SPY", "QQQ", "TLT", "IEF", "GLD"]
CRYPTO_UNIVERSE = CLASSIC_UNIVERSE + ["BTC-USD"]


# ---------------------------------------------------------------------------
# Daten
# ---------------------------------------------------------------------------
def load_closes(
    symbols: Sequence[str],
    history_days: int = 12000,
    cache_dir: str = "data_cache",
) -> pd.DataFrame:
    """Schlusskurse aller Symbole auf einem gemeinsamen Handelskalender.

    Der Kalender ist die Schnittmenge der ETF-Handelstage (NYSE). BTC handelt
    auch am Wochenende; wir reindexieren es auf die ETF-Tage, weil ein
    Portfolio, das ETFs enthaelt, nur an Boersentagen rebalanciert werden kann.
    Die Wochenend-Rendite von BTC steckt damit in der Montags-Rendite - das ist
    realistisch, nicht look-ahead.
    """
    closes = {}
    for sym in symbols:
        df = fetch_ohlcv(sym, history_days, cache_dir=cache_dir)
        closes[sym] = df["close"].sort_index()

    etf_symbols = [s for s in symbols if not s.endswith("-USD")]
    if etf_symbols:
        calendar = None
        for s in etf_symbols:
            idx = closes[s].index
            calendar = idx if calendar is None else calendar.intersection(idx)
    else:
        calendar = closes[symbols[0]].index

    out = pd.DataFrame({s: closes[s].reindex(calendar).ffill() for s in symbols})
    # Erster Tag, an dem ALLE Assets einen Kurs haben.
    out = out.dropna()
    return out


def common_window(frame: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    return frame.index.min(), frame.index.max()


# ---------------------------------------------------------------------------
# Kennzahlen
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class PortfolioResult:
    name: str
    returns: pd.Series
    equity_curve: pd.Series
    weights: pd.DataFrame  # tatsaechlich gehaltene Gewichte (Tagesbeginn)
    gross_exposure: pd.Series
    turnover: pd.Series  # einseitiger Umschlag pro Tag
    cost_drag: pd.Series

    @property
    def metrics(self) -> dict:
        m = performance_metrics(self.returns)
        m["avg_exposure"] = float(self.gross_exposure.mean())
        m["turnover_pa"] = float(self.turnover.sum() / _years(self.returns))
        m["cost_drag_pa"] = float(self.cost_drag.sum() / _years(self.returns))
        m["rebalances_pa"] = float((self.turnover > 1e-12).sum() / _years(self.returns))
        return m


def _years(returns: pd.Series) -> float:
    return max(len(returns) / TRADING_DAYS_PER_YEAR, 1e-9)


def performance_metrics(returns: pd.Series) -> dict:
    """Sharpe ohne risikofreien Zins (rf=0), konsistent mit src/backtest/engine.py."""
    returns = returns.dropna()
    if returns.empty:
        return {k: float("nan") for k in ("cagr", "vol", "sharpe", "max_drawdown", "calmar")}

    equity = (1.0 + returns).cumprod()
    total_return = float(equity.iloc[-1]) - 1.0
    years = _years(returns)
    cagr = (1.0 + total_return) ** (1.0 / years) - 1.0 if total_return > -1 else -1.0

    std = float(returns.std())
    vol = std * np.sqrt(TRADING_DAYS_PER_YEAR)
    sharpe = (float(returns.mean()) / std) * np.sqrt(TRADING_DAYS_PER_YEAR) if std > 0 else 0.0

    drawdown = equity / equity.cummax() - 1.0
    max_dd = float(drawdown.min())
    calmar = cagr / abs(max_dd) if max_dd < -1e-12 else float("nan")

    return {
        "cagr": float(cagr),
        "vol": float(vol),
        "sharpe": float(sharpe),
        "max_drawdown": max_dd,
        "calmar": float(calmar),
        "total_return": total_return,
    }


def annual_sharpe(returns: pd.Series) -> pd.Series:
    """Sharpe je Kalenderjahr - zeigt, ob ein Ergebnis von einem Jahr getragen wird."""
    out = {}
    for year, r in returns.groupby(returns.index.year):
        std = float(r.std())
        out[year] = (float(r.mean()) / std) * np.sqrt(TRADING_DAYS_PER_YEAR) if std > 0 else 0.0
    return pd.Series(out)


# ---------------------------------------------------------------------------
# Portfolio-Backtest
# ---------------------------------------------------------------------------
def rebalance_flags(index: pd.DatetimeIndex, frequency: str) -> pd.Series:
    """True an den Tagen, deren SCHLUSSKURSE eine neue Zielallokation bestimmen.

    Die Ausfuehrung erfolgt einen Bar spaeter (siehe `run_portfolio`).
    """
    s = pd.Series(False, index=index)
    if frequency == "daily":
        s[:] = True
        return s
    if frequency == "never":
        s.iloc[0] = True
        return s

    freq_map = {"monthly": "M", "quarterly": "Q", "weekly": "W", "annual": "A"}
    if frequency not in freq_map:
        raise ValueError(f"Unbekannte Rebalancing-Frequenz: {frequency!r}")
    period = index.to_period(freq_map[frequency])
    # Letzter Handelstag jeder Periode.
    last_of_period = ~pd.Series(period, index=index).duplicated(keep="last").values
    s[:] = last_of_period
    s.iloc[0] = True
    return s


def run_portfolio(
    closes: pd.DataFrame,
    target_weights: pd.DataFrame,
    rebalance: pd.Series,
    name: str = "portfolio",
    cost_bps: dict | None = None,
    cash_rate_pa: float = 0.0,
) -> PortfolioResult:
    """Bewertet eine Zielgewichts-Serie als Long-only-Portfolio ohne Hebel.

    target_weights[t] = Zielgewichte aus den Informationen bis Schluss t.
    rebalance[t] = True, wenn diese Ziele uebernommen werden sollen; wirksam
    wird das erst fuer die Rendite von t+1 (ein Bar Verzoegerung).

    Zwischen zwei Rebalancing-Terminen driften die Gewichte mit den
    Assetrenditen - genau wie ein echtes Portfolio, das nicht angefasst wird.
    Der nicht investierte Rest (1 - Summe der Gewichte) liegt in Cash.
    """
    cost_bps = cost_bps or COST_BPS
    assets = list(closes.columns)
    asset_returns = closes.pct_change().fillna(0.0)

    w_target = target_weights.reindex(index=closes.index, columns=assets).fillna(0.0)
    if (w_target.values < -1e-12).any():
        raise ValueError("Negative Gewichte - Long-only ist verletzt.")
    exposure = w_target.sum(axis=1)
    if (exposure.values > 1.0 + 1e-9).any():
        raise ValueError("Gesamtexposure > 1.0 - kein Hebel erlaubt.")

    rebalance = rebalance.reindex(closes.index).fillna(False).astype(bool)
    # Ein Bar Verzoegerung: Ziele vom Schluss t werden fuer die Rendite t+1 gehalten.
    exec_flag = rebalance.shift(1, fill_value=False)
    exec_flag.iloc[0] = False
    w_source = w_target.shift(1).fillna(0.0)

    bps = np.array([cost_bps.get(a, DEFAULT_COST_BPS) for a in assets]) / 10_000.0
    cash_daily = cash_rate_pa / TRADING_DAYS_PER_YEAR

    n = len(closes)
    held = np.zeros((n, len(assets)))
    port_ret = np.zeros(n)
    turnover = np.zeros(n)
    costs = np.zeros(n)

    w_prev_end = np.zeros(len(assets))
    r_mat = asset_returns.values
    w_src_mat = w_source.values
    exec_mat = exec_flag.values

    for t in range(n):
        if exec_mat[t]:
            w_start = w_src_mat[t].copy()
            trade = np.abs(w_start - w_prev_end)
            turnover[t] = trade.sum()
            costs[t] = float((trade * bps).sum())
        else:
            w_start = w_prev_end

        held[t] = w_start
        invested = float(w_start.sum())
        gross = float((w_start * r_mat[t]).sum()) + (1.0 - invested) * cash_daily
        net = gross - costs[t]
        port_ret[t] = net

        # Drift bis zum naechsten Handelstag: Gewichte wachsen mit ihren Assets.
        # Nenner ist der Bruttowert, da Kosten das Cash-Bein treffen.
        growth = 1.0 + gross
        if abs(growth) < 1e-12:
            w_prev_end = w_start
        else:
            w_prev_end = w_start * (1.0 + r_mat[t]) / growth

    returns = pd.Series(port_ret, index=closes.index, name=name)
    return PortfolioResult(
        name=name,
        returns=returns,
        equity_curve=(1.0 + returns).cumprod(),
        weights=pd.DataFrame(held, index=closes.index, columns=assets),
        gross_exposure=pd.Series(held.sum(axis=1), index=closes.index),
        turnover=pd.Series(turnover, index=closes.index),
        cost_drag=pd.Series(costs, index=closes.index),
    )


# ---------------------------------------------------------------------------
# Gewichtungs-Regeln (alle kausal: nur Daten bis t)
# ---------------------------------------------------------------------------
def weights_buy_and_hold(closes: pd.DataFrame, symbol: str) -> pd.DataFrame:
    w = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)
    w[symbol] = 1.0
    return w


def weights_static(closes: pd.DataFrame, allocation: dict) -> pd.DataFrame:
    w = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)
    for sym, weight in allocation.items():
        w[sym] = weight
    return w


def weights_equal(closes: pd.DataFrame, symbols: Sequence[str]) -> pd.DataFrame:
    return weights_static(closes, {s: 1.0 / len(symbols) for s in symbols})


def weights_inverse_vol(
    closes: pd.DataFrame,
    symbols: Sequence[str],
    lookback: int,
) -> pd.DataFrame:
    """Risk Parity im klassischen Sinn: Gewicht proportional zu 1/Vola.

    Die Vola zu t nutzt die letzten `lookback` Renditen bis einschliesslich t
    (rolling, nicht zentriert). Summe der Gewichte = 1.
    """
    rets = closes[list(symbols)].pct_change()
    vol = rets.rolling(lookback, min_periods=lookback).std()
    inv = 1.0 / vol.replace(0.0, np.nan)
    w = inv.div(inv.sum(axis=1), axis=0)
    out = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)
    out[list(symbols)] = w.fillna(0.0)
    return out


def ex_ante_portfolio_vol(
    closes: pd.DataFrame,
    weights: pd.DataFrame,
    lookback: int,
) -> pd.Series:
    """Erwartete Portfolio-Vola zu t: w_t' Sigma_t w_t, Sigma aus den letzten
    `lookback` Tagen bis einschliesslich t.

    Die volle Kovarianzmatrix (nicht nur die Diagonale) ist hier der Punkt:
    Diversifikation lebt von den Korrelationen, und ein Vol-Ziel, das sie
    ignoriert, skaliert systematisch falsch.
    """
    assets = list(closes.columns)
    rets = closes.pct_change()
    w_mat = weights.reindex(columns=assets).fillna(0.0).values
    r_mat = rets.values
    n = len(closes)
    out = np.full(n, np.nan)

    for t in range(lookback, n):
        window = r_mat[t - lookback + 1 : t + 1]
        if np.isnan(window).any():
            continue
        cov = np.cov(window, rowvar=False)
        w = w_mat[t]
        var = float(w @ np.atleast_2d(cov) @ w)
        if var > 0:
            out[t] = np.sqrt(var * TRADING_DAYS_PER_YEAR)
    return pd.Series(out, index=closes.index)


def apply_vol_target(
    closes: pd.DataFrame,
    base_weights: pd.DataFrame,
    lookback: int,
    target_vol: float,
    max_exposure: float = 1.0,
) -> pd.DataFrame:
    """Skaliert das Gesamtexposure so, dass die ex-ante Vola das Ziel trifft.

    Long-only, kein Hebel: der Skalar ist auf [0, max_exposure] begrenzt. Der
    Rest liegt in Cash. Fehlt die Vola-Schaetzung (Anlaufphase), ist das
    Exposure 0 - kein Raten.
    """
    vol = ex_ante_portfolio_vol(closes, base_weights, lookback)
    scale = (target_vol / vol).clip(upper=max_exposure)
    scale = scale.fillna(0.0)
    return base_weights.mul(scale, axis=0)


def add_sleeve(base_weights: pd.DataFrame, symbol: str, sleeve: float) -> pd.DataFrame:
    """Mischt `sleeve` (z.B. 0.03) in ein Portfolio, skaliert den Rest herunter.

    Gesamtexposure bleibt unveraendert - die Beimischung wird finanziert, nicht
    zusaetzlich gehebelt.
    """
    out = base_weights.copy()
    if symbol not in out.columns:
        out[symbol] = 0.0
    total = out.sum(axis=1)
    out = out.mul((1.0 - sleeve), axis=0)
    out[symbol] = out[symbol] + total * sleeve
    return out


# ---------------------------------------------------------------------------
# Varianten-Definition
# ---------------------------------------------------------------------------
def build_variants(
    closes: pd.DataFrame,
    universe: Sequence[str],
    benchmarks: Sequence[str],
    rp_lookback: int,
    vt_lookback: int,
    target_vol: float,
    rebalance_freq: str,
    crypto_symbol: str | None = None,
    crypto_sleeves: Iterable[float] = (0.01, 0.03, 0.05),
) -> dict:
    """Erzeugt (target_weights, rebalance_flags) je Variante."""
    reb = rebalance_flags(closes.index, rebalance_freq)
    daily = rebalance_flags(closes.index, "daily")
    never = rebalance_flags(closes.index, "never")
    variants = {}

    for b in benchmarks:
        if b in closes.columns:
            variants[f"B&H {b}"] = (weights_buy_and_hold(closes, b), never)

    if "SPY" in closes.columns and "TLT" in closes.columns:
        variants["60/40 SPY-TLT"] = (
            weights_static(closes, {"SPY": 0.6, "TLT": 0.4}),
            reb,
        )
    if "SPY" in closes.columns and "IEF" in closes.columns:
        variants["60/40 SPY-IEF"] = (
            weights_static(closes, {"SPY": 0.6, "IEF": 0.4}),
            reb,
        )

    eq = weights_equal(closes, universe)
    variants[f"Equal-Weight ({len(universe)})"] = (eq, reb)

    rp = weights_inverse_vol(closes, universe, rp_lookback)
    variants["Risk-Parity (inv-vol)"] = (rp, reb)

    # Vol-Targeting auf Portfolioebene, aufgesetzt auf zwei Basis-Allokationen.
    if "SPY" in closes.columns and "TLT" in closes.columns:
        variants["60/40 + VolTarget"] = (
            apply_vol_target(
                closes,
                weights_static(closes, {"SPY": 0.6, "TLT": 0.4}),
                vt_lookback,
                target_vol,
            ),
            daily,
        )
    variants["Risk-Parity + VolTarget"] = (
        apply_vol_target(closes, rp, vt_lookback, target_vol),
        daily,
    )
    variants["Equal-Weight + VolTarget"] = (
        apply_vol_target(closes, eq, vt_lookback, target_vol),
        daily,
    )

    if crypto_symbol and crypto_symbol in closes.columns:
        for sleeve in crypto_sleeves:
            pct = int(round(sleeve * 100))
            variants[f"Risk-Parity + {pct}% BTC"] = (
                add_sleeve(rp, crypto_symbol, sleeve),
                reb,
            )
            variants[f"RP+VT + {pct}% BTC"] = (
                apply_vol_target(
                    closes, add_sleeve(rp, crypto_symbol, sleeve), vt_lookback, target_vol
                ),
                daily,
            )
            variants[f"60/40 + {pct}% BTC"] = (
                add_sleeve(weights_static(closes, {"SPY": 0.6, "TLT": 0.4}), crypto_symbol, sleeve),
                reb,
            )

    return variants


def evaluate(
    closes: pd.DataFrame,
    variants: dict,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    cash_rate_pa: float = 0.0,
) -> tuple[pd.DataFrame, dict]:
    """Wertet alle Varianten auf einem Zeitfenster aus.

    Wichtig: Die Gewichte werden auf der VOLLEN Historie berechnet (rolling,
    kausal) und erst danach zugeschnitten. So braucht kein Fenster eine
    Anlaufphase, und der Hold-out beginnt ohne Warmup-Loch - ohne dass je
    Zukunftsinformation in ein Gewicht einfliesst.
    """
    rows = {}
    results = {}
    for name, (w, reb) in variants.items():
        sub_closes = closes.loc[slice(start, end)]
        sub_w = w.loc[sub_closes.index]
        sub_reb = reb.loc[sub_closes.index].copy()
        sub_reb.iloc[0] = True  # Einstieg am Fensterbeginn
        res = run_portfolio(
            sub_closes, sub_w, sub_reb, name=name, cash_rate_pa=cash_rate_pa
        )
        results[name] = res
        rows[name] = res.metrics
    table = pd.DataFrame(rows).T
    cols = [
        "sharpe",
        "cagr",
        "vol",
        "max_drawdown",
        "calmar",
        "avg_exposure",
        "turnover_pa",
        "cost_drag_pa",
    ]
    return table[cols], results


# ---------------------------------------------------------------------------
# Parametersuche (nur Dev!)
# ---------------------------------------------------------------------------
def dev_parameter_scan(
    closes: pd.DataFrame,
    universe: Sequence[str],
    lookbacks: Sequence[int] = (21, 63, 126, 252),
    target_vols: Sequence[float] = (0.06, 0.08, 0.10, 0.12),
    rebalance_freqs: Sequence[str] = ("monthly", "quarterly"),
) -> pd.DataFrame:
    """Gitter fuer Risk-Parity + VolTarget, bewertet ausschliesslich bis DEV_END."""
    rows = []
    for lb in lookbacks:
        rp = weights_inverse_vol(closes, universe, lb)
        for freq in rebalance_freqs:
            reb = rebalance_flags(closes.index, freq)
            sub = closes.loc[:DEV_END]
            base = run_portfolio(
                sub, rp.loc[sub.index], reb.loc[sub.index], name="rp"
            )
            m = base.metrics
            rows.append(
                {
                    "variant": "risk_parity",
                    "lookback": lb,
                    "target_vol": np.nan,
                    "rebalance": freq,
                    **{k: m[k] for k in ("sharpe", "cagr", "max_drawdown", "calmar", "turnover_pa")},
                }
            )
            for tv in target_vols:
                vt = apply_vol_target(closes, rp, lb, tv)
                daily = rebalance_flags(closes.index, "daily")
                res = run_portfolio(
                    sub, vt.loc[sub.index], daily.loc[sub.index], name="rp_vt"
                )
                m = res.metrics
                rows.append(
                    {
                        "variant": "rp_voltarget",
                        "lookback": lb,
                        "target_vol": tv,
                        "rebalance": freq,
                        **{
                            k: m[k]
                            for k in ("sharpe", "cagr", "max_drawdown", "calmar", "turnover_pa")
                        },
                    }
                )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def _fmt(table: pd.DataFrame) -> str:
    show = table.copy()
    for c in ("cagr", "vol", "max_drawdown", "cost_drag_pa"):
        if c in show:
            show[c] = (show[c] * 100).round(2)
    for c in ("sharpe", "calmar", "avg_exposure", "turnover_pa"):
        if c in show:
            show[c] = show[c].round(3)
    show = show.rename(
        columns={
            "cagr": "CAGR%",
            "vol": "Vol%",
            "max_drawdown": "MDD%",
            "cost_drag_pa": "Kosten%pa",
            "avg_exposure": "AvgExp",
            "turnover_pa": "Turn/J",
        }
    )
    return show.to_string()


def main() -> None:
    pd.set_option("display.width", 200)

    print("=" * 78)
    print("DATENVERFUEGBARKEIT")
    print("=" * 78)
    for sym in CRYPTO_UNIVERSE:
        df = fetch_ohlcv(sym, 12000, cache_dir="data_cache")
        print(f"  {sym:9s} {len(df):5d} Bars  {df.index.min().date()} .. {df.index.max().date()}")

    closes_classic = load_closes(CLASSIC_UNIVERSE)
    closes_crypto = load_closes(CRYPTO_UNIVERSE)
    c0, c1 = common_window(closes_classic)
    k0, k1 = common_window(closes_crypto)
    print(f"\n  Gemeinsamer Zeitraum OHNE BTC: {c0.date()} .. {c1.date()} ({len(closes_classic)} Bars)")
    print(f"  Gemeinsamer Zeitraum MIT  BTC: {k0.date()} .. {k1.date()} ({len(closes_crypto)} Bars)")
    print(f"  Dev bis {DEV_END.date()}, Hold-out ab {HOLDOUT_START.date()}")

    # --- Schritt 1: Parameterwahl, nur Dev, nur klassisches Universum ------
    print("\n" + "=" * 78)
    print("SCHRITT 1: PARAMETERSCAN (nur Dev 2004-2018, klassisches Universum)")
    print("=" * 78)
    scan = dev_parameter_scan(closes_classic, CLASSIC_UNIVERSE)
    vt = scan[scan.variant == "rp_voltarget"].sort_values("calmar", ascending=False)
    print("\nTop 12 nach Dev-Calmar:")
    print(vt.head(12).to_string(index=False))
    print("\nRisk-Parity ohne VolTarget (Referenz):")
    print(scan[scan.variant == "risk_parity"].to_string(index=False))

    print("\nStabilitaet: Dev-Sharpe-Matrix (Zeilen=Lookback, Spalten=Vol-Ziel)")
    print(
        vt.pivot_table(index="lookback", columns="target_vol", values="sharpe", aggfunc="mean")
        .round(3)
        .to_string()
    )
    print("\nDev-Calmar-Matrix")
    print(
        vt.pivot_table(index="lookback", columns="target_vol", values="calmar", aggfunc="mean")
        .round(3)
        .to_string()
    )

    # Fixierte Parameter - ab hier NICHT mehr angefasst.
    best = vt.iloc[0]
    params = {
        "rp_lookback": int(best.lookback),
        "vt_lookback": int(best.lookback),
        "target_vol": float(best.target_vol),
        "rebalance_freq": str(best.rebalance),
    }
    print(f"\n>>> FIXIERTE PARAMETER (aus Dev): {params}")

    # --- Schritt 2: Dev-Vergleichstabelle ----------------------------------
    print("\n" + "=" * 78)
    print(f"SCHRITT 2: DEV-VERGLEICH, klassisches Universum ({c0.date()} .. {DEV_END.date()})")
    print("=" * 78)
    v_classic = build_variants(
        closes_classic, CLASSIC_UNIVERSE, ["SPY", "QQQ"], **params
    )
    dev_classic, _ = evaluate(closes_classic, v_classic, end=DEV_END)
    print(_fmt(dev_classic.sort_values("sharpe", ascending=False)))

    print("\n" + "=" * 78)
    print(f"SCHRITT 2b: DEV-VERGLEICH mit BTC ({k0.date()} .. {DEV_END.date()})")
    print("=" * 78)
    v_crypto = build_variants(
        closes_crypto,
        CLASSIC_UNIVERSE,
        ["SPY", "QQQ", "BTC-USD"],
        crypto_symbol="BTC-USD",
        **params,
    )
    dev_crypto, _ = evaluate(closes_crypto, v_crypto, end=DEV_END)
    print(_fmt(dev_crypto.sort_values("sharpe", ascending=False)))

    # --- Schritt 3: Hold-out, EINMAL ---------------------------------------
    print("\n" + "=" * 78)
    print(f"SCHRITT 3: HOLD-OUT {HOLDOUT_START.date()} .. {c1.date()} - EINMALIGE BEWERTUNG")
    print("=" * 78)
    ho_classic, res_ho_classic = evaluate(closes_classic, v_classic, start=HOLDOUT_START)
    print("\nKlassisches Universum:")
    print(_fmt(ho_classic.sort_values("sharpe", ascending=False)))

    ho_crypto, _ = evaluate(closes_crypto, v_crypto, start=HOLDOUT_START)
    print("\nMit BTC-Beimischung:")
    print(_fmt(ho_crypto.sort_values("sharpe", ascending=False)))

    # --- Jahres-Sharpe ------------------------------------------------------
    print("\n" + "=" * 78)
    print("JAHRES-SHARPE (Hold-out, klassisches Universum)")
    print("=" * 78)
    focus = [
        "B&H SPY",
        "60/40 SPY-TLT",
        "Equal-Weight (5)",
        "Risk-Parity (inv-vol)",
        "Risk-Parity + VolTarget",
        "60/40 + VolTarget",
    ]
    ann = pd.DataFrame(
        {n: annual_sharpe(res_ho_classic[n].returns) for n in focus if n in res_ho_classic}
    )
    print(ann.round(2).to_string())

    print("\n" + "=" * 78)
    print("DEV -> HOLD-OUT SHARPE/CALMAR (Uebertragbarkeit)")
    print("=" * 78)
    comp = pd.DataFrame(
        {
            "dev_sharpe": dev_classic["sharpe"],
            "ho_sharpe": ho_classic["sharpe"],
            "dev_calmar": dev_classic["calmar"],
            "ho_calmar": ho_classic["calmar"],
            "dev_mdd": dev_classic["max_drawdown"],
            "ho_mdd": ho_classic["max_drawdown"],
        }
    ).round(3)
    print(comp.to_string())

    # --- Sensitivitaet: Cash verzinst --------------------------------------
    print("\n" + "=" * 78)
    print("SENSITIVITAET: Cash mit 2% p.a. (rf weiterhin 0 im Sharpe)")
    print("=" * 78)
    ho_cash, _ = evaluate(
        closes_classic, v_classic, start=HOLDOUT_START, cash_rate_pa=0.02
    )
    print(_fmt(ho_cash.sort_values("sharpe", ascending=False)))


if __name__ == "__main__":
    main()
