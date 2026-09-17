"""Test-Harness mit versiegeltem Hold-out-Zeitraum.

Kernidee gegen Overfitting: Strategien werden AUSSCHLIESSLICH auf dem
Development-Zeitraum entwickelt und optimiert. Der Hold-out-Zeitraum (das
letzte Jahr) wird erst angefasst, wenn eine Strategie final bewertet wird -
und zwar genau einmal. Wer im Hold-out nachoptimiert, hat keinen Hold-out
mehr, sondern nur einen zweiten Trainingsdatensatz.

Zusätzlich liefert evaluate() eine Jahres-Aufschlüsselung. Ein einzelner
guter Gesamt-Sharpe sagt fast nichts (siehe SHIB: 47% der Rendite aus
3 Tagen). Eine Strategie, die über mehrere Jahre hinweg konsistent
funktioniert, ist ein deutlich stärkeres Signal.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from src.backtest.engine import run_backtest
from src.data.fetch import fetch_ohlcv, to_yahoo_symbol

# Realistische Retail-Kosten für Krypto-Spot: ~0.1% Taker-Fee pro Seite.
# Zu niedrige Kostenannahmen sind die zweithäufigste Ursache für Backtests,
# die live nicht funktionieren (nach Look-Ahead-Bias).
DEFAULT_COST_BPS = 10.0

HISTORY_DAYS = 2200          # ~6 Jahre, damit Dev-Zeitraum UND Hold-out tragfähig sind
HOLDOUT_DAYS = 365           # letztes Jahr: versiegelt
CACHE_DIR = "data_cache"

RESEARCH_SYMBOLS = ["BTC/USD", "ETH/USD", "SOL/USD", "DOGE/USD", "LTC/USD", "ADA/USD"]


def holdout_cutoff() -> pd.Timestamp:
    """Alles ab diesem Datum ist versiegelt."""
    return pd.Timestamp(dt.datetime.utcnow().date() - dt.timedelta(days=HOLDOUT_DAYS))


def load_data(symbol: str, period: str = "dev") -> pd.DataFrame:
    """period: "dev" (Entwicklung erlaubt) | "holdout" (nur finale Bewertung) | "full".

    Für Strategie-Entwicklung IMMER period="dev" verwenden.
    """
    raw = fetch_ohlcv(to_yahoo_symbol(symbol), history_days=HISTORY_DAYS, cache_dir=CACHE_DIR)
    cutoff = holdout_cutoff()

    if period == "dev":
        return raw.loc[raw.index < cutoff]
    if period == "holdout":
        return raw.loc[raw.index >= cutoff]
    if period == "full":
        return raw
    raise ValueError(f"Unbekannter period-Wert: {period!r}")


@dataclasses.dataclass
class StrategyResult:
    symbol: str
    params: dict
    sharpe: float
    cagr: float
    max_drawdown: float
    win_rate: float
    n_trades: int
    total_return: float
    yearly_sharpe: dict[str, float]
    top3_day_concentration: float
    equity_curve: pd.Series

    def summary(self) -> dict:
        return {
            "symbol": self.symbol,
            **{f"param_{k}": v for k, v in self.params.items()},
            "sharpe": round(self.sharpe, 3),
            "cagr": round(self.cagr, 4),
            "max_drawdown": round(self.max_drawdown, 4),
            "win_rate": round(self.win_rate, 3),
            "n_trades": self.n_trades,
            "total_return": round(self.total_return, 4),
            "yearly_sharpe": {k: round(v, 2) for k, v in self.yearly_sharpe.items()},
            "top3_day_concentration": round(self.top3_day_concentration, 3),
        }


def _yearly_sharpe(returns: pd.Series) -> dict[str, float]:
    out = {}
    for year, group in returns.groupby(returns.index.year):
        std = group.std()
        out[str(year)] = float((group.mean() / std) * np.sqrt(252)) if std > 0 else 0.0
    return out


def _top3_day_concentration(returns: pd.Series) -> float:
    """Anteil der 3 besten Tage an der Summe aller positiven Tage.

    Hoher Wert (> 0.35) = die Rendite hängt an wenigen Glückstagen und ist
    mit hoher Wahrscheinlichkeit nicht reproduzierbar.
    """
    positive = returns[returns > 0]
    if positive.empty:
        return 0.0
    return float(positive.nlargest(3).sum() / positive.sum())


def evaluate(
    signals: pd.Series,
    prices: pd.Series,
    symbol: str,
    params: dict,
    cost_bps: float = DEFAULT_COST_BPS,
) -> StrategyResult:
    """Bewertet eine fertige Signal-Serie. signals: 1 = long, 0 = flat.

    Die Ausführung erfolgt im Backtest mit einem Bar Verzögerung
    (signals.shift(1)), d.h. ein Signal aus Bar t wird zu Bar t+1 gehandelt.
    Signale dürfen daher NUR aus Daten bis einschliesslich Bar t berechnet
    werden - sonst Look-Ahead-Bias.
    """
    result = run_backtest(prices, signals, transaction_cost_bps=cost_bps)
    return StrategyResult(
        symbol=symbol,
        params=params,
        sharpe=result.sharpe,
        cagr=result.cagr,
        max_drawdown=result.max_drawdown,
        win_rate=result.win_rate,
        n_trades=result.n_trades,
        total_return=result.total_return,
        yearly_sharpe=_yearly_sharpe(result.returns),
        top3_day_concentration=_top3_day_concentration(result.returns),
        equity_curve=result.equity_curve,
    )


def buy_and_hold_benchmark(prices: pd.Series, symbol: str) -> StrategyResult:
    """Referenz: Eine Strategie, die Buy-and-Hold nicht schlägt, ist wertlos -
    dann kann man einfach halten und spart sich Gebühren und Komplexität."""
    signals = pd.Series(1, index=prices.index)
    return evaluate(signals, prices, symbol, {"strategy": "buy_and_hold"}, cost_bps=0.0)
