"""Vektorisierter Backtest: Signale -> Positionen -> Equity-Kurve -> Kennzahlen.

Bewusst simpel (long/flat, keine Shorts, tägliches Rebalancing) - Komplexität
hier bringt in einer frühen Phase nur zusätzliche Bugs, keinen Erkenntnisgewinn.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


@dataclasses.dataclass
class BacktestResult:
    equity_curve: pd.Series
    returns: pd.Series
    total_return: float
    cagr: float
    sharpe: float
    max_drawdown: float
    win_rate: float
    n_trades: int


def run_backtest(
    prices: pd.Series,
    signals: pd.Series,
    starting_cash: float = 10_000.0,
    transaction_cost_bps: float = 5.0,
) -> BacktestResult:
    """signals: 1 = long, 0 = flat, ausgerichtet auf denselben Index wie prices.

    WICHTIG: signals[t] muss bereits die Entscheidung zu Handelsschluss t
    repräsentieren, ausgeführt zu Open t+1 (oder Close t+1) - der Aufrufer
    ist dafür verantwortlich, das Signal NICHT mit Daten aus t+1 zu berechnen.
    Hier wird zur Vereinfachung mit einem Bar Verzögerung zwischen Signal und
    Ausführung gerechnet (signals.shift(1)).
    """
    prices = prices.sort_index()
    signals = signals.reindex(prices.index).fillna(0).clip(0, 1)

    executed_position = signals.shift(1).fillna(0)
    asset_returns = prices.pct_change().fillna(0)

    strategy_returns = executed_position * asset_returns

    position_changes = executed_position.diff().abs().fillna(executed_position.abs())
    costs = position_changes * (transaction_cost_bps / 10_000.0)
    strategy_returns = strategy_returns - costs

    equity_curve = starting_cash * (1 + strategy_returns).cumprod()

    total_return = equity_curve.iloc[-1] / starting_cash - 1.0
    n_periods = len(strategy_returns)
    years = max(n_periods / TRADING_DAYS_PER_YEAR, 1e-9)
    cagr = (1 + total_return) ** (1 / years) - 1 if total_return > -1 else -1.0

    std = strategy_returns.std()
    sharpe = (strategy_returns.mean() / std) * np.sqrt(TRADING_DAYS_PER_YEAR) if std > 0 else 0.0

    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1.0
    max_drawdown = drawdown.min()

    n_trades, win_rate = _per_trade_stats(executed_position, strategy_returns)

    return BacktestResult(
        equity_curve=equity_curve,
        returns=strategy_returns,
        total_return=float(total_return),
        cagr=float(cagr),
        sharpe=float(sharpe),
        max_drawdown=float(max_drawdown),
        win_rate=win_rate,
        n_trades=n_trades,
    )


def _per_trade_stats(position: pd.Series, strategy_returns: pd.Series) -> tuple[int, float]:
    """Gruppiert zusammenhängende Long-Phasen zu einzelnen Trades und
    bestimmt, wie viele davon in Summe profitabel waren."""
    in_position = position > 0
    # shift(1, fill_value=False) hält den bool-dtype. Ein blankes shift(1)
    # erzeugt object-dtype (NaN), und dort negiert ~ arithmetisch statt
    # logisch (~False == -1, truthy) - dann zählt jeder Bar als neuer Trade.
    previously_in_position = in_position.shift(1, fill_value=False)
    trade_id = (in_position & ~previously_in_position).cumsum()
    trade_id = trade_id.where(in_position)

    if trade_id.dropna().empty:
        return 0, 0.0

    trade_pnl = strategy_returns.groupby(trade_id).apply(lambda r: (1 + r).prod() - 1)
    n_trades = len(trade_pnl)
    win_rate = float((trade_pnl > 0).sum() / n_trades) if n_trades > 0 else 0.0
    return n_trades, win_rate


def signals_from_probabilities(proba_up: pd.Series, threshold: float = 0.55) -> pd.Series:
    """Konvertiert Modell-Wahrscheinlichkeiten in binäre long/flat-Signale.

    threshold > 0.5: Modell muss sich seiner Sache einigermaßen sicher sein,
    bevor überhaupt eine Position eingegangen wird - reduziert Overtrading
    bei Wahrscheinlichkeiten nahe dem Münzwurf.
    """
    return (proba_up >= threshold).astype(int)
