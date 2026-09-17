import numpy as np
import pandas as pd

from src.backtest.engine import run_backtest, signals_from_probabilities


def _synthetic_prices(n=100, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    returns = rng.normal(0.0005, 0.01, size=n)
    prices = 100 * (1 + pd.Series(returns, index=dates)).cumprod()
    return prices


def test_flat_signal_yields_zero_return():
    prices = _synthetic_prices()
    signals = pd.Series(0, index=prices.index)
    result = run_backtest(prices, signals)
    assert result.total_return == 0.0
    assert result.n_trades == 0


def test_always_long_tracks_buy_and_hold_minus_costs():
    prices = _synthetic_prices()
    signals = pd.Series(1, index=prices.index)
    result = run_backtest(prices, signals, transaction_cost_bps=0.0)
    buy_and_hold = prices.iloc[-1] / prices.iloc[0] - 1.0
    # Ein Bar Verzögerung (Signal wird erst am Folgetag ausgeführt) => nicht exakt gleich,
    # aber in derselben Größenordnung.
    assert abs(result.total_return - buy_and_hold) < 0.05


def test_max_drawdown_is_non_positive():
    prices = _synthetic_prices(seed=1)
    signals = pd.Series(1, index=prices.index)
    result = run_backtest(prices, signals)
    assert result.max_drawdown <= 0.0


def test_signals_from_probabilities_threshold():
    proba = pd.Series([0.9, 0.5, 0.54, 0.56, 0.1])
    signals = signals_from_probabilities(proba, threshold=0.55)
    assert list(signals) == [1, 0, 0, 1, 0]


def test_win_rate_between_zero_and_one():
    prices = _synthetic_prices(seed=2)
    signals = pd.Series(np.tile([0, 1], 50)[: len(prices)], index=prices.index)
    result = run_backtest(prices, signals)
    assert 0.0 <= result.win_rate <= 1.0
