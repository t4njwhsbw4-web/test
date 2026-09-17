"""End-to-End-Test der Feature/Training/Backtest-Kette auf synthetischen
OHLCV-Daten - läuft ohne Netzwerkzugriff, damit CI nicht von yfinance abhängt."""
import numpy as np
import pandas as pd

from src.features.engineer import make_dataset
from src.models.train import evaluate_walk_forward, train_final_model
from src.backtest.engine import run_backtest, signals_from_probabilities


def _synthetic_ohlcv(n=400, seed=7):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", periods=n, freq="D")
    returns = rng.normal(0.0003, 0.015, size=n)
    close = 100 * (1 + pd.Series(returns, index=dates)).cumprod()
    high = close * (1 + rng.uniform(0, 0.01, size=n))
    low = close * (1 - rng.uniform(0, 0.01, size=n))
    open_ = close.shift(1).fillna(close.iloc[0])
    volume = rng.integers(1_000_000, 5_000_000, size=n)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=dates)


def test_dataset_has_no_nans_and_binary_labels():
    raw = _synthetic_ohlcv()
    dataset = make_dataset(
        raw, return_horizons=[1, 5], rsi_period=14, sma_windows=[10, 20], volatility_window=20, label_horizon=5
    )
    assert not dataset.isna().any().any()
    assert set(dataset["label"].unique()) <= {0, 1}
    assert len(dataset) > 0


def test_walk_forward_produces_folds_with_reasonable_metrics():
    raw = _synthetic_ohlcv(n=500)
    dataset = make_dataset(
        raw, return_horizons=[1, 5], rsi_period=14, sma_windows=[10, 20], volatility_window=20, label_horizon=5
    )
    result = evaluate_walk_forward(dataset, n_splits=4, min_train_bars=200)
    assert len(result.folds) > 0
    assert 0.0 <= result.mean_accuracy <= 1.0


def test_trained_model_predicts_valid_probabilities():
    raw = _synthetic_ohlcv(n=500)
    dataset = make_dataset(
        raw, return_horizons=[1, 5], rsi_period=14, sma_windows=[10, 20], volatility_window=20, label_horizon=5
    )
    model = train_final_model(dataset)
    proba = model.predict_proba(dataset.drop(columns=["label"]))[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()

    signals = signals_from_probabilities(pd.Series(proba, index=dataset.index))
    result = run_backtest(raw["close"].reindex(dataset.index), signals)
    assert result.equity_curve.iloc[0] > 0
    assert np.isfinite(result.sharpe)
