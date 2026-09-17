"""Strategie-Grid-Search: mehrere Assets × mehrere Parameter-Kombinationen,
alles walk-forward validiert, Ergebnis als JSON für das Dashboard.

WICHTIG gegen Data-Dredging: Wir zeigen ALLE getesteten Kombinationen,
nicht nur die beste. Wer nur die beste von 50 Kombinationen zeigt, betreibt
Rosinenpickerei - genau das, was einen Backtest wertlos macht.
"""
from __future__ import annotations

import itertools
import json
import logging
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

from src.backtest.engine import run_backtest, signals_from_probabilities
from src.data.fetch import fetch_ohlcv, to_yahoo_symbol
from src.features.engineer import make_dataset
from src.models.train import _feature_columns, walk_forward_splits
from src.utils.config import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("grid_search")

SYMBOLS = ["BTC/USD", "ETH/USD", "SOL/USD", "DOGE/USD", "SHIB/USD"]
LABEL_HORIZONS = [3, 5, 10]
THRESHOLDS = [0.50, 0.55, 0.60]

# DOGE/SHIB sind bewusst gewählt: liquide, an regulierten Börsen gelistet,
# echte Kurshistorie via yfinance verfügbar - NICHT die illiquiden,
# rug-pull-gefährdeten DEX-Neuerscheinungen. Das ist der verantwortbare
# Teil des "Memecoin"-Spektrums.


def oos_probabilities(dataset: pd.DataFrame, n_splits: int, min_train_bars: int) -> pd.Series:
    feature_cols = _feature_columns(dataset)
    oos = pd.Series(index=dataset.index, dtype=float)
    for train_idx, test_idx in walk_forward_splits(len(dataset), n_splits, min_train_bars):
        train = dataset.iloc[list(train_idx)]
        test = dataset.iloc[list(test_idx)]
        if test.empty or train["label"].nunique() < 2:
            continue
        model = GradientBoostingClassifier(random_state=42)
        model.fit(train[feature_cols], train["label"])
        oos.iloc[list(test_idx)] = model.predict_proba(test[feature_cols])[:, 1]
    return oos.dropna()


def run_grid(config: dict) -> list[dict]:
    data_cfg = config["data"]
    feat_cfg = config["features"]
    model_cfg = config["model"]

    results = []
    for symbol in SYMBOLS:
        yahoo_symbol = to_yahoo_symbol(symbol)
        try:
            raw = fetch_ohlcv(yahoo_symbol, history_days=data_cfg["history_days"], cache_dir=data_cfg["cache_dir"])
        except Exception:
            logger.exception("%s: Datenabruf fehlgeschlagen, überspringe.", symbol)
            continue

        for horizon in LABEL_HORIZONS:
            dataset = make_dataset(
                raw,
                return_horizons=feat_cfg["return_horizons"],
                rsi_period=feat_cfg["rsi_period"],
                sma_windows=feat_cfg["sma_windows"],
                volatility_window=feat_cfg["volatility_window"],
                label_horizon=horizon,
            )
            try:
                oos_proba = oos_probabilities(
                    dataset, n_splits=model_cfg["walk_forward_splits"], min_train_bars=model_cfg["min_train_bars"]
                )
            except Exception:
                logger.exception("%s h=%d: Walk-Forward fehlgeschlagen, überspringe.", symbol, horizon)
                continue

            if oos_proba.empty:
                continue

            aligned_prices = raw["close"].reindex(oos_proba.index)

            for threshold in THRESHOLDS:
                signals = signals_from_probabilities(oos_proba, threshold=threshold)
                result = run_backtest(aligned_prices, signals)
                results.append(
                    {
                        "symbol": symbol,
                        "label_horizon": horizon,
                        "threshold": threshold,
                        "sharpe": round(result.sharpe, 3),
                        "cagr": round(result.cagr, 4),
                        "max_drawdown": round(result.max_drawdown, 4),
                        "win_rate": round(result.win_rate, 3),
                        "n_trades": result.n_trades,
                        "total_return": round(result.total_return, 4),
                        "equity_curve": {
                            str(ts.date()): round(float(v), 2)
                            for ts, v in result.equity_curve.iloc[::max(1, len(result.equity_curve) // 60)].items()
                        },
                    }
                )
                logger.info(
                    "%s h=%d thr=%.2f: Sharpe=%.3f DD=%.2f%% WinRate=%.2f%% Trades=%d",
                    symbol, horizon, threshold, result.sharpe, result.max_drawdown * 100,
                    result.win_rate * 100, result.n_trades,
                )

    return results


if __name__ == "__main__":
    cfg = load_config()
    out = run_grid(cfg)
    out_path = pathlib.Path("artifacts/research/grid_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    logger.info("Fertig: %d Kombinationen getestet, Ergebnis in %s", len(out), out_path)
