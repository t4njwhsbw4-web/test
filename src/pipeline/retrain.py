"""Retrain-Orchestrierung: das ist der 'selbständig lernt'-Teil des Bots.

Ablauf pro Symbol:
  1. Neueste Historie laden
  2. Features/Labels bauen
  3. Kandidatenmodell per Walk-Forward validieren (Out-of-Sample!)
  4. Nur wenn der Kandidat das aktuell produktive Modell nachweisbar schlägt
     (oder noch keins existiert), wird er zum neuen Produktionsmodell.

Kein Schritt hier fasst reales Kapital an - das ist strikt getrennt vom
Trading-Loop in run_paper.py.
"""
from __future__ import annotations

import logging
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from src.backtest.engine import run_backtest, signals_from_probabilities
from src.data.fetch import fetch_ohlcv, to_yahoo_symbol
from src.features.engineer import make_dataset
from src.models.train import evaluate_walk_forward, load_latest_model, save_model, train_final_model
from src.utils.config import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("retrain")


def _out_of_sample_sharpe(dataset, symbol: str, prices, n_splits: int, min_train_bars: int) -> float:
    """Schätzt die Out-of-Sample-Performance über die Walk-Forward-Folds,
    indem die Testfold-Vorhersagen zu einer durchgängigen Signal-Serie
    zusammengesetzt und rückwirkend gebacktestet werden."""
    from src.models.train import _feature_columns, walk_forward_splits
    from sklearn.ensemble import GradientBoostingClassifier
    import pandas as pd

    feature_cols = _feature_columns(dataset)
    oos_proba = pd.Series(index=dataset.index, dtype=float)

    for train_idx, test_idx in walk_forward_splits(len(dataset), n_splits, min_train_bars):
        train = dataset.iloc[list(train_idx)]
        test = dataset.iloc[list(test_idx)]
        if test.empty or train["label"].nunique() < 2:
            continue
        model = GradientBoostingClassifier(random_state=42)
        model.fit(train[feature_cols], train["label"])
        oos_proba.iloc[list(test_idx)] = model.predict_proba(test[feature_cols])[:, 1]

    oos_proba = oos_proba.dropna()
    signals = signals_from_probabilities(oos_proba)
    aligned_prices = prices.reindex(oos_proba.index)
    result = run_backtest(aligned_prices, signals)
    return result.sharpe


def retrain_symbol(symbol: str, config: dict) -> dict:
    data_cfg = config["data"]
    feat_cfg = config["features"]
    model_cfg = config["model"]

    yahoo_symbol = to_yahoo_symbol(symbol)
    raw = fetch_ohlcv(
        yahoo_symbol,
        history_days=data_cfg["history_days"],
        cache_dir=data_cfg["cache_dir"],
    )
    dataset = make_dataset(
        raw,
        return_horizons=feat_cfg["return_horizons"],
        rsi_period=feat_cfg["rsi_period"],
        sma_windows=feat_cfg["sma_windows"],
        volatility_window=feat_cfg["volatility_window"],
        label_horizon=feat_cfg["label_horizon"],
    )

    wf_result = evaluate_walk_forward(
        dataset,
        n_splits=model_cfg["walk_forward_splits"],
        min_train_bars=model_cfg["min_train_bars"],
    )
    logger.info(
        "%s: Walk-Forward mean_accuracy=%.3f mean_auc=%.3f (%d Folds)",
        symbol, wf_result.mean_accuracy, wf_result.mean_auc, len(wf_result.folds),
    )

    candidate_sharpe = _out_of_sample_sharpe(
        dataset, symbol, raw["close"],
        n_splits=model_cfg["walk_forward_splits"],
        min_train_bars=model_cfg["min_train_bars"],
    )
    logger.info("%s: Out-of-Sample Sharpe (Kandidat) = %.3f", symbol, candidate_sharpe)

    try:
        current_model = load_latest_model(model_cfg["artifact_dir"], symbol)
        current_proba = current_model.predict_proba(dataset.drop(columns=["label"]))[:, 1]
        import pandas as pd
        current_signals = signals_from_probabilities(pd.Series(current_proba, index=dataset.index))
        current_result = run_backtest(raw["close"].reindex(dataset.index), current_signals)
        current_sharpe = current_result.sharpe
    except FileNotFoundError:
        current_sharpe = float("-inf")

    improvement = candidate_sharpe - current_sharpe
    should_promote = improvement >= model_cfg["promote_min_sharpe_improvement"] or current_sharpe == float("-inf")

    if not should_promote:
        logger.info(
            "%s: Kandidat NICHT befördert (Sharpe %.3f vs. aktuell %.3f, "
            "Verbesserung %.3f < Schwelle %.3f). Produktionsmodell bleibt unverändert.",
            symbol, candidate_sharpe, current_sharpe, improvement, model_cfg["promote_min_sharpe_improvement"],
        )
        return {
            "symbol": symbol,
            "promoted": False,
            "candidate_sharpe": candidate_sharpe,
            "current_sharpe": current_sharpe,
        }

    final_model = train_final_model(dataset)
    path = save_model(final_model, model_cfg["artifact_dir"], symbol)
    logger.info("%s: Neues Produktionsmodell gespeichert unter %s", symbol, path)

    return {
        "symbol": symbol,
        "promoted": True,
        "candidate_sharpe": candidate_sharpe,
        "current_sharpe": current_sharpe,
        "model_path": str(path),
    }


def run_retrain_cycle(config: dict) -> list[dict]:
    universe_cfg = config["universe"]
    symbols = universe_cfg["symbols"][universe_cfg["mode"]]

    results = []
    for symbol in symbols:
        try:
            results.append(retrain_symbol(symbol, config))
        except Exception:
            logger.exception("%s: Retrain fehlgeschlagen, Symbol wird übersprungen.", symbol)
    return results


if __name__ == "__main__":
    cfg = load_config()
    run_retrain_cycle(cfg)
