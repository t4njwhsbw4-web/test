"""Walk-Forward-Training: nie auf Zukunftsdaten testen.

Ein normales k-fold Cross-Validation ist bei Zeitreihen falsch, weil es dem
Modell erlaubt, aus der "Zukunft" zu lernen und in die "Vergangenheit" zu
testen. Walk-Forward simuliert stattdessen, wie ein Retrain in der Praxis
abläuft: immer nur auf Daten trainieren, die zeitlich vor dem Testfenster
liegen.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import pathlib

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

FEATURE_COLUMNS_EXCLUDE = {"label"}


@dataclasses.dataclass
class FoldResult:
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    accuracy: float
    auc: float
    n_test: int


@dataclasses.dataclass
class WalkForwardResult:
    folds: list[FoldResult]
    mean_accuracy: float
    mean_auc: float


def walk_forward_splits(n_rows: int, n_splits: int, min_train_bars: int):
    """Erzeugt (train_idx, test_idx) Indexpaare in chronologischer Reihenfolge."""
    usable = n_rows - min_train_bars
    if usable <= n_splits:
        raise ValueError("Zu wenig Datenpunkte für die gewünschte Anzahl Walk-Forward-Splits.")
    fold_size = usable // n_splits
    for i in range(n_splits):
        train_end = min_train_bars + i * fold_size
        test_end = train_end + fold_size if i < n_splits - 1 else n_rows
        yield range(0, train_end), range(train_end, test_end)


def _feature_columns(dataset: pd.DataFrame) -> list[str]:
    return [c for c in dataset.columns if c not in FEATURE_COLUMNS_EXCLUDE]


def evaluate_walk_forward(dataset: pd.DataFrame, n_splits: int, min_train_bars: int) -> WalkForwardResult:
    feature_cols = _feature_columns(dataset)
    folds: list[FoldResult] = []

    for train_idx, test_idx in walk_forward_splits(len(dataset), n_splits, min_train_bars):
        train = dataset.iloc[list(train_idx)]
        test = dataset.iloc[list(test_idx)]
        if test.empty or train["label"].nunique() < 2:
            continue

        model = GradientBoostingClassifier(random_state=42)
        model.fit(train[feature_cols], train["label"])

        preds = model.predict(test[feature_cols])
        proba = model.predict_proba(test[feature_cols])[:, 1]

        acc = accuracy_score(test["label"], preds)
        try:
            auc = roc_auc_score(test["label"], proba)
        except ValueError:
            auc = float("nan")

        folds.append(
            FoldResult(
                train_end=train.index[-1],
                test_start=test.index[0],
                test_end=test.index[-1],
                accuracy=acc,
                auc=auc,
                n_test=len(test),
            )
        )

    if not folds:
        raise RuntimeError("Kein Walk-Forward-Fold konnte ausgewertet werden (zu wenig Daten oder Label-Varianz).")

    mean_acc = sum(f.accuracy for f in folds) / len(folds)
    aucs = [f.auc for f in folds if f.auc == f.auc]  # NaN rausfiltern
    mean_auc = sum(aucs) / len(aucs) if aucs else float("nan")

    return WalkForwardResult(folds=folds, mean_accuracy=mean_acc, mean_auc=mean_auc)


def train_final_model(dataset: pd.DataFrame) -> GradientBoostingClassifier:
    """Trainiert das Produktionsmodell auf dem gesamten verfügbaren Datensatz.

    Wird erst aufgerufen, nachdem evaluate_walk_forward() gezeigt hat, dass
    der Ansatz auf Out-of-Sample-Daten überhaupt funktioniert.
    """
    feature_cols = _feature_columns(dataset)
    model = GradientBoostingClassifier(random_state=42)
    model.fit(dataset[feature_cols], dataset["label"])
    return model


def save_model(model: GradientBoostingClassifier, artifact_dir: str, symbol: str) -> pathlib.Path:
    out_dir = pathlib.Path(artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    safe_symbol = symbol.replace("/", "-")
    path = out_dir / f"{safe_symbol}_{timestamp}.joblib"
    joblib.dump(model, path)

    latest_path = out_dir / f"{safe_symbol}_latest.joblib"
    joblib.dump(model, latest_path)
    return path


def load_latest_model(artifact_dir: str, symbol: str) -> GradientBoostingClassifier:
    safe_symbol = symbol.replace("/", "-")
    path = pathlib.Path(artifact_dir) / f"{safe_symbol}_latest.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Kein trainiertes Modell gefunden für {symbol!r} unter {path}.")
    return joblib.load(path)
