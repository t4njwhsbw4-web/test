"""Feature-Engineering und Label-Erzeugung auf OHLCV-Daten.

Bewusst einfach gehalten: Momentum-, Trend- und Volatilitäts-Features.
Mehr Features != besseres Modell - jedes zusätzliche Feature erhöht die
Overfitting-Gefahr bei begrenzter Trainingshistorie.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def build_features(
    df: pd.DataFrame,
    return_horizons: list[int],
    rsi_period: int,
    sma_windows: list[int],
    volatility_window: int,
) -> pd.DataFrame:
    """Erwartet df mit Spalten [open, high, low, close, volume]."""
    out = pd.DataFrame(index=df.index)
    close = df["close"]

    for h in return_horizons:
        out[f"return_{h}d"] = close.pct_change(h)

    out["rsi"] = _rsi(close, rsi_period)

    for w in sma_windows:
        sma = close.rolling(w).mean()
        out[f"close_over_sma_{w}"] = close / sma - 1.0

    out["volatility"] = close.pct_change().rolling(volatility_window).std()

    volume = df["volume"]
    vol_mean = volume.rolling(volatility_window).mean()
    vol_std = volume.rolling(volatility_window).std().replace(0, np.nan)
    out["volume_zscore"] = ((volume - vol_mean) / vol_std).fillna(0.0)

    macd_fast = close.ewm(span=12, adjust=False).mean()
    macd_slow = close.ewm(span=26, adjust=False).mean()
    macd_line = macd_fast - macd_slow
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    out["macd_hist"] = macd_line - macd_signal

    return out


def build_labels(df: pd.DataFrame, horizon: int) -> pd.Series:
    """Binäres Label: steigt der Close in `horizon` Bars? (1 = ja, 0 = nein)

    WICHTIG: Label nutzt Zukunftsdaten (future return) - beim Training/
    Backtest muss sichergestellt sein, dass diese Zeilen NIE als Feature
    für denselben Zeitpunkt verwendet werden (Look-Ahead-Bias).
    """
    future_return = df["close"].shift(-horizon) / df["close"] - 1.0
    return (future_return > 0).astype(int)


def make_dataset(
    df: pd.DataFrame,
    return_horizons: list[int],
    rsi_period: int,
    sma_windows: list[int],
    volatility_window: int,
    label_horizon: int,
) -> pd.DataFrame:
    """Baut Feature-Matrix + Label, entfernt Zeilen mit NaN (Warmup-Phase
    der rollierenden Fenster + letzte `label_horizon` Zeilen ohne Label)."""
    features = build_features(df, return_horizons, rsi_period, sma_windows, volatility_window)
    labels = build_labels(df, label_horizon)
    dataset = features.copy()
    dataset["label"] = labels
    return dataset.dropna()
