"""Mean-Reversion-Strategie: Rückkehr kurzfristiger Überdehnungen zum Mittel.

THESE
-----
Kurzfristige Preisabweichungen vom rollierenden Mittel entstehen zu einem
relevanten Teil aus temporären Orderflow-Ungleichgewichten (Liquidationen,
erzwungene Verkäufe, Nachrichtenschocks). Liquiditätsanbieter werden dafür
bezahlt, diese Ungleichgewichte zu absorbieren - sie kaufen ins fallende
Messer und verlangen dafür eine Prämie. Wer systematisch mit ihnen kauft,
vereinnahmt einen Teil dieser Prämie.

UMSETZUNG
---------
z-Score des Close gegen ein rollierendes Mittel (mathematisch identisch zur
Bollinger-Band-Rückkehr: entry_z = -2.0 entspricht dem unteren 2-Sigma-Band).
Long, wenn der z-Score unter ``entry_z`` fällt; flat, wenn er wieder über
``exit_z`` steigt, ein ``max_hold``-Limit greift oder ein optionaler
Trendfilter die Position verbietet.

Optionale Zusatzfilter:
- ``trend_window``: Long nur, wenn Close über der langen SMA liegt. Idee:
  Mean Reversion in Aufwärtstrends ist ein Dip-Kauf, in Abwärtstrends ein
  fallendes Messer.
- ``rsi_confirm``: zusätzlich RSI(14) unter Schwelle - zweite, unabhängige
  Bestätigung der Überdehnung.

KEIN LOOK-AHEAD
---------------
Alle Fenster sind ausschliesslich nachlaufend (``rolling(...)`` ohne
``center=True``, kein ``shift(-n)``). Das Signal für Bar t verwendet nur
Daten bis einschliesslich Bar t; der Backtest führt es mit
``signals.shift(1)`` zu Bar t+1 aus.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Getesteter Parameter-Raum, als Vereinigung zweier Grid-Läufe auf den
# Dev-Daten aller RESEARCH_SYMBOLS:
#   Runde 1 (192 Kombinationen): engere Einstiege, enge Ausstiege, Trendfilter
#     lookback [10,20,40] x entry_z [-1.0,-1.5,-2.0,-2.5] x exit_z [-0.5,0.0]
#     x trend_window [None,200] x max_hold [10,20] x rsi_confirm [None,30]
#   Runde 2 (162 Kombinationen): höhere Kapitalauslastung, weitere Ausstiege
#     lookback [5,10,20] x entry_z [-0.5,-1.0,-1.5] x exit_z [0.5,1.0]
#     x trend_window [None] x max_hold [20,40,60] x rsi_confirm [None,40,50]
#
# ERGEBNIS: KEIN Edge gegenüber Buy-and-Hold. Der beste mittlere Sharpe über
# alle 6 Symbole war 0.586 (lookback=5, entry_z=-1.5, exit_z=1.0, max_hold=40,
# kein Filter) gegen 0.845 für Buy-and-Hold. Keine der 354 Kombinationen
# schlug Buy-and-Hold auf mehr als 1 von 6 Symbolen.
PARAM_GRID: dict[str, list] = {
    "lookback": [5, 10, 20, 40],
    "entry_z": [-0.5, -1.0, -1.5, -2.0, -2.5],
    "exit_z": [-0.5, 0.0, 0.5, 1.0],
    "trend_window": [None, 200],
    "max_hold": [10, 20, 40, 60],
    "rsi_confirm": [None, 30.0, 40.0, 50.0],
}


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder-RSI, rein nachlaufend."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    # avg_loss == 0 -> ausschliesslich Gewinne -> RSI 100
    return rsi.where(avg_loss > 0, 100.0)


def zscore(close: pd.Series, lookback: int) -> pd.Series:
    """z-Score des Close gegen die nachlaufende SMA/Std über ``lookback`` Bars."""
    mean = close.rolling(lookback, min_periods=lookback).mean()
    std = close.rolling(lookback, min_periods=lookback).std()
    return (close - mean) / std.replace(0.0, np.nan)


def generate_signals(
    df: pd.DataFrame,
    lookback: int = 20,
    entry_z: float = -2.0,
    exit_z: float = 0.0,
    trend_window: int | None = None,
    max_hold: int = 10,
    rsi_confirm: float | None = None,
    rsi_period: int = 14,
) -> pd.Series:
    """Erzeugt long/flat-Signale (1 = long, 0 = flat).

    Parameters
    ----------
    df : OHLCV-DataFrame mit Spalte ``close`` und DatetimeIndex.
    lookback : Fenster für Mittel und Standardabweichung des z-Scores.
    entry_z : Einstiegsschwelle (negativ), z.B. -2.0 = unteres 2-Sigma-Band.
    exit_z : Ausstiegsschwelle; Position wird geschlossen, sobald der
        z-Score diesen Wert erreicht (0.0 = Rückkehr zum Mittel).
    trend_window : Wenn gesetzt, Long nur bei Close > SMA(trend_window).
    max_hold : Maximale Haltedauer in Bars (Zwangsausstieg).
    rsi_confirm : Wenn gesetzt, Einstieg zusätzlich nur bei RSI < Schwelle.
    """
    close = df["close"].astype(float)

    z = zscore(close, lookback)
    entry_ok = (z <= entry_z).to_numpy()
    exit_ok = (z >= exit_z).to_numpy()

    if trend_window is not None:
        trend = close.rolling(trend_window, min_periods=trend_window).mean()
        regime_ok = (close > trend).to_numpy()
        # Solange der Trendfilter noch keinen Wert hat: kein Einstieg.
        regime_ok = np.where(np.isnan(trend.to_numpy()), False, regime_ok)
    else:
        regime_ok = np.ones(len(close), dtype=bool)

    if rsi_confirm is not None:
        rsi = _rsi(close, rsi_period)
        rsi_ok = (rsi < rsi_confirm).to_numpy()
        rsi_ok = np.where(np.isnan(rsi.to_numpy()), False, rsi_ok)
    else:
        rsi_ok = np.ones(len(close), dtype=bool)

    # NaN-Phasen (Warmup) sind kein Einstieg und kein Ausstiegssignal.
    z_valid = ~np.isnan(z.to_numpy())
    entry_ok = entry_ok & z_valid & regime_ok & rsi_ok
    exit_ok = exit_ok & z_valid

    n = len(close)
    out = np.zeros(n, dtype=np.int8)
    position = 0
    bars_held = 0

    for i in range(n):
        if position == 1:
            bars_held += 1
            if exit_ok[i] or bars_held >= max_hold or not regime_ok[i]:
                position = 0
                bars_held = 0
        if position == 0 and entry_ok[i]:
            position = 1
            bars_held = 0
        out[i] = position

    return pd.Series(out, index=close.index, name="signal").astype(int)
