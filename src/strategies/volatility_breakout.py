"""Volatility-Breakout: Ausbruch aus einer Kompressionsphase.

These: Information diffundiert langsam. Eine Phase niedriger Volatilität
(enge Handelsspanne) bedeutet, dass wenig neue Information verarbeitet wird.
Bricht der Kurs aus dieser Spanne aus, signalisiert das eintreffende neue
Information - und die Anpassung des Preises braucht mehrere Bars. Wer den
Ausbruch handelt, verdient an der Anpassungsbewegung.

Umsetzung
---------
1. Squeeze-Filter: die realisierte Volatilität (bzw. die normierte Donchian-
   Kanalbreite) muss im unteren Perzentil ihrer eigenen Historie liegen.
2. Entry: Close > höchstes High der letzten `entry_lookback` Bars
   (Donchian-Oberkante), optional zusätzlich um `atr_mult` * ATR überschritten.
3. Exit: Close < tiefstes Low der letzten `exit_lookback` Bars (Donchian-
   Unterkante) ODER nach `max_hold` Bars, falls gesetzt.

Look-Ahead
----------
Alle Kanal-, ATR- und Perzentil-Werte werden mit `.shift(1)` gebildet, also
ausschliesslich aus Bars <= t-1, und nur mit dem Close von Bar t verglichen.
Das Signal für Bar t nutzt damit nur Daten bis einschliesslich t; die
Ausführung erfolgt im Engine mit einem Bar Verzögerung. Kein `.shift(-n)`,
kein zentriertes Fenster.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _atr(df: pd.DataFrame, window: int) -> pd.Series:
    """Average True Range (Wilder-Bestandteile, einfaches rolling mean)."""
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window, min_periods=window).mean()


def _squeeze_mask(
    df: pd.DataFrame,
    atr: pd.Series,
    squeeze_window: int,
    squeeze_pct: float,
    squeeze_metric: str,
) -> pd.Series:
    """True, wenn die Vola im unteren `squeeze_pct`-Perzentil ihrer Historie liegt.

    Der Perzentil-Rang wird über ein trailing Fenster gebildet (rolling rank,
    also nur Vergangenheit), anschliessend um einen Bar verschoben, damit der
    Ausbruchsbar selbst die Kompressionsmessung nicht verfälscht.
    """
    if squeeze_pct >= 1.0:
        return pd.Series(True, index=df.index)

    if squeeze_metric == "atr":
        vol = atr / df["close"]
    elif squeeze_metric == "bandwidth":
        # Bollinger-/Donchian-artige Kanalbreite, preisnormiert
        upper = df["high"].rolling(squeeze_window // 4 or 5, min_periods=3).max()
        lower = df["low"].rolling(squeeze_window // 4 or 5, min_periods=3).min()
        vol = (upper - lower) / df["close"]
    elif squeeze_metric == "std":
        vol = df["close"].pct_change().rolling(20, min_periods=20).std()
    else:
        raise ValueError(f"Unbekannte squeeze_metric: {squeeze_metric!r}")

    rank = vol.rolling(squeeze_window, min_periods=squeeze_window // 2).rank(pct=True)
    return (rank.shift(1) <= squeeze_pct).fillna(False)


def generate_signals(
    df: pd.DataFrame,
    entry_lookback: int = 20,
    exit_lookback: int = 10,
    squeeze_window: int = 120,
    squeeze_pct: float = 0.4,
    squeeze_metric: str = "atr",
    atr_window: int = 14,
    atr_mult: float = 0.0,
    max_hold: int = 0,
    **_ignored,
) -> pd.Series:
    """1 = long, 0 = flat. Signal für Bar t nutzt nur Daten bis t.

    max_hold = 0 bedeutet: kein Zeit-Exit, nur der Kanal-Exit.
    """
    df = df.sort_index()
    close = df["close"]

    atr = _atr(df, atr_window)

    # Kanalgrenzen aus BARS BIS t-1 (shift(1)) - der aktuelle Bar darf sein
    # eigenes Hoch nicht in die Ausbruchsschwelle einbringen.
    upper = df["high"].rolling(entry_lookback, min_periods=entry_lookback).max().shift(1)
    lower = df["low"].rolling(exit_lookback, min_periods=exit_lookback).min().shift(1)
    atr_prev = atr.shift(1)

    threshold = upper + atr_mult * atr_prev
    breakout = (close > threshold).fillna(False)
    squeeze = _squeeze_mask(df, atr, squeeze_window, squeeze_pct, squeeze_metric)

    entry = (breakout & squeeze).to_numpy()
    breakdown = (close < lower).fillna(False).to_numpy()

    n = len(df)
    pos = np.zeros(n, dtype=np.int8)
    bars_held = 0
    in_pos = False
    for i in range(n):
        if in_pos:
            bars_held += 1
            if breakdown[i] or (max_hold > 0 and bars_held >= max_hold):
                in_pos = False
                bars_held = 0
        if not in_pos and entry[i]:
            in_pos = True
            bars_held = 0
        pos[i] = 1 if in_pos else 0

    return pd.Series(pos, index=df.index, name="signal").astype(int)


PARAM_GRID: dict[str, list] = {
    "entry_lookback": [10, 20, 40, 55],
    "exit_lookback": [5, 10, 20],
    "squeeze_window": [120],
    "squeeze_pct": [0.3, 0.5, 1.0],
    "squeeze_metric": ["atr", "bandwidth"],
    "atr_window": [14],
    "atr_mult": [0.0, 0.5],
    "max_hold": [0, 20],
}
