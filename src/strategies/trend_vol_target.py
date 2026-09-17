"""Trendfolge mit Volatilitäts-Targeting (Managed-Futures-/CTA-Logik).

These
-----
Der Edge kommt NICHT aus besserer Vorhersage, sondern aus Risikomanagement:
in ruhigen Trendphasen gross, in turbulenten Phasen klein positioniert sein
und Verluste früh begrenzen. Die Trefferquote darf niedrig sein, solange die
Gewinner deutlich grösser sind als die Verlierer.

Aufbau
------
signal[t] = trend_strength[t] * vol_scale[t]   (danach auf [0, 1] begrenzt)

1. `trend_strength` in [0, 1] - je nach `trend_mode`:
   - "price_ma":   Close über SMA(slow)                     -> 0/1
   - "ma_cross":   SMA(fast) über SMA(slow)                 -> 0/1
   - "tsmom":      Anteil positiver Renditen über 3/6/12 M  -> 0, 1/3, 2/3, 1
   - "donchian":   Donchian-Ausbruch long, Donchian-Unterkante flat (stateful)
   - "none":       konstant 1  (ABLATION: nur Vol-Targeting, immer long)
2. `vol_scale`:
   - "off":        konstant 1  (ABLATION: nur Trendsignal, binär)
   - "target":     target_vol / realisierte annualisierte Vol, gecappt auf
                   `max_weight` (<= 1, kein Hebel)

Die zwei ABLATIONS-Schalter (`trend_mode="none"` bzw. `vol_mode="off"`) sind
bewusst Teil des Parameterraums: nur so lässt sich messen, welche Komponente
den Mehrwert liefert.

Turnover
--------
Kontinuierliche Gewichte erzeugen täglich Positionsänderungen und damit
Kosten (Engine: Kosten proportional zu |Positionsänderung|). `rebalance_band`
hält das Gewicht fest, solange die Zieländerung kleiner als die Bandbreite
ist - das drückt den Turnover deutlich, ohne die Sizing-Logik zu verfälschen.

Look-Ahead
----------
Alle Kennzahlen sind trailing (`rolling`/`ewm`, min_periods gesetzt) und
nutzen ausschliesslich Bars <= t. Donchian-Kanalgrenzen zusätzlich mit
`.shift(1)`, damit der Ausbruchsbar sein eigenes Hoch nicht in die Schwelle
einbringt. Kein `.shift(-n)`, kein zentriertes Fenster. Die Ausführung
verzögert die Engine ohnehin um einen Bar.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252

TSMOM_LOOKBACKS = (63, 126, 252)  # ~3 / 6 / 12 Monate


def _realized_vol(close: pd.Series, window: int, method: str = "rolling") -> pd.Series:
    """Annualisierte realisierte Volatilität aus Tagesrenditen (nur Vergangenheit)."""
    ret = close.pct_change()
    if method == "ewm":
        vol = ret.ewm(span=window, min_periods=window).std()
    else:
        vol = ret.rolling(window, min_periods=window).std()
    return vol * np.sqrt(TRADING_DAYS_PER_YEAR)


def _trend_strength(
    df: pd.DataFrame,
    trend_mode: str,
    fast_ma: int,
    slow_ma: int,
    donchian_entry: int,
    donchian_exit: int,
) -> pd.Series:
    close = df["close"]

    if trend_mode == "none":
        return pd.Series(1.0, index=df.index)

    if trend_mode == "price_ma":
        sma = close.rolling(slow_ma, min_periods=slow_ma).mean()
        return (close > sma).astype(float).where(sma.notna(), 0.0)

    if trend_mode == "ma_cross":
        fast = close.rolling(fast_ma, min_periods=fast_ma).mean()
        slow = close.rolling(slow_ma, min_periods=slow_ma).mean()
        ok = fast.notna() & slow.notna()
        return (fast > slow).astype(float).where(ok, 0.0)

    if trend_mode == "tsmom":
        parts = []
        for lb in TSMOM_LOOKBACKS:
            past = close.shift(lb)
            parts.append((close > past).astype(float).where(past.notna(), np.nan))
        stacked = pd.concat(parts, axis=1)
        # Erst wenn alle Lookbacks verfügbar sind, wird gehandelt.
        return stacked.mean(axis=1).where(stacked.notna().all(axis=1), 0.0)

    if trend_mode == "donchian":
        upper = df["high"].rolling(donchian_entry, min_periods=donchian_entry).max().shift(1)
        lower = df["low"].rolling(donchian_exit, min_periods=donchian_exit).min().shift(1)
        entry = (close > upper).fillna(False).to_numpy()
        exit_ = (close < lower).fillna(False).to_numpy()
        out = np.zeros(len(df))
        in_pos = False
        for i in range(len(df)):
            if in_pos and exit_[i]:
                in_pos = False
            if not in_pos and entry[i]:
                in_pos = True
            out[i] = 1.0 if in_pos else 0.0
        return pd.Series(out, index=df.index)

    raise ValueError(f"Unbekannter trend_mode: {trend_mode!r}")


def _apply_rebalance_band(target: pd.Series, band: float) -> pd.Series:
    """Gewicht nur anpassen, wenn die Zieländerung >= band ist (Turnover-Bremse).

    Ausnahme: ein Ziel von 0 (Trend aus) wird IMMER sofort umgesetzt -
    "cut losses short" darf nicht an einer Turnover-Bremse hängen.
    """
    if band <= 0:
        return target

    values = target.to_numpy(dtype=float)
    held = np.zeros_like(values)
    current = 0.0
    for i, want in enumerate(values):
        if want == 0.0 or abs(want - current) >= band:
            current = want
        held[i] = current
    return pd.Series(held, index=target.index)


def generate_signals(
    df: pd.DataFrame,
    trend_mode: str = "price_ma",
    fast_ma: int = 50,
    slow_ma: int = 200,
    donchian_entry: int = 55,
    donchian_exit: int = 20,
    vol_mode: str = "target",
    target_vol: float = 0.30,
    vol_window: int = 30,
    vol_method: str = "rolling",
    max_weight: float = 1.0,
    rebalance_band: float = 0.10,
    **_ignored,
) -> pd.Series:
    """Positionsgrösse in [0, 1]. Signal für Bar t nutzt nur Daten bis t."""
    df = df.sort_index()
    close = df["close"]

    trend = _trend_strength(df, trend_mode, fast_ma, slow_ma, donchian_entry, donchian_exit)

    if vol_mode == "off":
        vol_scale = pd.Series(1.0, index=df.index)
    elif vol_mode == "target":
        vol = _realized_vol(close, vol_window, vol_method)
        vol_scale = (target_vol / vol).replace([np.inf, -np.inf], np.nan)
        # Vor dem ersten gültigen Vol-Wert: keine Position (nicht blind 1.0).
        vol_scale = vol_scale.fillna(0.0)
    else:
        raise ValueError(f"Unbekannter vol_mode: {vol_mode!r}")

    target = (trend * vol_scale).clip(0.0, min(max_weight, 1.0)).fillna(0.0)
    signal = _apply_rebalance_band(target, rebalance_band)
    return signal.clip(0.0, 1.0).rename("signal")


PARAM_GRID: dict[str, list] = {
    "trend_mode": ["price_ma", "ma_cross", "tsmom", "donchian", "none"],
    "fast_ma": [50],
    "slow_ma": [100, 200],
    "donchian_entry": [55],
    "donchian_exit": [20],
    "vol_mode": ["target", "off"],
    "target_vol": [0.20, 0.30, 0.50],
    "vol_window": [20, 60],
    "vol_method": ["rolling"],
    "max_weight": [1.0],
    "rebalance_band": [0.10],
}
