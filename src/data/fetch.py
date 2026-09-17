"""Historische OHLCV-Daten laden und lokal cachen.

Nutzt yfinance für Equities/Krypto-Tickers im Yahoo-Format (z.B. "BTC-USD").
Kapselt das Caching, damit Trainings- und Backtest-Läufe nicht bei jedem
Aufruf neu vom Netz laden.
"""
from __future__ import annotations

import pathlib
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf


def _cache_path(cache_dir: str, symbol: str, interval: str) -> pathlib.Path:
    safe_symbol = symbol.replace("/", "-")
    return pathlib.Path(cache_dir) / f"{safe_symbol}_{interval}.parquet"


def fetch_ohlcv(
    symbol: str,
    history_days: int,
    interval: str = "1d",
    cache_dir: str = "data_cache",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Lädt OHLCV-Daten für ein Symbol, cached lokal als Parquet.

    symbol: Yahoo-Finance-Notation, z.B. "AAPL" oder "BTC-USD".
    Rückgabe: DataFrame mit Spalten [open, high, low, close, volume], DatetimeIndex.
    """
    path = _cache_path(cache_dir, symbol, interval)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists() and not force_refresh:
        cached = pd.read_parquet(path)
        latest = cached.index.max()
        if latest is not None and latest >= pd.Timestamp.utcnow().tz_localize(None) - timedelta(days=1):
            return cached

    start = (datetime.utcnow() - timedelta(days=history_days)).strftime("%Y-%m-%d")
    yf_interval = "1d" if interval in ("1d", "1Day") else interval
    raw = yf.download(symbol, start=start, interval=yf_interval, progress=False, auto_adjust=True)

    if raw.empty:
        raise ValueError(f"Keine Daten für Symbol {symbol!r} erhalten.")

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
    df.index.name = "timestamp"
    df.to_parquet(path)
    return df


def to_yahoo_symbol(symbol: str) -> str:
    """Konvertiert z.B. "BTC/USD" -> "BTC-USD" fürs Yahoo-Format."""
    return symbol.replace("/", "-")
