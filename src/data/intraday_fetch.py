"""Intraday-Klines von der oeffentlichen Binance-Data-API (kein API-Key noetig).

api.binance.com antwortet aus dieser Umgebung mit HTTP 451 (geo-restricted).
data-api.binance.vision spricht dieselbe API und ist erreichbar - dieser Host
ist ausdruecklich fuer reine Datenabfragen gedacht.

Historie: 15m-Bars fuer BTCUSDT/ETHUSDT reichen zurueck bis 2017-08-17,
also ~8 Jahre bzw. ~280k Bars pro Symbol - deutlich mehr als die ~60 Tage,
die yfinance fuer feine Intervalle liefert.
"""
from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

import pandas as pd

BASE_URL = "https://data-api.binance.vision/api/v3/klines"
MAX_LIMIT = 1000
CACHE_DIR = Path(__file__).resolve().parents[2] / "data_cache" / "intraday"

_INTERVAL_MS = {
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
}

_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "n_trades", "taker_buy_base", "taker_buy_quote", "ignore",
]


def _get_json(url: str, retries: int = 5) -> list:
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "research/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode())
        except Exception as exc:  # noqa: BLE001 - Netzfehler sind hier erwartbar
            last_err = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Abruf fehlgeschlagen: {url}") from last_err


def fetch_klines(
    symbol: str,
    interval: str = "15m",
    start: str = "2017-08-01",
    end: str | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Laedt die volle Klines-Historie paginiert und cacht sie als Parquet."""
    if interval not in _INTERVAL_MS:
        raise ValueError(f"Unbekanntes Intervall: {interval}")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{symbol}_{interval}.parquet"
    if use_cache and cache_path.exists():
        return pd.read_parquet(cache_path)

    step = _INTERVAL_MS[interval]
    cursor = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    end_ms = int(
        (pd.Timestamp(end, tz="UTC") if end else pd.Timestamp.now(tz="UTC")).timestamp() * 1000
    )

    chunks: list[list] = []
    while cursor < end_ms:
        url = (
            f"{BASE_URL}?symbol={symbol}&interval={interval}"
            f"&startTime={cursor}&endTime={end_ms}&limit={MAX_LIMIT}"
        )
        batch = _get_json(url)
        if not batch:
            break
        chunks.extend(batch)
        new_cursor = int(batch[-1][0]) + step
        if new_cursor <= cursor:  # kein Fortschritt -> Abbruch statt Endlosschleife
            break
        cursor = new_cursor
        time.sleep(0.12)  # freundlich zum Endpunkt, Gewicht ist unkritisch

    if not chunks:
        raise RuntimeError(f"Keine Daten fuer {symbol} {interval}")

    df = pd.DataFrame(chunks, columns=_COLUMNS)
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    numeric = ["open", "high", "low", "close", "volume", "quote_volume", "n_trades"]
    df[numeric] = df[numeric].astype(float)
    df = (
        df[["timestamp", *numeric]]
        .drop_duplicates(subset="timestamp")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    df.to_parquet(cache_path, index=False)
    return df


def load_panel(
    symbols: tuple[str, ...] = ("BTCUSDT", "ETHUSDT"),
    interval: str = "15m",
    field: str = "close",
) -> pd.DataFrame:
    """Panel (Index = UTC-Zeitstempel, Spalten = Symbole) eines Kline-Feldes."""
    series = {}
    for sym in symbols:
        df = fetch_klines(sym, interval=interval)
        series[sym] = df.set_index("timestamp")[field]
    return pd.DataFrame(series).sort_index()


if __name__ == "__main__":
    for sym in ("BTCUSDT", "ETHUSDT"):
        d = fetch_klines(sym, "15m")
        print(f"{sym}: {len(d):,} Bars  {d['timestamp'].min()} .. {d['timestamp'].max()}")
