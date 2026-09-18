"""Datenbeschaffung fuer die Aufmerksamkeits-Studie (Volumen/Trades als Hype-Proxy).

Warum eine eigene Datei: `src/data/intraday_fetch.py` verwirft beim Parsen die
Felder `taker_buy_base`/`taker_buy_quote` - genau die brauchen wir hier, um
aggressiven Kaufdruck zu messen. Bestehende Dateien duerfen nicht geaendert
werden, darum wird nur `BASE_URL`/`MAX_LIMIT`/`_get_json` importiert und die
Paginierung hier lokal mit allen Feldern wiederholt. Eigener Cache-Ordner
(data_cache/attention/), damit parallel laufende Agenten sich nicht stoeren.
"""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.intraday_fetch import BASE_URL, MAX_LIMIT, _get_json  # noqa: E402

EXCHANGE_INFO_URL = "https://data-api.binance.vision/api/v3/exchangeInfo"
CACHE_DIR = Path(__file__).resolve().parents[2] / "data_cache" / "attention"

_RAW_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "n_trades", "taker_buy_base", "taker_buy_quote", "ignore",
]

_INTERVAL_MS = {
    "15m": 900_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}

# Memecoins, die auf Binance mit USDT-Paar handelbar sind. Bewusst breit:
# alte (DOGE/SHIB) und junge Listings (TRUMP/PENGU/PUMP). Dass diese Liste nur
# UEBERLEBENDE enthaelt, ist eine Aufwaertsverzerrung und wird im Bericht
# ausdruecklich als solche benannt.
MEMECOINS = (
    "DOGEUSDT", "SHIBUSDT", "PEPEUSDT", "FLOKIUSDT", "BONKUSDT", "WIFUSDT",
    "BOMEUSDT", "TRUMPUSDT", "PNUTUSDT", "ACTUSDT", "NEIROUSDT", "MEMEUSDT",
    "TURBOUSDT", "PENGUUSDT", "1MBABYDOGEUSDT", "PEOPLEUSDT", "NOTUSDT",
    "DOGSUSDT", "HMSTRUSDT", "CATIUSDT", "ORDIUSDT", "MUBARAKUSDT", "TSTUSDT",
    "BANANAUSDT", "AIXBTUSDT", "ANIMEUSDT", "PUMPUSDT",
)

MAJORS = ("BTCUSDT", "ETHUSDT")


def fetch_klines_full(
    symbol: str,
    interval: str = "1h",
    start: str = "2017-01-01",
    use_cache: bool = True,
) -> pd.DataFrame | None:
    """Volle Kline-Historie MIT Volumen-, Trade- und Taker-Feldern."""
    if interval not in _INTERVAL_MS:
        raise ValueError(f"Unbekanntes Intervall: {interval}")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{symbol}_{interval}.parquet"
    if use_cache and cache_path.exists():
        return pd.read_parquet(cache_path)

    step = _INTERVAL_MS[interval]
    cursor = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)

    chunks: list[list] = []
    while cursor < end_ms:
        url = (
            f"{BASE_URL}?symbol={symbol}&interval={interval}"
            f"&startTime={cursor}&endTime={end_ms}&limit={MAX_LIMIT}"
        )
        try:
            batch = _get_json(url, retries=3)
        except RuntimeError:
            return None
        if not batch:
            break
        chunks.extend(batch)
        new_cursor = int(batch[-1][0]) + step
        if new_cursor <= cursor:
            break
        cursor = new_cursor

    if not chunks:
        return None

    df = pd.DataFrame(chunks, columns=_RAW_COLUMNS)
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    numeric = [
        "open", "high", "low", "close", "volume", "quote_volume", "n_trades",
        "taker_buy_base", "taker_buy_quote",
    ]
    df[numeric] = df[numeric].astype(float)
    df = (
        df[["timestamp", *numeric]]
        .drop_duplicates(subset="timestamp")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    df.to_parquet(cache_path, index=False)
    return df


def load_universe(
    symbols: tuple[str, ...], interval: str = "1h", min_bars: int = 2000
) -> dict[str, pd.DataFrame]:
    """Laedt alle Symbole; verwirft zu kurze Historien (Signal braucht Vorlauf)."""
    out: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        df = fetch_klines_full(sym, interval=interval)
        if df is None or len(df) < min_bars:
            print(f"  uebersprungen: {sym} ({0 if df is None else len(df)} Bars)")
            continue
        out[sym] = df.set_index("timestamp")
        print(f"  {sym}: {len(df):,} Bars  {df['timestamp'].min().date()} .. {df['timestamp'].max().date()}")
    return out


def trading_usdt_symbols() -> list[str]:
    req = urllib.request.Request(EXCHANGE_INFO_URL, headers={"User-Agent": "research/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        info = json.loads(resp.read().decode())
    return [
        s["symbol"] for s in info["symbols"]
        if s["status"] == "TRADING" and s["quoteAsset"] == "USDT"
    ]


if __name__ == "__main__":
    print("Memecoins:")
    memes = load_universe(MEMECOINS, "1h")
    print("Majors:")
    majors = load_universe(MAJORS, "1h")
    print(f"\n{len(memes)} Memecoins, {len(majors)} Majors geladen.")
