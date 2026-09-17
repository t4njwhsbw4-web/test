"""Daten-Helfer fuer die Memecoin-Untersuchung.

Warum eine eigene Datei: `src/data/intraday_fetch.py` kennt in `_INTERVAL_MS`
nur Intervalle bis "1h", nicht "1d". Bestehende Dateien duerfen hier nicht
geaendert werden, darum wird `_get_json`/`BASE_URL` importiert und die
Tagesbar-Paginierung hier lokal ergaenzt. Cache-Verzeichnis ist getrennt
(data_cache/memecoin/), damit parallel laufende Agenten sich nicht ins
Gehege kommen.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.intraday_fetch import BASE_URL, MAX_LIMIT, _get_json  # noqa: E402

EXCHANGE_INFO_URL = "https://data-api.binance.vision/api/v3/exchangeInfo"
DEPTH_URL = "https://data-api.binance.vision/api/v3/depth"
CACHE_DIR = Path(__file__).resolve().parents[2] / "data_cache" / "memecoin"

_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "n_trades", "taker_buy_base", "taker_buy_quote", "ignore",
]

_DAY_MS = 86_400_000


def fetch_daily(symbol: str, start: str = "2015-01-01", use_cache: bool = True) -> pd.DataFrame | None:
    """Volle Tagesbar-Historie eines Symbols; None wenn es keine Daten gibt."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{symbol}_1d.parquet"
    if use_cache and cache_path.exists():
        return pd.read_parquet(cache_path)

    cursor = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
    chunks: list[list] = []
    while cursor < end_ms:
        url = (
            f"{BASE_URL}?symbol={symbol}&interval=1d"
            f"&startTime={cursor}&endTime={end_ms}&limit={MAX_LIMIT}"
        )
        try:
            batch = _get_json(url, retries=3)
        except RuntimeError:
            return None
        if not batch:
            break
        chunks.extend(batch)
        new_cursor = int(batch[-1][0]) + _DAY_MS
        if new_cursor <= cursor:
            break
        cursor = new_cursor
        time.sleep(0.1)

    if not chunks:
        return None

    df = pd.DataFrame(chunks, columns=_COLUMNS)
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    num = ["open", "high", "low", "close", "volume", "quote_volume", "n_trades"]
    df[num] = df[num].astype(float)
    df = (
        df[["timestamp", *num]]
        .drop_duplicates(subset="timestamp")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    df.to_parquet(cache_path, index=False)
    return df


def fetch_exchange_info(use_cache: bool = True) -> dict:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / "exchange_info.json"
    if use_cache and path.exists():
        return json.loads(path.read_text())
    req = urllib.request.Request(EXCHANGE_INFO_URL, headers={"User-Agent": "research/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode())
    path.write_text(json.dumps(data))
    return data


def fetch_depth(symbol: str, limit: int = 100) -> dict:
    return _get_json(f"{DEPTH_URL}?symbol={symbol}&limit={limit}", retries=3)


# Kandidaten-Universum: bewusst breit und VOR dem Blick auf die Renditen
# fixiert (Gewinner wie Verlierer). Was Binance nicht liefert, fliegt beim
# Abruf automatisch raus und wird als "nicht auf Binance" gezaehlt.
MEMECOIN_CANDIDATES = (
    "DOGE", "SHIB", "PEPE", "FLOKI", "BONK", "WIF", "BOME", "MEME",
    "1000SATS", "1MBABYDOGE", "1000CAT", "DOGS", "NEIRO", "NEIROETH", "TURBO",
    "ACT", "PNUT", "PENGU", "PUMP", "TRUMP", "MELANIA", "HMSTR", "CATI",
    "NOT", "ORDI", "RATS", "BANANA", "BANANAS31", "BROCCOLI714", "BROCCOLI",
    "MUBARAK", "TST", "ANIME", "BABY", "AIXBT", "SLERF", "MYRO", "WEN",
    "SPX", "GOAT", "MOODENG", "POPCAT", "MEW", "ZEREBRO", "FARTCOIN",
    "BRETT", "MOG", "LADYS", "WOJAK", "ELON", "KISHU", "AKITA", "SAMO",
    "PONKE", "TOSHI", "CHILLGUY", "HIPPO", "BAN", "WHY", "GIGA", "FWOG",
    "APU", "SUNDOG", "SWARMS", "PIPPIN", "ARC", "GRIFFAIN", "AI16Z",
    "USELESS", "SNEK", "AIDOGE", "VINE", "LIBRA", "DOGWIFHAT", "KOMA",
    "PEOPLE", "TROY", "SHIB1000", "BABYDOGE",
)
