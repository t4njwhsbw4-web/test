"""Funding-Rate-Historie und Spot-Klines fuer den Cash-and-Carry-Trade (Basis-Trade).

Datenquellen (in dieser Umgebung geprueft, 2026-09):
  - fapi.binance.com          -> HTTP 451 (geo-blockiert)
  - api.bybit.com             -> HTTP 403 (geo-blockiert)
  - www.okx.com               -> HTTP 200, aber nur ~3 Monate Historie pro Abruf
  - data.binance.vision       -> HTTP 200, monatliche CSV-Dumps ab 2020-01  <== genutzt
  - data-api.binance.vision   -> HTTP 200, Spot-Klines (siehe src/data/intraday_fetch.py)

Die monatlichen Funding-Dumps liegen unter
  data/futures/um/monthly/fundingRate/<SYMBOL>/<SYMBOL>-fundingRate-YYYY-MM.zip
mit den Spalten calc_time, funding_interval_hours, last_funding_rate.

Wichtig: funding_interval_hours ist NICHT konstant 8. Binance hat fuer mehrere
Symbole auf 4h umgestellt (u.a. in Hochvola-Phasen). Die Spalte wird daher
ausgewertet und nicht mit "3x taeglich" hart verdrahtet.
"""
from __future__ import annotations

import io
import time
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = REPO_ROOT / "data_cache" / "funding"
SPOT_CACHE_DIR = REPO_ROOT / "data_cache" / "funding_spot"

VISION_BASE = "https://data.binance.vision/data/futures/um/monthly/fundingRate"
S3_LIST = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
SPOT_KLINES = "https://data-api.binance.vision/api/v3/klines"

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT"]


def _fetch(url: str, retries: int = 4) -> bytes:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "research/1.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise FileNotFoundError(url) from exc
            last = exc
            time.sleep(1.0 * (attempt + 1))
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(1.0 * (attempt + 1))
    raise RuntimeError(f"Abruf fehlgeschlagen: {url}") from last


def list_available_months(symbol: str) -> list[str]:
    """Listet die vorhandenen Monats-Dumps ueber die S3-XML-API."""
    months: list[str] = []
    token = ""
    prefix = f"data/futures/um/monthly/fundingRate/{symbol}/"
    while True:
        url = f"{S3_LIST}?list-type=2&prefix={prefix}&max-keys=1000"
        if token:
            url += f"&continuation-token={urllib.parse.quote(token, safe='')}"
        xml = _fetch(url).decode()
        for key in pd.Series(xml.split("<Key>")).iloc[1:]:
            k = key.split("</Key>")[0]
            if k.endswith(".zip"):
                months.append(k.split("-fundingRate-")[-1][:-4])
        if "<IsTruncated>true</IsTruncated>" not in xml:
            break
        token = xml.split("<NextContinuationToken>")[1].split("</NextContinuationToken>")[0]
    return sorted(set(months))


def fetch_funding(symbol: str, use_cache: bool = True) -> pd.DataFrame:
    """Volle Funding-Historie eines Symbols als DataFrame (UTC-Index)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"{symbol}_funding.parquet"
    if use_cache and cache.exists():
        return pd.read_parquet(cache)

    frames = []
    for month in list_available_months(symbol):
        url = f"{VISION_BASE}/{symbol}/{symbol}-fundingRate-{month}.zip"
        try:
            raw = _fetch(url)
        except FileNotFoundError:
            continue
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            name = zf.namelist()[0]
            df = pd.read_csv(zf.open(name))
        # Manche Dumps haben eine Kopfzeile, andere nicht.
        if df.columns[0] != "calc_time":
            df = pd.read_csv(
                zipfile.ZipFile(io.BytesIO(raw)).open(name),
                names=["calc_time", "funding_interval_hours", "last_funding_rate"],
            )
        frames.append(df)

    if not frames:
        raise RuntimeError(f"Keine Funding-Daten fuer {symbol}")

    out = pd.concat(frames, ignore_index=True)
    out = out[pd.to_numeric(out["calc_time"], errors="coerce").notna()]
    out["calc_time"] = pd.to_numeric(out["calc_time"])
    out["ts"] = pd.to_datetime(out["calc_time"], unit="ms", utc=True)
    out["funding_interval_hours"] = pd.to_numeric(out["funding_interval_hours"])
    out["last_funding_rate"] = pd.to_numeric(out["last_funding_rate"])
    out = out.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    out = out[["ts", "funding_interval_hours", "last_funding_rate"]]
    out.to_parquet(cache, index=False)
    return out


def fetch_spot_daily(symbol: str, use_cache: bool = True) -> pd.DataFrame:
    """Taegliche Spot-Klines (OHLC) fuer Regime-Einteilung und Liquidationstest."""
    SPOT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = SPOT_CACHE_DIR / f"{symbol}_1d.parquet"
    if use_cache and cache.exists():
        return pd.read_parquet(cache)

    import json

    cols = ["open_time", "open", "high", "low", "close", "volume", "close_time",
            "qv", "n", "tbb", "tbq", "ig"]
    rows: list[list] = []
    start = int(pd.Timestamp("2019-09-01", tz="UTC").timestamp() * 1000)
    while True:
        url = f"{SPOT_KLINES}?symbol={symbol}&interval=1d&startTime={start}&limit=1000"
        batch = json.loads(_fetch(url).decode())
        if not batch:
            break
        rows.extend(batch)
        start = batch[-1][0] + 86_400_000
        if len(batch) < 1000:
            break
        time.sleep(0.2)

    df = pd.DataFrame(rows, columns=cols)
    df["ts"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c])
    df = df[["ts", "open", "high", "low", "close"]].drop_duplicates("ts")
    df = df.sort_values("ts").reset_index(drop=True)
    df.to_parquet(cache, index=False)
    return df


if __name__ == "__main__":
    for sym in SYMBOLS:
        try:
            f = fetch_funding(sym)
            s = fetch_spot_daily(sym)
            print(f"{sym:10s} funding {f['ts'].min().date()} .. {f['ts'].max().date()} "
                  f"n={len(f):6d}  intervals={sorted(f['funding_interval_hours'].unique())}  "
                  f"spot n={len(s)}")
        except Exception as exc:  # noqa: BLE001
            print(f"{sym:10s} FEHLER: {exc}")
