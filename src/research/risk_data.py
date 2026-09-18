"""Tagesrenditen fuer die Risikoanalyse laden (yfinance-Cache + Binance-Daily).

Zwei Quellen, weil beide Luecken haben:

* `src/data/fetch.py` (yfinance) deckt BTC/ETH/SOL/DOGE ab, hat aber fuer
  junge Memecoins nur Bruchstuecke (SHIB-USD im Cache: 731 Bars).
* `data-api.binance.vision` (derselbe Host wie in `src/data/intraday_fetch.py`,
  kein API-Key) liefert 1d-Klines auch fuer PEPE/WIF/BONK/FLOKI. Die dortige
  `fetch_klines` erlaubt nur Intraday-Intervalle, darum hier ein eigener,
  bewusst minimaler Daily-Loader - bestehende Dateien werden nicht angefasst.

Alle Renditen sind einfache Tagesrenditen (close/close - 1) auf Basis von
UTC-Tagesbars.
"""
from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
YF_CACHE = REPO_ROOT / "data_cache"
BINANCE_CACHE = REPO_ROOT / "data_cache" / "binance_daily"
BINANCE_URL = "https://data-api.binance.vision/api/v3/klines"

# Asset-Gruppen nach erwartetem Volatilitaetsniveau.
YF_ASSETS = {
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
    "SOL": "SOL-USD",
    "DOGE": "DOGE-USD",
}
BINANCE_ASSETS = {
    "SHIB": "SHIBUSDT",
    "PEPE": "PEPEUSDT",
    "WIF": "WIFUSDT",
    "BONK": "BONKUSDT",
    "FLOKI": "FLOKIUSDT",
}

GROUPS = {
    "BTC": "Large-Cap",
    "ETH": "Large-Cap",
    "SOL": "Altcoin",
    "DOGE": "Memecoin (alt)",
    "SHIB": "Memecoin",
    "PEPE": "Memecoin",
    "WIF": "Memecoin",
    "BONK": "Memecoin",
    "FLOKI": "Memecoin",
}


def _get_json(url: str, retries: int = 5) -> list:
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "research/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode())
        except Exception as exc:  # noqa: BLE001 - Netzfehler sind erwartbar
            last_err = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Abruf fehlgeschlagen: {url}") from last_err


def fetch_binance_daily(symbol: str, start: str = "2017-01-01", use_cache: bool = True) -> pd.Series:
    """Close-Serie aus 1d-Klines, paginiert, lokal gecacht."""
    BINANCE_CACHE.mkdir(parents=True, exist_ok=True)
    path = BINANCE_CACHE / f"{symbol}_1d.parquet"
    if use_cache and path.exists():
        df = pd.read_parquet(path)
        return df.set_index("timestamp")["close"]

    step = 86_400_000
    cursor = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
    rows: list[list] = []
    while cursor < end_ms:
        url = f"{BINANCE_URL}?symbol={symbol}&interval=1d&startTime={cursor}&endTime={end_ms}&limit=1000"
        batch = _get_json(url)
        if not batch:
            break
        rows.extend(batch)
        new_cursor = int(batch[-1][0]) + step
        if new_cursor <= cursor:
            break
        cursor = new_cursor
        time.sleep(0.12)
    if not rows:
        raise RuntimeError(f"Keine Daily-Daten fuer {symbol}")

    df = pd.DataFrame(rows).iloc[:, [0, 4]]
    df.columns = ["open_time", "close"]
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.tz_localize(None)
    df["close"] = df["close"].astype(float)
    df = df[["timestamp", "close"]].drop_duplicates("timestamp").sort_values("timestamp")
    df.to_parquet(path, index=False)
    return df.set_index("timestamp")["close"]


def load_yf_close(symbol: str) -> pd.Series:
    path = YF_CACHE / f"{symbol}_1d.parquet"
    if not path.exists():
        from src.data.fetch import fetch_ohlcv  # lazy, damit Netz nur bei Bedarf

        df = fetch_ohlcv(symbol, history_days=4000, cache_dir=str(YF_CACHE))
    else:
        df = pd.read_parquet(path)
    return df["close"]


def load_returns(use_cache: bool = True) -> dict[str, pd.Series]:
    """{Asset-Kuerzel: Tagesrenditen}. Einträge, die nicht laden, werden übersprungen."""
    out: dict[str, pd.Series] = {}
    for name, sym in YF_ASSETS.items():
        try:
            close = load_yf_close(sym)
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] {name}: {exc}")
            continue
        out[name] = close.pct_change().dropna()
    for name, sym in BINANCE_ASSETS.items():
        try:
            close = fetch_binance_daily(sym, use_cache=use_cache)
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] {name}: {exc}")
            continue
        out[name] = close.pct_change().dropna()
    return {k: v[np.isfinite(v)] for k, v in out.items() if len(v) > 200}


def describe(returns: dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    for name, r in returns.items():
        ann_vol = r.std(ddof=1) * np.sqrt(365)
        ann_mean = r.mean() * 365
        log_r = np.log1p(r)
        cagr = np.expm1(log_r.mean() * 365)
        rows.append(
            {
                "Asset": name,
                "Gruppe": GROUPS.get(name, "?"),
                "Tage": len(r),
                "Start": r.index.min().date(),
                "Vol p.a.": ann_vol,
                "Mean p.a. (arith)": ann_mean,
                "CAGR (geom)": cagr,
                "Sharpe (hist)": ann_mean / ann_vol,
                "Tagesvol": r.std(ddof=1),
                "Schiefe": r.skew(),
                "Kurtosis": r.kurtosis(),
                "Worst Tag": r.min(),
            }
        )
    return pd.DataFrame(rows).set_index("Asset")


if __name__ == "__main__":
    rets = load_returns()
    pd.set_option("display.width", 200)
    print(describe(rets).to_string(float_format=lambda x: f"{x:,.3f}"))
