"""TEIL 2 der historischen Wallet-Backtest-Analyse: für jeden in
wallet_backtest_fetch.py gefundenen Token-Kauf wird die AKTUELLE Bewertung
über Dexscreener nachgeschlagen (`token-pairs/v1/solana/{mint}`, siehe
src/paper_memecoin/dexscreener_client.py - wiederverwendet, nicht dupliziert)
und mit dem historischen Einstiegswert aus der Kauf-Transaktion verglichen.

WERTVERGLEICH OHNE HISTORISCHEN USD-PREIS-FEED (ehrlich dokumentiert):
Wir haben KEINEN historischen SOL/USD-Preis zum jeweiligen Kaufzeitpunkt
(keylos nicht verfügbar, siehe fetcher.py). Statt hilfsweise den AKTUELLEN
SOL/USD-Kurs auf einen Monate alten SOL-Betrag anzuwenden (das würde SOL-
Kursschwankungen fälschlich dem Memecoin zuschlagen), wird konsequent in
SOL denominiert:
    entry_price_sol = amount_sol_beim_kauf / amount_tokens_beim_kauf
    current_price_sol = aktueller_dexscreener_preis_usd / aktueller_sol_preis_usd
    multiple = current_price_sol / entry_price_sol
Das ist exakt (kein Proxy) für die SOL-Seite (amount_sol stammt direkt aus
der on-chain Pre/Post-Balance-Differenz der Kauf-Tx) und nutzt den aktuellen
SOL-Kurs nur als gemeinsamer Nenner auf BEIDEN Seiten - SOL-Kursschwankungen
zwischen Kauf und heute kürzen sich dadurch weitgehend heraus. Das ist eine
Näherung (SOL selbst hat sich ggf. anders zu USD entwickelt als der Memecoin
zu SOL), aber ehrlicher als eine falsche historische USD-Schätzung.

TOTALVERLUST-REGEL (laut Auftrag): Token ohne auffindbaren Dexscreener-Pool
(delisted / nie richtig gelistet / Pool leergezogen) zählen als Totalverlust
(multiple=0.0, return_pct=-1.0), NICHT als fehlender Datenpunkt.
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.paper_memecoin.dexscreener_client import _best_pair, fetch_pairs_for_token  # noqa: E402
from src.wallet_tracker.fetcher import WSOL_MINT, _sol_price_usd  # noqa: E402

ARTIFACT_DIR = REPO_ROOT / "artifacts" / "wallet_tracker"
BUYS_PATH = ARTIFACT_DIR / "historical_buys.json"
VALUATIONS_PATH = ARTIFACT_DIR / "historical_valuations.json"
ENTRIES_JSON_PATH = ARTIFACT_DIR / "historical_entries.json"
ENTRIES_CSV_PATH = ARTIFACT_DIR / "historical_entries.csv"

DEXSCREENER_DELAY_S = 0.25  # kleiner Respektabstand zwischen Dexscreener-Calls


def _load_json(path: Path, default):
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default


def _save_json(path: Path, data) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


def fetch_valuation(mint: str) -> dict:
    pairs = fetch_pairs_for_token(mint)
    best = _best_pair(pairs)
    if not best:
        return {"mint": mint, "found": False, "price_usd": None, "market_cap": None,
                "fdv": None, "liquidity_usd": None, "pair_created_at": None, "dex_id": None}
    price_raw = best.get("priceUsd")
    try:
        price_usd = float(price_raw) if price_raw is not None else None
    except (TypeError, ValueError):
        price_usd = None
    created_ms = best.get("pairCreatedAt")
    pair_created_at = (
        dt.datetime.fromtimestamp(created_ms / 1000, tz=dt.timezone.utc).isoformat() if created_ms else None
    )
    liq = best.get("liquidity") or {}
    return {
        "mint": mint,
        "found": price_usd is not None,
        "price_usd": price_usd,
        "market_cap": best.get("marketCap"),
        "fdv": best.get("fdv"),
        "liquidity_usd": liq.get("usd"),
        "pair_created_at": pair_created_at,
        "dex_id": best.get("dexId"),
    }


def build_valuations(mints: list[str]) -> dict:
    cache = _load_json(VALUATIONS_PATH, {})
    todo = [m for m in mints if m not in cache]
    print(f"[valuation] {len(mints)} eindeutige Mints, {len(todo)} noch ohne Cache-Eintrag.", flush=True)
    for i, mint in enumerate(todo, start=1):
        cache[mint] = fetch_valuation(mint)
        if i % 10 == 0 or i == len(todo):
            _save_json(VALUATIONS_PATH, cache)
            print(f"[valuation] ({i}/{len(todo)}) zuletzt: {mint} -> gefunden={cache[mint]['found']}", flush=True)
        time.sleep(DEXSCREENER_DELAY_S)
    _save_json(VALUATIONS_PATH, cache)
    return cache


def build_entries(buys_cache: dict, valuations: dict, sol_price_now: float) -> list[dict]:
    """Ein Entry pro (wallet_address, token_mint): der FRÜHESTE Buy dieser
    Wallet in diesem Mint innerhalb des Fetch-Fensters gilt als 'Einstieg'
    (spätere Nachkäufe desselben Mints fließen nicht zusätzlich in die
    Rendite-Verteilung ein, um nicht künstlich viele korrelierte Datenpunkte
    aus ein und demselben Trade zu erzeugen)."""
    earliest: dict[tuple[str, str], dict] = {}
    for b in buys_cache.get("buys", []):
        key = (b["wallet_address"], b["token_mint"])
        if key not in earliest or b["block_time"] < earliest[key]["block_time"]:
            earliest[key] = b

    now = dt.datetime.now(dt.timezone.utc)
    entries = []
    for (wallet_address, mint), b in earliest.items():
        amount_sol = b.get("amount_sol")
        amount_tokens = b.get("amount_tokens")
        if not amount_sol or not amount_tokens:
            continue
        entry_price_sol = amount_sol / amount_tokens
        val = valuations.get(mint, {"found": False})
        block_time = dt.datetime.fromisoformat(b["block_time"])
        hold_days = (now - block_time).total_seconds() / 86400.0

        dead = not val.get("found")
        if dead:
            multiple = 0.0
            return_pct = -1.0
        else:
            current_price_sol = val["price_usd"] / sol_price_now
            multiple = current_price_sol / entry_price_sol
            return_pct = multiple - 1.0

        age_at_buy_hours = None
        if val.get("pair_created_at"):
            pair_created = dt.datetime.fromisoformat(val["pair_created_at"])
            age_at_buy_hours = (block_time - pair_created).total_seconds() / 3600.0

        entries.append({
            "wallet_address": wallet_address,
            "wallet_label": b.get("wallet_label"),
            "token_mint": mint,
            "block_time": b["block_time"],
            "amount_sol": amount_sol,
            "entry_price_sol": entry_price_sol,
            "dead_or_delisted": dead,
            "current_price_usd": val.get("price_usd"),
            "current_market_cap": val.get("market_cap"),
            "multiple_since_buy": multiple,
            "return_pct_since_buy": return_pct,
            "hold_days_since_buy": hold_days,
            "age_at_buy_hours": age_at_buy_hours,
            "tx_signature": b.get("tx_signature"),
        })
    entries.sort(key=lambda e: e["block_time"])
    return entries


def main() -> None:
    buys_cache = _load_json(BUYS_PATH, None)
    if buys_cache is None:
        print("[valuation] Keine historical_buys.json gefunden - erst wallet_backtest_fetch.py laufen lassen.")
        return

    mints = sorted({b["token_mint"] for b in buys_cache.get("buys", [])})
    valuations = build_valuations(mints)

    sol_price_now = _sol_price_usd()
    if sol_price_now is None:
        print("[valuation] Konnte aktuellen SOL/USD-Preis nicht ermitteln - breche ab.")
        return
    print(f"[valuation] aktueller SOL/USD-Preis (Dexscreener, WSOL={WSOL_MINT}): {sol_price_now}", flush=True)

    entries = build_entries(buys_cache, valuations, sol_price_now)
    _save_json(ENTRIES_JSON_PATH, {"generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                                    "sol_price_usd_now": sol_price_now, "entries": entries})

    if entries:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        with open(ENTRIES_CSV_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(entries[0].keys()))
            writer.writeheader()
            writer.writerows(entries)

    print(f"[valuation] fertig. {len(entries)} Entries (eindeutige Wallet x Mint Kombinationen) "
          f"aus {len(buys_cache.get('buys', []))} rohen Buy-Events.", flush=True)


if __name__ == "__main__":
    main()
