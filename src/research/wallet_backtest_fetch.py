"""TEIL 1 der historischen Wallet-Backtest-Analyse (siehe Auftrag): holt für
eine Stichprobe der 722 beobachteten Wallets (artifacts/wallet_tracker/
watched_wallets.txt) die Transaktionshistorie der letzten LOOKBACK_DAYS Tage
über die öffentliche Solana-RPC und extrahiert daraus alle Token-KÄUFE
(SOL -> SPL-Token), analog zur bereits bestehenden Logik in
src/wallet_tracker/fetcher.py (extract_wallet_trades wird 1:1 wiederverwendet,
NICHT dupliziert).

Dies ist ein EINMALIGER RECHERCHE-Lauf (kein Cron, kein Daemon), separat vom
produktiven wallet_tracker-Modul - schreibt NUR in
artifacts/wallet_tracker/historical_buys.json (nicht in state.json oder
wallet_trades.csv, um den laufenden produktiven Fetcher nicht zu stören).

STICHPROBENAUSWAHL (dokumentiert, siehe Auftrag "dokumentiere welche Auswahl"):
Die ersten SAMPLE_SIZE Adressen aus watched_wallets.txt in Dateireihenfolge
(nach Überspringen von Kommentar-/Leerzeilen) - keine Zufallsauswahl, damit
der Lauf bei Bedarf exakt reproduzierbar ist. Das ist die Reihenfolge, in der
der Nutzer sie aus Axiom exportiert hat (mutmasslich nach Axiom-eigenem
Ranking) - keine Behauptung, dass das eine repräsentative Zufallsstichprobe
der vollen 722 ist.

ZEITFENSTER: LOOKBACK_DAYS=60 (Mitte der geforderten 30-90 Tage Spanne).

RPC-BUDGET-EINSCHRÄNKUNG (ehrlich, siehe auch solana_rpc_client.py-Docstring):
getSignaturesForAddress liefert max. 1000 Signaturen pro Call. Dieses Skript
paginiert per `before`-Cursor rückwärts (solana_rpc_client selbst unterstützt
das nicht, siehe _rpc_call-Direktaufruf unten - reiner Lesezugriff, keine
Änderung an solana_rpc_client.py), bis entweder (a) der Cutoff erreicht ist,
(b) PAGES_CAP Seiten erreicht sind, (c) keine älteren Signaturen mehr
existieren, oder (d) bereits reichlich erfolgreiche (err=None) Signaturen im
Fenster gefunden wurden (MIN_INWINDOW_TARGET).

WICHTIGER EMPIRISCHER BEFUND (2026-09-18, beim Testen dieses Skripts
entdeckt, siehe Bericht): mehrere der meistkopierten Top-Wallets der Liste
(z.B. "cented", "kadenox", "king trey", "decu") werden zum Zeitpunkt dieses
Laufs mit MASSIVEM Spam überflutet - tausende fehlgeschlagene Transaktionen
(err != None, vermutlich Front-Running-/Copy-Bots) pro paar Sekunden. Bei
diesen Wallets reicht selbst PAGES_CAP*1000 Signaturen oft nur Sekunden bis
wenige Minuten in die Vergangenheit, NICHT 60 Tage - das tatsächlich
abgedeckte Fenster pro Wallet wird explizit mitgeloggt
(oldest_fetched_block_time vs. cutoff, window_fully_covered_by_pagination).
Für solche Wallets liefert dieser Lauf ehrlich wenig/keine historischen Buys
- das ist eine reale Dateneinschränkung dieser Stichprobe, kein Bug.
Zusätzlich wird pro Wallet höchstens MAX_TX_PER_WALLET Transaktionen per
getTransaction nachgeladen (gleichmässig über die im Fenster liegenden
Signaturen verteilt, nicht nur die neuesten).

INKREMENTELLES CACHING: Nach JEDER Wallet wird der Zwischenstand auf Platte
geschrieben (siehe _save_cache) - bei Abbruch (Zeit/Rate-Limit) bleibt ein
sauberes, auswertbares Teilergebnis mit dokumentierter Stichprobengrösse
erhalten, statt nichts.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.wallet_tracker import solana_rpc_client, wallet_list  # noqa: E402
from src.wallet_tracker.fetcher import extract_wallet_trades  # noqa: E402

ARTIFACT_DIR = REPO_ROOT / "artifacts" / "wallet_tracker"
OUT_PATH = ARTIFACT_DIR / "historical_buys.json"

SAMPLE_SIZE = 30
LOOKBACK_DAYS = 60
MAX_TX_PER_WALLET = 20  # Obergrenze getTransaction-Calls pro Wallet (Budget-Grenze, siehe Docstring)
SIGNATURES_PAGE_LIMIT = 1000  # Solana-RPC-Maximum pro getSignaturesForAddress-Call
PAGES_CAP = 8  # Obergrenze Signaturen-Seiten (=RPC-Calls) pro Wallet fuer das rueckwaertige Paging
MIN_INWINDOW_TARGET = 40  # frueher Abbruch des Pagings, sobald genug erfolgreiche Sig. im Fenster gefunden wurden


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _load_cache() -> dict:
    if OUT_PATH.exists():
        with open(OUT_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"run_started_at": _now().isoformat(), "sample_size": SAMPLE_SIZE,
            "lookback_days": LOOKBACK_DAYS, "max_tx_per_wallet": MAX_TX_PER_WALLET,
            "wallets": {}, "buys": []}


def _save_cache(cache: dict) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = OUT_PATH.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)
    tmp.replace(OUT_PATH)


def _evenly_spaced_indices(n: int, k: int) -> list[int]:
    """Wählt k gleichmässig über [0, n) verteilte Indizes (inkl. Rand),
    damit sowohl ältere als auch jüngere Signaturen im Fenster vertreten
    sind, statt nur die allerneuesten (relevant für TEIL 3, Haltedauer-
    Verteilung)."""
    if k >= n:
        return list(range(n))
    if k <= 1:
        return [0]
    step = (n - 1) / (k - 1)
    return sorted({round(i * step) for i in range(k)})


def _fetch_signature_pages(address: str, cutoff: dt.datetime) -> tuple[list[dict], int]:
    """Rückwärtiges `before`-Paging über getSignaturesForAddress (siehe
    Modul-Docstring). Nutzt solana_rpc_client._rpc_call direkt (throttled
    über dasselbe globale Budget wie get_signatures_for_address/
    get_transaction) - reiner Lesezugriff, solana_rpc_client.py bleibt
    unverändert. Gibt (alle_signaturen, anzahl_seiten) zurück."""
    all_sigs: list[dict] = []
    before: str | None = None
    pages = 0
    while pages < PAGES_CAP:
        params: dict = {"limit": SIGNATURES_PAGE_LIMIT}
        if before:
            params["before"] = before
        page = solana_rpc_client._rpc_call("getSignaturesForAddress", [address, params])
        pages += 1
        if not isinstance(page, list) or not page:
            break
        all_sigs.extend(page)
        oldest_bt = page[-1].get("blockTime")
        before = page[-1].get("signature")
        in_window_count = sum(
            1 for s in page if s.get("err") is None and s.get("blockTime")
            and s["blockTime"] >= cutoff.timestamp()
        )
        if oldest_bt is not None and oldest_bt < cutoff.timestamp():
            break  # Fenster-Rand erreicht
        if len(page) < SIGNATURES_PAGE_LIMIT:
            break  # keine weitere Historie fuer diese Adresse vorhanden
        total_in_window_so_far = sum(
            1 for s in all_sigs if s.get("err") is None and s.get("blockTime")
            and s["blockTime"] >= cutoff.timestamp()
        )
        if total_in_window_so_far >= MIN_INWINDOW_TARGET:
            break  # genug brauchbare Signaturen gefunden, weiteres Paging bringt wenig zusaetzlichen Wert
    return all_sigs, pages


def fetch_wallet_buys(address: str, label: str | None, cutoff: dt.datetime, now: dt.datetime) -> dict:
    from src.wallet_tracker.models import WatchedWallet

    wallet = WatchedWallet(address=address, label=label)
    sigs, pages_fetched = _fetch_signature_pages(address, cutoff)

    in_window = [
        s for s in sigs
        if s.get("blockTime") is not None and s.get("err") is None
        and dt.datetime.fromtimestamp(s["blockTime"], tz=dt.timezone.utc) >= cutoff
    ]
    # älteste zuerst, damit die gleichmässige Auswahl unten Sinn ergibt
    in_window.sort(key=lambda s: s["blockTime"])

    oldest_fetched_bt = min((s["blockTime"] for s in sigs if s.get("blockTime")), default=None)
    window_fully_covered = oldest_fetched_bt is not None and oldest_fetched_bt <= cutoff.timestamp()

    idxs = _evenly_spaced_indices(len(in_window), MAX_TX_PER_WALLET)
    selected = [in_window[i] for i in idxs]

    buys = []
    tx_fetch_failures = 0
    for entry in selected:
        sig = entry["signature"]
        tx = solana_rpc_client.get_transaction(sig)
        if tx is None:
            tx_fetch_failures += 1
            continue
        block_time = dt.datetime.fromtimestamp(entry["blockTime"], tz=dt.timezone.utc)
        # sol_price_usd=None bewusst: wir wollen HIER keine mit dem AKTUELLEN
        # SOL-Kurs verfälschte "estimated_usd"-Spalte für historische Käufe -
        # die Bewertung erfolgt in TEIL 2 sauber SOL-denominiert (siehe
        # wallet_backtest_valuation.py).
        trades = extract_wallet_trades(wallet, tx, sig, block_time, now, sol_price_usd=None)
        for t in trades:
            if t.action != "buy" or not t.amount_sol or t.amount_sol <= 0 or t.amount_tokens <= 0:
                continue
            buys.append(dataclasses.asdict(t) | {
                "block_time": t.block_time.isoformat(),
                "detected_at": None,  # nicht relevant fuer retrospektive Analyse, siehe Einschraenkung im Bericht
            })

    return {
        "address": address,
        "label": label,
        "signatures_fetched": len(sigs),
        "signature_pages_fetched": pages_fetched,
        "signatures_in_window": len(in_window),
        "window_fully_covered_by_pagination": window_fully_covered,
        "oldest_fetched_block_time": (
            dt.datetime.fromtimestamp(oldest_fetched_bt, tz=dt.timezone.utc).isoformat()
            if oldest_fetched_bt else None
        ),
        "transactions_fetched": len(selected),
        "tx_fetch_failures": tx_fetch_failures,
        "buys_found": len(buys),
        "buys": buys,
    }


def main() -> None:
    wallets, warnings = wallet_list.load_watched_wallets()
    sample = wallets[:SAMPLE_SIZE]
    print(f"[fetch] {len(wallets)} Wallets geladen, Stichprobe: erste {len(sample)}.", flush=True)
    if warnings:
        print(f"[fetch] Warnungen beim Laden: {warnings}", flush=True)

    now = _now()
    cutoff = now - dt.timedelta(days=LOOKBACK_DAYS)
    cache = _load_cache()
    cache["sample_addresses"] = [w.address for w in sample]

    done = set(cache["wallets"].keys())
    t0 = time.monotonic()
    for i, w in enumerate(sample, start=1):
        if w.address in done:
            print(f"[fetch] ({i}/{len(sample)}) {w.label or w.address} - bereits im Cache, ueberspringe.", flush=True)
            continue
        result = fetch_wallet_buys(w.address, w.label, cutoff, now)
        cache["wallets"][w.address] = result
        cache["buys"].extend(result["buys"])
        _save_cache(cache)
        elapsed = time.monotonic() - t0
        print(
            f"[fetch] ({i}/{len(sample)}) {w.label or w.address}: "
            f"{result['signatures_in_window']} Sig im Fenster, "
            f"{result['transactions_fetched']} Tx geladen, {result['buys_found']} Buys "
            f"({result['signature_pages_fetched']} Sig-Seiten, "
            f"voll abgedeckt: {result['window_fully_covered_by_pagination']}) "
            f"[t={elapsed:.0f}s]",
            flush=True,
        )

    cache["run_finished_at"] = _now().isoformat()
    _save_cache(cache)
    total_buys = len(cache["buys"])
    print(f"[fetch] fertig. {len(cache['wallets'])} Wallets verarbeitet, {total_buys} Buy-Events gesamt.", flush=True)


if __name__ == "__main__":
    main()
