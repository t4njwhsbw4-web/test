"""Hauptskript: EIN Poll-Durchlauf. Für wiederholten Aufruf per Cron gedacht
(z.B. alle 5-10 Minuten), kein Dauerprozess/Daemon.

Ablauf pro Aufruf:
    1. State laden (offene Positionen, Cash) aus artifacts/memecoin_paper/state.json
       - existiert die Datei nicht, wird mit STARTING_CASH neu begonnen.
    2. Offene Positionen: aktuellen Preis holen, Exit-Regeln prüfen (strategy.py),
       bei Trigger verkaufen und loggen.
    3. Neue Kandidaten scannen (scanner.py), filtern (filter.py).
    4. Kandidaten, die den Filter bestehen UND noch keine offene Position sind
       UND Platz im Portfolio ist (MAX_OPEN_POSITIONS) -> Paper-Kauf, loggen.
       Kandidaten, die NICHT bestehen -> als SKIP mit Begründung geloggt
       (kein Kauf ist ein valides Ergebnis, kein Fehler).
    5. State zurückschreiben.

Idempotenz: Ein erneuter Aufruf kurz nach dem letzten führt NICHT zu
Doppelkäufen, weil offene Positionen aus state.json geprüft werden, bevor neu
gekauft wird. Preis-/Filterentscheidungen sind rein aus den aktuell
abgefragten Live-Daten abgeleitet, nicht aus vorherigem Prozesszustand im
Speicher.
"""
from __future__ import annotations

import csv
import datetime as dt
import json
from pathlib import Path

from . import filter as filter_mod
from . import scanner, strategy
from .dexscreener_client import _best_pair, fetch_pairs_for_token, pair_to_candidate
from .paper_broker import SlippageAwarePaperBroker

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = REPO_ROOT / "artifacts" / "memecoin_paper"
STATE_PATH = ARTIFACT_DIR / "state.json"
TRADES_CSV_PATH = ARTIFACT_DIR / "trades.csv"

STARTING_CASH = 1000.0
CSV_FIELDS = ["timestamp", "token_address", "symbol", "action", "price", "qty", "reason", "equity_after"]


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _load_state() -> dict:
    if not STATE_PATH.exists():
        return {"cash": STARTING_CASH, "positions": {}}
    with open(STATE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_state(state: dict) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = STATE_PATH.with_suffix(".json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
    tmp_path.replace(STATE_PATH)


def _append_trade_log(rows: list[dict]) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    file_exists = TRADES_CSV_PATH.exists()
    with open(TRADES_CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _rebuild_broker(state: dict) -> SlippageAwarePaperBroker:
    """Baut den Broker-Zustand aus state.json neu auf (kein DB, JSON ist die
    Quelle der Wahrheit zwischen den Cron-Aufrufen)."""
    broker = SlippageAwarePaperBroker(starting_cash=state["cash"])
    for token_address, pos in state.get("positions", {}).items():
        broker._positions[token_address] = broker._positions.get(token_address)
        from src.execution.broker import Position as _Position
        broker._positions[token_address] = _Position(
            symbol=token_address, qty=pos["qty"], entry_price=pos["entry_price"],
        )
    return broker


def _current_price(token_address: str) -> float | None:
    pairs = fetch_pairs_for_token(token_address)
    best = _best_pair(pairs)
    if not best:
        return None
    candidate = pair_to_candidate(best, source="price-check")
    return candidate.price_usd if candidate else None


def run_once() -> dict:
    state = _load_state()
    broker = _rebuild_broker(state)
    log_rows: list[dict] = []
    now = _now()

    # --- 1. Offene Positionen auf Exit prüfen ---
    open_positions = dict(state.get("positions", {}))
    for token_address, pos in list(open_positions.items()):
        current_price = _current_price(token_address)
        if current_price is None:
            log_rows.append({
                "timestamp": now.isoformat(), "token_address": token_address,
                "symbol": pos.get("symbol", "?"), "action": "SKIP", "price": None,
                "qty": None, "reason": "Kein aktueller Preis abrufbar, Position bleibt offen.",
                "equity_after": None,
            })
            continue
        entry_time = dt.datetime.fromisoformat(pos["entry_time"])
        exit_flag, exit_reason = strategy.should_exit(pos["entry_price"], current_price, entry_time, now)
        if exit_flag:
            order = broker.submit_order(token_address, qty=pos["qty"], side="sell", price=current_price)
            del open_positions[token_address]
            log_rows.append({
                "timestamp": now.isoformat(), "token_address": token_address,
                "symbol": pos.get("symbol", "?"), "action": "SELL", "price": order.filled_price,
                "qty": order.qty, "reason": exit_reason, "equity_after": broker.get_equity(),
            })

    # --- 2. Neue Kandidaten scannen + filtern ---
    candidates = scanner.scan()
    n_scanned = len(candidates)
    n_passed = 0
    n_bought = 0

    for c in candidates:
        if c.token_address in open_positions:
            continue  # Idempotenz: keine Doppelposition im selben Token
        result = filter_mod.score_candidate(c)
        if not result.passed:
            log_rows.append({
                "timestamp": now.isoformat(), "token_address": c.token_address,
                "symbol": c.symbol, "action": "SKIP", "price": c.price_usd, "qty": None,
                "reason": f"score={result.score:.0f} FAIL: " + " | ".join(result.reasons),
                "equity_after": None,
            })
            continue
        n_passed += 1

        if len(open_positions) >= strategy.MAX_OPEN_POSITIONS:
            log_rows.append({
                "timestamp": now.isoformat(), "token_address": c.token_address,
                "symbol": c.symbol, "action": "SKIP", "price": c.price_usd, "qty": None,
                "reason": f"score={result.score:.0f} PASS, aber MAX_OPEN_POSITIONS erreicht.",
                "equity_after": None,
            })
            continue
        if c.price_usd is None or c.price_usd <= 0:
            log_rows.append({
                "timestamp": now.isoformat(), "token_address": c.token_address,
                "symbol": c.symbol, "action": "SKIP", "price": None, "qty": None,
                "reason": f"score={result.score:.0f} PASS, aber kein gültiger Preis.",
                "equity_after": None,
            })
            continue

        size_usd = strategy.position_size_usd(broker._cash)
        if size_usd <= 0 or size_usd > broker._cash:
            log_rows.append({
                "timestamp": now.isoformat(), "token_address": c.token_address,
                "symbol": c.symbol, "action": "SKIP", "price": c.price_usd, "qty": None,
                "reason": "Nicht genug Cash für Positionsgrösse.", "equity_after": None,
            })
            continue

        qty = size_usd / c.price_usd
        order = broker.submit_order(c.token_address, qty=qty, side="buy", price=c.price_usd)
        open_positions[c.token_address] = {
            "symbol": c.symbol,
            "qty": order.qty,
            "entry_price": order.filled_price,
            "entry_time": now.isoformat(),
            "opened_reason": f"score={result.score:.0f} PASS: " + " | ".join(result.reasons),
        }
        n_bought += 1
        log_rows.append({
            "timestamp": now.isoformat(), "token_address": c.token_address,
            "symbol": c.symbol, "action": "BUY", "price": order.filled_price,
            "qty": order.qty, "reason": f"score={result.score:.0f} PASS: " + " | ".join(result.reasons),
            "equity_after": broker.get_equity(),
        })

    new_state = {"cash": broker._cash, "positions": open_positions}
    _save_state(new_state)
    if log_rows:
        _append_trade_log(log_rows)

    summary = {
        "timestamp": now.isoformat(),
        "candidates_scanned": n_scanned,
        "candidates_passed_filter": n_passed,
        "trades_opened": n_bought,
        "open_positions": len(open_positions),
        "cash": broker._cash,
        "equity": broker.get_equity(),
    }
    return summary


if __name__ == "__main__":
    result = run_once()
    print(json.dumps(result, indent=2, ensure_ascii=False))
