"""Hauptskript: EIN Poll-Durchlauf. Für wiederholten Aufruf per Cron gedacht
(z.B. alle 5-10 Minuten), kein Dauerprozess/Daemon.

Ablauf pro Aufruf:
    1. State laden (offene Positionen, Cash) aus artifacts/memecoin_paper/state.json
       - existiert die Datei nicht, wird mit STARTING_CASH neu begonnen.
    2. Offene Positionen: aktuellen Preis/Marketcap holen, Exit-Regeln prüfen
       (strategy.py), bei Trigger verkaufen und loggen.
    3. Neue Kandidaten scannen (scanner.py), Stufe-1-Filter (filter.score_candidate),
       für Survivors Stufe-2-Filter (filter.apply_rugcheck, Tagesbudget-begrenzt).
    4. Kandidaten, die den Filter bestehen: ZWEI PARALLELE, UNABHÄNGIGE Paper-
       Positionen möglich (siehe strategy.py):
         a) "baseline"        - generische %-TP/SL/Zeit-Regel, immer möglich.
         b) "mcap_hypothesis" - Nutzer-Hypothese "~20k Mcap Einstieg, 4x-Ziel
                                 (~80k Mcap)", NUR wenn der aktuelle Marketcap
                                 im Fenster strategy.MCAP_HYPOTHESIS_PARAMS
                                 liegt.
       Beide werden getrennt im State (Key "{token}::{strategy}") und in der
       CSV (Spalte "strategy") geführt, damit sie unabhängig ausgewertet
       werden können. Kandidaten, die den Filter NICHT bestehen -> SKIP mit
       Begründung geloggt (kein Kauf ist ein valides Ergebnis, kein Fehler).
    5. State zurückschreiben.

Idempotenz: Ein erneuter Aufruf kurz nach dem letzten führt NICHT zu
Doppelkäufen, weil offene Positionen (pro Token UND Strategie) aus state.json
geprüft werden, bevor neu gekauft wird. Preis-/Filterentscheidungen sind rein
aus den aktuell abgefragten Live-Daten abgeleitet, nicht aus vorherigem
Prozesszustand im Speicher.
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
STRATEGIES = ("baseline", "mcap_hypothesis")
CSV_FIELDS = ["timestamp", "token_address", "symbol", "strategy", "action", "price", "qty", "reason", "equity_after"]


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _position_key(token_address: str, strategy_name: str) -> str:
    return f"{token_address}::{strategy_name}"


def _load_state() -> dict:
    if not STATE_PATH.exists():
        return {"cash": STARTING_CASH, "positions": {}}
    with open(STATE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    data.setdefault("cash", STARTING_CASH)
    data.setdefault("positions", {})
    return data


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
    Quelle der Wahrheit zwischen den Cron-Aufrufen). Positionen werden unter
    ihrem zusammengesetzten Key ("{token}::{strategy}") als eigenständiges
    Broker-Symbol geführt, damit baseline- und mcap-Hypothese-Position im
    selben Token unabhängig voneinander bilanziert werden."""
    broker = SlippageAwarePaperBroker(starting_cash=state["cash"])
    from src.execution.broker import Position as _Position
    for pos_key, pos in state.get("positions", {}).items():
        broker._positions[pos_key] = _Position(
            symbol=pos_key, qty=pos["qty"], entry_price=pos["entry_price"],
        )
    return broker


def _current_price_and_mcap(token_address: str) -> tuple[float | None, float | None]:
    pairs = fetch_pairs_for_token(token_address)
    best = _best_pair(pairs)
    if not best:
        return None, None
    candidate = pair_to_candidate(best, source="price-check")
    if candidate is None:
        return None, None
    return candidate.price_usd, candidate.market_cap


def run_once() -> dict:
    state = _load_state()
    broker = _rebuild_broker(state)
    log_rows: list[dict] = []
    now = _now()

    open_positions = dict(state.get("positions", {}))

    # --- 1. Offene Positionen auf Exit prüfen ---
    for pos_key, pos in list(open_positions.items()):
        token_address = pos["token_address"]
        strategy_name = pos["strategy"]
        current_price, current_mcap = _current_price_and_mcap(token_address)
        if current_price is None:
            log_rows.append({
                "timestamp": now.isoformat(), "token_address": token_address,
                "symbol": pos.get("symbol", "?"), "strategy": strategy_name, "action": "SKIP",
                "price": None, "qty": None,
                "reason": "Kein aktueller Preis abrufbar, Position bleibt offen.",
                "equity_after": None,
            })
            continue

        entry_time = dt.datetime.fromisoformat(pos["entry_time"])
        if strategy_name == "mcap_hypothesis":
            exit_flag, exit_reason = strategy.should_exit_mcap(
                entry_mcap=pos["entry_mcap"], current_mcap=current_mcap,
                entry_price=pos["entry_price"], current_price=current_price,
                entry_time=entry_time, now=now,
            )
        else:
            exit_flag, exit_reason = strategy.should_exit(pos["entry_price"], current_price, entry_time, now)

        if exit_flag:
            order = broker.submit_order(pos_key, qty=pos["qty"], side="sell", price=current_price)
            del open_positions[pos_key]
            log_rows.append({
                "timestamp": now.isoformat(), "token_address": token_address,
                "symbol": pos.get("symbol", "?"), "strategy": strategy_name, "action": "SELL",
                "price": order.filled_price, "qty": order.qty, "reason": exit_reason,
                "equity_after": broker.get_equity(),
            })

    # --- 2. Neue Kandidaten scannen + zweistufig filtern ---
    candidates = scanner.scan()
    n_scanned = len(candidates)
    n_passed_stage1 = 0
    n_passed_stage2 = 0
    n_bought = 0

    for c in candidates:
        result = filter_mod.score_candidate(c)
        if not result.passed:
            log_rows.append({
                "timestamp": now.isoformat(), "token_address": c.token_address,
                "symbol": c.symbol, "strategy": "-", "action": "SKIP", "price": c.price_usd, "qty": None,
                "reason": f"Stufe1 score={result.score:.0f} FAIL: " + " | ".join(result.reasons),
                "equity_after": None,
            })
            continue
        n_passed_stage1 += 1

        result = filter_mod.apply_rugcheck(result)
        if not result.passed:
            log_rows.append({
                "timestamp": now.isoformat(), "token_address": c.token_address,
                "symbol": c.symbol, "strategy": "-", "action": "SKIP", "price": c.price_usd, "qty": None,
                "reason": f"Stufe2(RugCheck) score={result.score:.0f} FAIL: " + " | ".join(result.reasons),
                "equity_after": None,
            })
            continue
        n_passed_stage2 += 1

        if c.price_usd is None or c.price_usd <= 0:
            log_rows.append({
                "timestamp": now.isoformat(), "token_address": c.token_address,
                "symbol": c.symbol, "strategy": "-", "action": "SKIP", "price": None, "qty": None,
                "reason": f"score={result.score:.0f} PASS beide Stufen, aber kein gültiger Preis.",
                "equity_after": None,
            })
            continue

        # welche der beiden parallelen Strategien sind für diesen Kandidaten
        # überhaupt anwendbar UND haben noch keine offene Position?
        candidate_strategies = ["baseline"]
        if strategy.mcap_hypothesis_entry_eligible(c.market_cap):
            candidate_strategies.append("mcap_hypothesis")

        for strategy_name in candidate_strategies:
            pos_key = _position_key(c.token_address, strategy_name)
            if pos_key in open_positions:
                continue  # Idempotenz: keine Doppelposition im selben Token+Strategie

            n_open_this_strategy = sum(1 for p in open_positions.values() if p["strategy"] == strategy_name)
            if n_open_this_strategy >= strategy.MAX_OPEN_POSITIONS:
                log_rows.append({
                    "timestamp": now.isoformat(), "token_address": c.token_address,
                    "symbol": c.symbol, "strategy": strategy_name, "action": "SKIP",
                    "price": c.price_usd, "qty": None,
                    "reason": f"score={result.score:.0f} PASS, aber MAX_OPEN_POSITIONS ({strategy_name}) erreicht.",
                    "equity_after": None,
                })
                continue

            size_usd = strategy.position_size_usd(broker._cash)
            if size_usd <= 0 or size_usd > broker._cash:
                log_rows.append({
                    "timestamp": now.isoformat(), "token_address": c.token_address,
                    "symbol": c.symbol, "strategy": strategy_name, "action": "SKIP",
                    "price": c.price_usd, "qty": None,
                    "reason": "Nicht genug Cash für Positionsgrösse.", "equity_after": None,
                })
                continue

            qty = size_usd / c.price_usd
            order = broker.submit_order(pos_key, qty=qty, side="buy", price=c.price_usd)
            reason_text = f"score={result.score:.0f} PASS: " + " | ".join(result.reasons)
            new_pos = {
                "token_address": c.token_address,
                "strategy": strategy_name,
                "symbol": c.symbol,
                "qty": order.qty,
                "entry_price": order.filled_price,
                "entry_time": now.isoformat(),
                "opened_reason": reason_text,
            }
            if strategy_name == "mcap_hypothesis":
                new_pos["entry_mcap"] = c.market_cap
            open_positions[pos_key] = new_pos
            n_bought += 1
            log_rows.append({
                "timestamp": now.isoformat(), "token_address": c.token_address,
                "symbol": c.symbol, "strategy": strategy_name, "action": "BUY",
                "price": order.filled_price, "qty": order.qty, "reason": reason_text,
                "equity_after": broker.get_equity(),
            })

    new_state = {"cash": broker._cash, "positions": open_positions}
    _save_state(new_state)
    if log_rows:
        _append_trade_log(log_rows)

    summary = {
        "timestamp": now.isoformat(),
        "candidates_scanned": n_scanned,
        "candidates_passed_stage1_dexscreener": n_passed_stage1,
        "candidates_passed_stage2_rugcheck": n_passed_stage2,
        "trades_opened": n_bought,
        "open_positions": len(open_positions),
        "open_positions_by_strategy": {
            s: sum(1 for p in open_positions.values() if p["strategy"] == s) for s in STRATEGIES
        },
        "cash": broker._cash,
        "equity": broker.get_equity(),
    }
    return summary


if __name__ == "__main__":
    result = run_once()
    print(json.dumps(result, indent=2, ensure_ascii=False))
