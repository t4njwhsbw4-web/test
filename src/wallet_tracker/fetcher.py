"""Ein Poll-Durchlauf über alle beobachteten Wallets. Für wiederholten Aufruf
per Cron gedacht (analog zu src/paper_memecoin/run_loop.py), kein Dauerprozess.

Ablauf pro Aufruf (run_once):
    1. Wallet-Liste laden (wallet_list.load_watched_wallets) - Default-Pfad
       ist artifacts/wallet_tracker/watched_wallets.txt (die ECHTE Liste,
       sobald sie da ist); solange sie fehlt, meldet der Lauf "0 Wallets"
       statt abzustürzen.
    2. State laden (artifacts/wallet_tracker/state.json): pro Wallet die
       zuletzt verarbeitete Signatur, damit ein erneuter Aufruf dieselbe
       Transaktion nicht doppelt loggt (Idempotenz, analog zu run_loop.py).
    3. Pro Wallet: neue Signaturen seit der letzten bekannten holen
       (solana_rpc_client.get_signatures_for_address mit `until`), ältestes
       zuerst verarbeiten, pro Transaktion Token-Balance-Deltas extrahieren
       -> WalletTrade-Zeilen.
    4. Neue WalletTrade-Zeilen an wallet_trades.csv anhängen, State
       zurückschreiben.

LATENZ / "IMITATION PENALTY" (ehrlich einpreisen, nicht schönrechnen):
Die Verzögerung zwischen einer on-chain Transaktion (block_time) und dem
Moment, in dem WIR sie über diesen Fetcher sehen (detected_at), setzt sich
zusammen aus:
    a) Wartezeit bis zum nächsten Poll-Zyklus (0 bis POLL_INTERVAL_SECONDS,
       im Mittel POLL_INTERVAL_SECONDS/2, WENN der Cron-Takt eingehalten wird)
    b) RPC-Antwortzeit pro Call (Beobachtung in dieser Umgebung: ca. 0.2-0.5s
       pro Call bei ruhiger Last, siehe REQUESTS_PER_SECOND_BUDGET-Throttle)
    c) bei 1000 Wallets zusätzlich: Wartezeit INNERHALB des Zyklus, bis genau
       DIESE Wallet an der Reihe ist (siehe estimate_full_cycle_seconds unten)
Jede WalletTrade-Zeile führt ihre TATSÄCHLICH gemessene
detection_latency_seconds (detected_at - block_time), keine Schätzung. Diese
Latenz ist der Kern der bereits recherchierten "Imitation Penalty": wir
sehen einen Kauf einer Smart-Money-Wallet frühestens Sekunden, realistisch
bei 1000 Wallets eher MINUTEN nach dem eigentlichen on-chain-Ereignis (siehe
estimate_full_cycle_seconds) - ein Paper-Trade, der so tut, als würde er
GLEICHZEITIG mit der beobachteten Wallet ausgeführt, wäre unehrlich optimistisch.

RATE-LIMIT-REALISMUS BEI 1000 WALLETS:
Pro Wallet braucht ein Zyklus mindestens 1 RPC-Call (getSignaturesForAddress)
plus 1 weiteren Call PRO NEUER TRANSAKTION (getTransaction). Für die grobe
Zyklusdauer-Schätzung wird MIN_CALLS_PER_WALLET (>=1, siehe unten) als
Untergrenze angenommen - reale Wallets mit mehreren neuen Trades pro Zyklus
brauchen entsprechend mehr Calls und verlängern den Zyklus zusätzlich.
"""
from __future__ import annotations

import csv
import dataclasses
import datetime as dt
import json
from pathlib import Path

from . import solana_rpc_client, wallet_list
from .models import WalletTrade, WatchedWallet

# Import rein lesend aus dem bestehenden Paper-Trading-Modul, um die
# aktuelle SOL/USD-Näherung nicht doppelt zu implementieren. Es wird NICHTS
# an paper_memecoin verändert.
from src.paper_memecoin.dexscreener_client import _best_pair, fetch_pairs_for_token

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = REPO_ROOT / "artifacts" / "wallet_tracker"
STATE_PATH = ARTIFACT_DIR / "state.json"
TRADES_CSV_PATH = ARTIFACT_DIR / "wallet_trades.csv"

WSOL_MINT = "So11111111111111111111111111111111111111112"  # wrapped SOL - kein "Memecoin", dient nur als Gegenwert-Näherung

SIGNATURES_PER_WALLET_PER_POLL = 10  # Obergrenze neuer Signaturen, die pro Wallet und Zyklus verarbeitet werden
MIN_TOKEN_DELTA = 1e-6  # Rundungsrauschen unterhalb dieser Schwelle zählt nicht als Buy/Sell

# Für die Zyklusdauer-/Latenz-Rechnung im Bericht angenommener Cron-Takt.
# Das ist eine PLANUNGSANNAHME, kein technisches Limit - kleiner = aktueller,
# aber mehr RPC-Last; grösser = weniger Last, aber ältere Sicht auf Trades.
ASSUMED_POLL_INTERVAL_SECONDS = 300  # 5 Minuten

CSV_FIELDS = [f.name for f in dataclasses.fields(WalletTrade)]


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _load_state() -> dict:
    if not STATE_PATH.exists():
        return {}
    with open(STATE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_state(state: dict) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = STATE_PATH.with_suffix(".json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
    tmp_path.replace(STATE_PATH)


def _append_trades_csv(trades: list[WalletTrade]) -> None:
    if not trades:
        return
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    file_exists = TRADES_CSV_PATH.exists()
    with open(TRADES_CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not file_exists:
            writer.writeheader()
        for t in trades:
            row = dataclasses.asdict(t)
            row["block_time"] = t.block_time.isoformat()
            row["detected_at"] = t.detected_at.isoformat()
            writer.writerow(row)


def _sol_price_usd() -> float | None:
    """AKTUELLER SOL/USD-Preis (Dexscreener, WSOL-Mint) - wird als grobe
    Näherung für estimated_usd genutzt. NICHT der historische Preis zum
    Zeitpunkt der jeweiligen Transaktion (dafür bräuchte man einen
    historischen Preis-Feed, den wir keylos nicht haben) - bei stark
    schwankendem SOL-Kurs kann die USD-Schätzung älterer Trades daneben liegen."""
    try:
        pairs = fetch_pairs_for_token(WSOL_MINT)
        best = _best_pair(pairs)
        if not best:
            return None
        price = best.get("priceUsd")
        return float(price) if price is not None else None
    except (TypeError, ValueError):
        return None


def _account_index(tx: dict, address: str) -> int | None:
    keys = ((tx.get("transaction") or {}).get("message") or {}).get("accountKeys") or []
    for i, k in enumerate(keys):
        pubkey = k.get("pubkey") if isinstance(k, dict) else k
        if pubkey == address:
            return i
    return None


def extract_wallet_trades(
    wallet: WatchedWallet, tx: dict, signature: str, block_time: dt.datetime, detected_at: dt.datetime,
    sol_price_usd: float | None,
) -> list[WalletTrade]:
    """Leitet aus einer geparsten Transaktion (encoding=jsonParsed) alle
    Buy-/Sell-Events DIESER Wallet ab. Ein Buy/Sell pro betroffenem Mint
    (ausser WSOL selbst - das dient nur als Gegenwert, siehe WSOL_MINT).

    Kein SPL-Token-Filter auf "ist das ein Memecoin": jede Mint-Änderung
    ausser WSOL wird als Trade gewertet - Einordnung, ob es sich um einen
    Memecoin handelt, ist Aufgabe der aufrufenden Analyse (z.B. via
    paper_memecoin.dexscreener_client Nachschlag), nicht dieses Extraktors."""
    meta = tx.get("meta") or {}
    pre = {b["accountIndex"]: b for b in (meta.get("preTokenBalances") or [])}
    post = {b["accountIndex"]: b for b in (meta.get("postTokenBalances") or [])}

    # Netto-Delta pro Mint für Token-Accounts, die dieser Wallet gehören.
    deltas: dict[str, float] = {}
    for idx in set(pre) | set(post):
        pre_b, post_b = pre.get(idx), post.get(idx)
        owner = (post_b or pre_b or {}).get("owner")
        mint = (post_b or pre_b or {}).get("mint")
        if owner != wallet.address or mint is None or mint == WSOL_MINT:
            continue
        pre_amt = float(((pre_b or {}).get("uiTokenAmount") or {}).get("uiAmount") or 0.0)
        post_amt = float(((post_b or {}).get("uiTokenAmount") or {}).get("uiAmount") or 0.0)
        deltas[mint] = deltas.get(mint, 0.0) + (post_amt - pre_amt)

    # SOL-Gegenwert-Näherung über die eigene Wallet-Account-Zeile (falls die
    # Wallet selbst als top-level Account-Key auftaucht, z.B. als Fee-Payer/
    # Signer). Enthält ggf. die Netzwerkgebühr mit drin, daher NUR eine grobe
    # Näherung, kein exakter Swap-Betrag.
    sol_delta = None
    idx = _account_index(tx, wallet.address)
    pre_bal, post_bal = meta.get("preBalances"), meta.get("postBalances")
    if idx is not None and pre_bal and post_bal and idx < len(pre_bal) and idx < len(post_bal):
        sol_delta = (post_bal[idx] - pre_bal[idx]) / 1_000_000_000.0

    trades: list[WalletTrade] = []
    for mint, delta in deltas.items():
        if abs(delta) < MIN_TOKEN_DELTA:
            continue
        action = "buy" if delta > 0 else "sell"
        amount_sol = abs(sol_delta) if sol_delta is not None else None
        estimated_usd = amount_sol * sol_price_usd if (amount_sol is not None and sol_price_usd is not None) else None
        trades.append(WalletTrade(
            wallet_address=wallet.address,
            wallet_label=wallet.label,
            token_mint=mint,
            action=action,
            amount_tokens=abs(delta),
            amount_sol=amount_sol,
            estimated_usd=estimated_usd,
            block_time=block_time,
            detected_at=detected_at,
            detection_latency_seconds=(detected_at - block_time).total_seconds(),
            tx_signature=signature,
            slot=tx.get("slot"),
        ))
    return trades


def run_once(wallets_path: Path | str | None = None, max_wallets: int | None = None) -> dict:
    wallets, warnings = wallet_list.load_watched_wallets(wallets_path)
    if max_wallets is not None:
        wallets = wallets[:max_wallets]

    state = _load_state()
    sol_price = _sol_price_usd()
    now = _now()

    new_trades: list[WalletTrade] = []
    rpc_calls = 0
    tx_fetch_failures = 0

    for w in wallets:
        wallet_state = state.get(w.address, {})
        last_sig = wallet_state.get("last_signature")

        sigs = solana_rpc_client.get_signatures_for_address(w.address, until=last_sig, limit=SIGNATURES_PER_WALLET_PER_POLL)
        rpc_calls += 1

        newest_processed_sig = last_sig
        for entry in reversed(sigs):  # älteste zuerst verarbeiten
            sig = entry.get("signature")
            block_time_raw = entry.get("blockTime")
            if entry.get("err") is not None or block_time_raw is None:
                # fehlgeschlagene Tx oder kein Zeitstempel -> nichts zu loggen,
                # aber als verarbeitet markieren, damit sie nicht jeden Zyklus
                # erneut angefasst wird.
                newest_processed_sig = sig
                continue

            tx = solana_rpc_client.get_transaction(sig)
            rpc_calls += 1
            if tx is None:
                tx_fetch_failures += 1
                break  # State NICHT über diese Signatur hinaus vorrücken -> nächster Zyklus holt sie erneut

            block_time = dt.datetime.fromtimestamp(block_time_raw, tz=dt.timezone.utc)
            trades = extract_wallet_trades(w, tx, sig, block_time, now, sol_price)
            new_trades.extend(trades)
            newest_processed_sig = sig

        state[w.address] = {
            "last_signature": newest_processed_sig,
            "last_polled_at": now.isoformat(),
            "label": w.label,
        }

    _append_trades_csv(new_trades)
    _save_state(state)

    return {
        "timestamp": now.isoformat(),
        "wallets_tracked": len(wallets),
        "wallet_list_warnings": warnings,
        "rpc_calls_made": rpc_calls,
        "tx_fetch_failures": tx_fetch_failures,
        "new_trades_found": len(new_trades),
        "sol_price_usd_used": sol_price,
        "trades": [dataclasses.asdict(t) | {"block_time": t.block_time.isoformat(), "detected_at": t.detected_at.isoformat()} for t in new_trades],
    }


def estimate_full_cycle_seconds(
    n_wallets: int, requests_per_second_budget: float = solana_rpc_client.REQUESTS_PER_SECOND_BUDGET,
    avg_calls_per_wallet: float = 1.3,
) -> float:
    """Grobe Schätzung der Zyklusdauer für n_wallets bei gegebenem RPC-Budget.
    avg_calls_per_wallet=1.3 nimmt an, dass im Schnitt jede 3. Wallet EINE
    neue Transaktion seit dem letzten Poll hat (1 Signaturen-Call + anteilig
    einen Transaktions-Call) - reine Planungsannahme, keine Messung. Bei
    aktiveren Wallet-Listen (mehr neue Trades pro Zyklus) steigt dieser Wert
    entsprechend, siehe Modul-Docstring."""
    total_calls = n_wallets * avg_calls_per_wallet
    return total_calls / requests_per_second_budget


if __name__ == "__main__":
    result = run_once()
    print(json.dumps(result, indent=2, ensure_ascii=False))
