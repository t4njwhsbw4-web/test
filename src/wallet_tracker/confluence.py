"""Konfluenz-Signal: mehrere beobachtete Wallets kaufen/verkaufen denselben
Token innerhalb eines kurzen Zeitfensters. Das ist das eigentliche Signal
dieses Moduls - eine einzelne "Smart-Money"-Wallet zu kopieren ist laut der
bereits vorhandenen Recherche riskant (Imitation Penalty, Bot-Manipulation,
Survivorship-Bias bei "guten" Wallets); wenn dagegen MEHRERE unabhängige,
manuell kuratierte Wallets im selben kurzen Fenster denselben Token anfassen,
ist die Wahrscheinlichkeit eines Zufallstreffers oder einer gezielten
Einzel-Manipulation geringer (auch wenn koordinierte Gruppen/Insider-Cluster
das ebenfalls faken könnten - dieses Signal ist ein Heuristik-Baustein, kein
Beweis für "echte" Smart Money).

WICHTIG zur Zeitbasis: Konfluenz wird über block_time (on-chain-Zeitpunkt)
berechnet, NICHT über detected_at - das beschreibt also, wann die Wallets
TATSÄCHLICH gehandelt haben, nicht wann wir es gesehen haben. Ob und wann WIR
ein Konfluenz-Signal erkennen können, hängt zusätzlich von
detection_latency_seconds der beteiligten Trades ab (siehe fetcher.py) - im
schlechtesten Fall wird ein Signal erst erkannt, nachdem die LETZTE der
beteiligten Wallets mit typischer Verzögerung gepollt wurde.
"""
from __future__ import annotations

import csv
import dataclasses
import datetime as dt
import json
from pathlib import Path

from .fetcher import ARTIFACT_DIR, TRADES_CSV_PATH
from .models import WalletTrade

CONFLUENCE_CSV_PATH = ARTIFACT_DIR / "confluence_signals.csv"

# Benannte Default-Parameter (nicht hart im Code verdrahtet, hier zentral
# dokumentiert und beim Aufruf überschreibbar):
DEFAULT_MIN_WALLETS_FOR_CONFLUENCE = 3  # ab wie vielen unterschiedlichen Wallets ein Kauf/Verkauf als "Konfluenz" zählt
DEFAULT_CONFLUENCE_WINDOW_MINUTES = 15.0  # Zeitfenster (on-chain block_time), in dem diese Wallets gehandelt haben müssen


@dataclasses.dataclass
class ConfluenceSignal:
    token_mint: str
    action: str  # "buy" | "sell"
    window_start: dt.datetime
    window_end: dt.datetime
    wallet_addresses: list[str]
    tx_signatures: list[str]
    min_wallets_required: int
    window_minutes_used: float

    @property
    def wallet_count(self) -> int:
        return len(self.wallet_addresses)


def _load_trades_csv(path: Path | str = TRADES_CSV_PATH) -> list[WalletTrade]:
    path = Path(path)
    if not path.exists():
        return []
    trades: list[WalletTrade] = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            trades.append(WalletTrade(
                wallet_address=row["wallet_address"],
                wallet_label=row["wallet_label"] or None,
                token_mint=row["token_mint"],
                action=row["action"],
                amount_tokens=float(row["amount_tokens"]),
                amount_sol=float(row["amount_sol"]) if row["amount_sol"] not in ("", None) else None,
                estimated_usd=float(row["estimated_usd"]) if row["estimated_usd"] not in ("", None) else None,
                block_time=dt.datetime.fromisoformat(row["block_time"]),
                detected_at=dt.datetime.fromisoformat(row["detected_at"]),
                detection_latency_seconds=float(row["detection_latency_seconds"]),
                tx_signature=row["tx_signature"],
                slot=int(row["slot"]) if row.get("slot") not in ("", None) else None,
            ))
    return trades


def detect_confluence(
    trades: list[WalletTrade],
    action: str,
    min_wallets: int = DEFAULT_MIN_WALLETS_FOR_CONFLUENCE,
    window_minutes: float = DEFAULT_CONFLUENCE_WINDOW_MINUTES,
) -> list[ConfluenceSignal]:
    """Findet Zeitfenster (Länge window_minutes, gleitend über block_time),
    in denen mindestens min_wallets UNTERSCHIEDLICHE Wallets denselben Token
    mit derselben `action` ("buy" oder "sell") gehandelt haben.

    Einfache O(n^2)-Sliding-Window-Implementierung pro (Token, Action)-Gruppe
    - für die hier realistische Grössenordnung (Trades pro Token in einem
    Poll-Zyklus, nicht Millionen Zeilen) bewusst simpel statt cleverer
    Datenstruktur gehalten. Sobald ein Fenster den Schwellwert erreicht, wird
    EIN Signal für die früheste Trefferkonstellation emittiert und der
    Scan hinter dem ersten beteiligten Trade fortgesetzt (verhindert
    Duplikat-Meldungen für praktisch dasselbe, nur leicht verschobene Fenster)."""
    window = dt.timedelta(minutes=window_minutes)
    by_token: dict[str, list[WalletTrade]] = {}
    for t in trades:
        if t.action != action:
            continue
        by_token.setdefault(t.token_mint, []).append(t)

    signals: list[ConfluenceSignal] = []
    for token_mint, token_trades in by_token.items():
        token_trades = sorted(token_trades, key=lambda t: t.block_time)
        i = 0
        n = len(token_trades)
        while i < n:
            j = i
            wallets_in_window: dict[str, WalletTrade] = {}
            while j < n and token_trades[j].block_time - token_trades[i].block_time <= window:
                wallets_in_window.setdefault(token_trades[j].wallet_address, token_trades[j])
                j += 1
            if len(wallets_in_window) >= min_wallets:
                members = list(wallets_in_window.values())
                signals.append(ConfluenceSignal(
                    token_mint=token_mint,
                    action=action,
                    window_start=min(m.block_time for m in members),
                    window_end=max(m.block_time for m in members),
                    wallet_addresses=[m.wallet_address for m in members],
                    tx_signatures=[m.tx_signature for m in members],
                    min_wallets_required=min_wallets,
                    window_minutes_used=window_minutes,
                ))
                i += 1  # hinter den ersten beteiligten Trade weiterscannen, nicht dasselbe Fenster nochmal melden
            else:
                i += 1
    return signals


def _write_confluence_csv(signals: list[ConfluenceSignal], path: Path | str = CONFLUENCE_CSV_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["token_mint", "action", "window_start", "window_end", "wallet_count",
              "min_wallets_required", "window_minutes_used", "wallet_addresses", "tx_signatures"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for s in signals:
            writer.writerow({
                "token_mint": s.token_mint, "action": s.action,
                "window_start": s.window_start.isoformat(), "window_end": s.window_end.isoformat(),
                "wallet_count": s.wallet_count, "min_wallets_required": s.min_wallets_required,
                "window_minutes_used": s.window_minutes_used,
                "wallet_addresses": "|".join(s.wallet_addresses),
                "tx_signatures": "|".join(s.tx_signatures),
            })


if __name__ == "__main__":
    all_trades = _load_trades_csv()
    buy_signals = detect_confluence(all_trades, action="buy")
    sell_signals = detect_confluence(all_trades, action="sell")
    combined = buy_signals + sell_signals
    _write_confluence_csv(combined)
    print(json.dumps({
        "trades_loaded": len(all_trades),
        "min_wallets_for_confluence": DEFAULT_MIN_WALLETS_FOR_CONFLUENCE,
        "confluence_window_minutes": DEFAULT_CONFLUENCE_WINDOW_MINUTES,
        "buy_confluence_signals": len(buy_signals),
        "sell_confluence_signals": len(sell_signals),
        "signals_written_to": str(CONFLUENCE_CSV_PATH),
    }, indent=2, ensure_ascii=False))
