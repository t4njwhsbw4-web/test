"""Gemeinsame Datenstrukturen für das Wallet-Tracking-Modul.

Analog zu src/paper_memecoin/models.py (Candidate/FilterResult) gehalten:
einfache Dataclasses, keine ORM/Pydantic-Abhängigkeit, Felder die eine Quelle
NICHT liefert werden explizit als None geführt statt erfunden.
"""
from __future__ import annotations

import dataclasses
import datetime as dt


@dataclasses.dataclass
class WatchedWallet:
    """Eine beobachtete Solana-Wallet-Adresse.

    address: Solana-Base58-Pubkey (der Wallet-OWNER, nicht ein Token-Account).
    label:   optionale menschenlesbare Notiz (z.B. "axiom_top1" oder
             "creator_TROLLFACE") - rein informativ, keine Programmlogik
             hängt davon ab. None, wenn keine Label-Spalte in der Quelldatei
             vorhanden war.
    """

    address: str
    label: str | None = None


@dataclasses.dataclass
class WalletTrade:
    """Ein erkanntes Buy- oder Sell-Event einer beobachteten Wallet bei einem
    SPL-Token, abgeleitet aus einer einzelnen on-chain Transaktion.

    Felder, die die Solana-RPC-Quelle NICHT direkt liefert, sind bewusst als
    Optional/None geführt statt geschätzt zu erfinden - siehe fetcher.py für
    die genaue Herleitung jedes Feldes.
    """

    wallet_address: str
    wallet_label: str | None
    token_mint: str
    action: str  # "buy" | "sell"
    amount_tokens: float  # absoluter Betrag der Token-Mengenänderung (>0)
    amount_sol: float | None  # SOL-Gegenwert lt. Pre/Post-SOL-Balance-Delta derselben Tx, None wenn nicht sauber zuordenbar (z.B. Multi-Hop-Swap über mehrere Signer)
    estimated_usd: float | None  # NÄHERUNG: amount_sol * AKTUELLER SOL-Preis zum Zeitpunkt des Fetchens, NICHT der historische Preis zum Tx-Zeitpunkt (siehe fetcher.py)
    block_time: dt.datetime  # on-chain Zeitstempel der Transaktion (UTC)
    detected_at: dt.datetime  # Zeitpunkt, zu dem WIR die Tx über die Quelle gesehen haben (UTC)
    detection_latency_seconds: float  # detected_at - block_time, siehe fetcher.py-Docstring ("Imitation Penalty")
    tx_signature: str
    slot: int | None = None
