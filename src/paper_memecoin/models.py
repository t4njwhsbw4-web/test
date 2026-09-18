"""Gemeinsame Datenstrukturen für den Memecoin-Paper-Trading-Prototyp."""
from __future__ import annotations

import dataclasses
import datetime as dt


@dataclasses.dataclass
class Candidate:
    """Ein Solana-Token, wie ihn die Dexscreener-API liefert.

    Felder, die Dexscreener NICHT liefert (Holder-Anzahl, Holder-Konzentration,
    LP-Lock-Status, Dev-Wallet-Anteil), sind bewusst NICHT Teil dieser Klasse -
    sie existieren in der öffentlichen, keyless API schlicht nicht. Siehe
    filter.py für die ehrliche Dokumentation, was der Rug-Pull-Score daraus
    ableiten kann und was nicht.
    """

    chain_id: str
    dex_id: str
    token_address: str
    pair_address: str
    symbol: str
    name: str
    price_usd: float | None
    liquidity_usd: float | None  # None = nicht verfügbar (siehe filter.py)
    market_cap: float | None
    fdv: float | None
    pair_created_at: dt.datetime | None
    age_seconds: float | None
    volume_h1: float | None
    volume_m5: float | None
    txns_buys_h1: int | None
    txns_sells_h1: int | None
    has_website: bool
    has_social: bool
    source: str  # "token-profiles" oder "token-boosts" (Herkunft im Scan)

    @property
    def liquidity_to_mcap(self) -> float | None:
        if self.liquidity_usd is None or not self.market_cap:
            return None
        return self.liquidity_usd / self.market_cap

    @property
    def buy_sell_ratio_h1(self) -> float | None:
        if self.txns_buys_h1 is None or self.txns_sells_h1 is None:
            return None
        if self.txns_sells_h1 == 0:
            return float("inf") if self.txns_buys_h1 > 0 else None
        return self.txns_buys_h1 / self.txns_sells_h1


@dataclasses.dataclass
class FilterResult:
    candidate: Candidate
    score: float
    passed: bool
    reasons: list[str]
    unavailable_signals: list[str]


@dataclasses.dataclass
class Position:
    token_address: str
    symbol: str
    qty: float
    entry_price: float
    entry_time: str  # ISO8601
    opened_reason: str


@dataclasses.dataclass
class TradeLogEntry:
    timestamp: str
    token_address: str
    symbol: str
    action: str  # "BUY" | "SELL" | "SKIP"
    price: float | None
    qty: float | None
    reason: str
    equity_after: float | None = None
