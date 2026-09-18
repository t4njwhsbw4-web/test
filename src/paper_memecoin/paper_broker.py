"""Paper-Broker für dünne Memecoin-Märkte.

Adaptiert src/execution/broker.py::PaperBroker (unverändert dort, nur hier
per Vererbung erweitert): die Basisklasse simuliert Fills exakt zum
übergebenen Preis, was für liquide Assets als "bewusst konservativ" markiert
ist. Bei ganz frischen Sub-100k-MC-Solana-Launches ist Preis=Fill aber
UNREALISTISCH optimistisch - die Orderbücher/Bonding-Curves sind extrem dünn.
SlippageAwarePaperBroker wendet deshalb vor jedem Fill einen Slippage-Aufschlag
an (Kauf wird teurer, Verkauf bekommt weniger) statt 0% Slippage anzunehmen.

Ausschliesslich Paper-Trading: es wird kein Wallet, kein Private Key und keine
echte Transaktion irgendeiner Art verwendet. submit_order() verändert nur den
lokalen In-Memory-/JSON-Zustand dieses Prozesses.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # Repo-Root für src.*-Importe

from src.execution.broker import Order, PaperBroker, Position  # noqa: E402

# Dünne Märkte -> deutlich höhere Slippage-Annahme als bei liquiden Assets.
# Kauf: Preis wird um SLIPPAGE_BUY_PCT nach oben verschoben (wir zahlen mehr).
# Verkauf: Preis wird um SLIPPAGE_SELL_PCT nach unten verschoben (wir bekommen weniger).
SLIPPAGE_BUY_PCT = 0.04  # 4%
SLIPPAGE_SELL_PCT = 0.04  # 4%


class SlippageAwarePaperBroker(PaperBroker):
    """PaperBroker mit realistischer Slippage-Annahme für dünne Memecoin-Märkte."""

    def __init__(self, starting_cash: float,
                 slippage_buy_pct: float = SLIPPAGE_BUY_PCT,
                 slippage_sell_pct: float = SLIPPAGE_SELL_PCT):
        super().__init__(starting_cash=starting_cash)
        self.slippage_buy_pct = slippage_buy_pct
        self.slippage_sell_pct = slippage_sell_pct

    def submit_order(self, symbol: str, qty: float, side: str, price: float) -> Order:
        if side == "buy":
            effective_price = price * (1 + self.slippage_buy_pct)
        elif side == "sell":
            effective_price = price * (1 - self.slippage_sell_pct)
        else:
            effective_price = price  # löst in super() den ValueError für ungültige Seite aus
        return super().submit_order(symbol, qty, side, effective_price)


__all__ = ["SlippageAwarePaperBroker", "Order", "Position"]
