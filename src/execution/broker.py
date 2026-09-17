"""Broker-Abstraktion: austauschbare Ausführungsschicht.

PaperBroker simuliert Fills lokal (kein Netzwerk, kein Konto nötig) und ist
der Default für jede Entwicklungs- und Validierungsphase.

AlpacaBroker verbindet sich mit Alpacas Paper- ODER Live-Endpoint. Live
erfordert eine EXPLIZITE Bestätigung im Code-Aufruf (confirm_live=True) -
ein falsch gesetztes Config-Flag allein reicht nicht, um mit echtem Geld
zu handeln.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import os
from abc import ABC, abstractmethod


@dataclasses.dataclass
class Position:
    symbol: str
    qty: float
    entry_price: float


@dataclasses.dataclass
class Order:
    symbol: str
    qty: float
    side: str  # "buy" | "sell"
    filled_price: float
    timestamp: dt.datetime


class Broker(ABC):
    @abstractmethod
    def get_equity(self) -> float: ...

    @abstractmethod
    def get_position(self, symbol: str) -> Position | None: ...

    @abstractmethod
    def submit_order(self, symbol: str, qty: float, side: str, price: float) -> Order: ...


class PaperBroker(Broker):
    """Simuliert Fills exakt zum übergebenen Preis (kein Slippage-Modell -
    bewusst konservativ vereinfacht für die erste Ausbaustufe)."""

    def __init__(self, starting_cash: float):
        self._cash = starting_cash
        self._positions: dict[str, Position] = {}
        self._orders: list[Order] = []

    def get_equity(self) -> float:
        positions_value = sum(p.qty * p.entry_price for p in self._positions.values())
        return self._cash + positions_value

    def get_position(self, symbol: str) -> Position | None:
        return self._positions.get(symbol)

    def submit_order(self, symbol: str, qty: float, side: str, price: float) -> Order:
        if side not in ("buy", "sell"):
            raise ValueError(f"Ungültige Order-Seite: {side!r}")

        cost = qty * price
        if side == "buy":
            if cost > self._cash:
                raise ValueError(f"Nicht genug Cash für Order: benötigt {cost:.2f}, verfügbar {self._cash:.2f}")
            self._cash -= cost
            existing = self._positions.get(symbol)
            if existing:
                total_qty = existing.qty + qty
                avg_price = (existing.qty * existing.entry_price + cost) / total_qty
                self._positions[symbol] = Position(symbol, total_qty, avg_price)
            else:
                self._positions[symbol] = Position(symbol, qty, price)
        else:
            existing = self._positions.get(symbol)
            if not existing or existing.qty < qty:
                raise ValueError(f"Nicht genug Bestand von {symbol!r} zum Verkauf.")
            self._cash += cost
            remaining = existing.qty - qty
            if remaining <= 1e-9:
                del self._positions[symbol]
            else:
                self._positions[symbol] = Position(symbol, remaining, existing.entry_price)

        order = Order(symbol=symbol, qty=qty, side=side, filled_price=price, timestamp=dt.datetime.utcnow())
        self._orders.append(order)
        return order

    @property
    def order_history(self) -> list[Order]:
        return list(self._orders)


class AlpacaBroker(Broker):
    """Dünner Wrapper um alpaca-py. Erfordert ALPACA_API_KEY / ALPACA_SECRET_KEY
    als Umgebungsvariablen (z.B. via .env, NIEMALS im Code oder in config.yaml)."""

    def __init__(self, confirm_live: bool = False):
        try:
            from alpaca.trading.client import TradingClient
        except ImportError as exc:
            raise ImportError("alpaca-py ist nicht installiert (siehe requirements.txt).") from exc

        api_key = os.environ.get("ALPACA_API_KEY")
        secret_key = os.environ.get("ALPACA_SECRET_KEY")
        if not api_key or not secret_key:
            raise RuntimeError(
                "ALPACA_API_KEY / ALPACA_SECRET_KEY fehlen in der Umgebung. "
                "Ohne diese Variablen wird bewusst keine Verbindung aufgebaut."
            )

        if not confirm_live:
            paper = True
        else:
            paper = False
            print(
                "!!! LIVE-HANDEL MIT ECHTEM GELD AKTIVIERT (confirm_live=True) !!!\n"
                "Stelle sicher, dass Paper-Trading und Walk-Forward-Validierung "
                "erfolgreich abgeschlossen wurden, bevor du fortfährst."
            )

        self._client = TradingClient(api_key, secret_key, paper=paper)

    def get_equity(self) -> float:
        account = self._client.get_account()
        return float(account.equity)

    def get_position(self, symbol: str) -> Position | None:
        try:
            pos = self._client.get_open_position(symbol.replace("/", ""))
        except Exception:
            return None
        return Position(symbol=symbol, qty=float(pos.qty), entry_price=float(pos.avg_entry_price))

    def submit_order(self, symbol: str, qty: float, side: str, price: float) -> Order:
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest

        order_side = OrderSide.BUY if side == "buy" else OrderSide.SELL
        request = MarketOrderRequest(
            symbol=symbol.replace("/", ""),
            qty=qty,
            side=order_side,
            time_in_force=TimeInForce.DAY,
        )
        result = self._client.submit_order(request)
        return Order(
            symbol=symbol,
            qty=qty,
            side=side,
            filled_price=price,
            timestamp=dt.datetime.utcnow(),
        )
