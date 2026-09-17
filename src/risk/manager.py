"""Risk-Management, das dem Modell übergeordnet ist.

Design-Prinzip: Das Modell darf über RICHTUNG entscheiden (long/flat), aber
nie über GRÖSSE oder darüber, ob überhaupt gehandelt werden darf. Der
Kill-Switch kennt das Modell nicht und lässt sich von ihm nicht umgehen.
"""
from __future__ import annotations

import dataclasses
import datetime as dt


@dataclasses.dataclass
class RiskLimits:
    max_position_pct: float
    max_daily_loss_pct: float
    max_drawdown_pct: float
    stop_loss_pct: float
    take_profit_pct: float
    max_open_positions: int


@dataclasses.dataclass
class AccountState:
    equity: float
    starting_equity_today: float
    peak_equity: float
    open_positions: int


class KillSwitchTriggered(Exception):
    """Wird geworfen, wenn ein hartes Risikolimit verletzt ist. Muss vom
    Aufrufer so behandelt werden, dass KEIN neuer Trade mehr eröffnet wird."""


class RiskManager:
    def __init__(self, limits: RiskLimits):
        self.limits = limits
        self._halted = False
        self._halt_reason: str | None = None

    @property
    def halted(self) -> bool:
        return self._halted

    def check_kill_switch(self, account: AccountState) -> None:
        """Muss vor JEDEM neuen Trade aufgerufen werden."""
        if self._halted:
            raise KillSwitchTriggered(self._halt_reason or "Handel bereits gestoppt.")

        daily_pnl_pct = (account.equity - account.starting_equity_today) / account.starting_equity_today
        if daily_pnl_pct <= -self.limits.max_daily_loss_pct:
            self._halt(f"Tagesverlust-Limit erreicht: {daily_pnl_pct:.2%}")

        drawdown_pct = (account.equity - account.peak_equity) / account.peak_equity
        if drawdown_pct <= -self.limits.max_drawdown_pct:
            self._halt(f"Max-Drawdown-Limit erreicht: {drawdown_pct:.2%}")

        if self._halted:
            raise KillSwitchTriggered(self._halt_reason)

    def _halt(self, reason: str) -> None:
        self._halted = True
        self._halt_reason = reason

    def reset_daily(self) -> None:
        """Nur der Tagesverlust-Zähler wird täglich zurückgesetzt - Drawdown
        und ein einmal ausgelöster Halt NICHT automatisch, das erfordert
        manuelles Eingreifen (bewusst, kein Selbst-Reset des Kill-Switches)."""
        pass

    def position_size(self, account: AccountState, price: float) -> float:
        """Fixed-fractional Sizing: max. max_position_pct des Eigenkapitals
        pro Position, begrenzt zusätzlich durch max_open_positions."""
        if account.open_positions >= self.limits.max_open_positions:
            return 0.0
        capital_for_position = account.equity * self.limits.max_position_pct
        return capital_for_position / price if price > 0 else 0.0

    def stop_loss_price(self, entry_price: float) -> float:
        return entry_price * (1 - self.limits.stop_loss_pct)

    def take_profit_price(self, entry_price: float) -> float:
        return entry_price * (1 + self.limits.take_profit_pct)

    def should_exit(self, entry_price: float, current_price: float) -> str | None:
        if current_price <= self.stop_loss_price(entry_price):
            return "stop_loss"
        if current_price >= self.take_profit_price(entry_price):
            return "take_profit"
        return None
