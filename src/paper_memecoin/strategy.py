"""Baseline-Handelsregel - bewusst simpel und nachvollziehbar, keine Prognose.

Einstieg:
    Kaufe einen Kandidaten, wenn filter.score_candidate() ihn durchlässt
    (passed=True, siehe filter.py für die Schwellen) UND aktuell keine
    offene Position in diesem Token besteht UND genug Cash verfügbar ist.

Positionsgrösse:
    Fixer Anteil des AKTUELLEN Cash-Bestands pro Trade (nicht des Startkapitals),
    damit das System nach Verlusten nicht "pleite rechnet" und weiter mit
    Phantomgeld handelt.

Ausstieg (wer zuerst eintritt):
    - Take-Profit:  Preis >= Einstiegspreis * (1 + TAKE_PROFIT_PCT)
    - Stop-Loss:    Preis <= Einstiegspreis * (1 - STOP_LOSS_PCT)
    - Max-Haltedauer: Position seit >= MAX_HOLD_HOURS Stunden offen
    -> egal was zuerst zutrifft, es wird komplett verkauft (keine Teilverkäufe).
"""
from __future__ import annotations

import datetime as dt

TAKE_PROFIT_PCT = 0.50  # +50%
STOP_LOSS_PCT = 0.30  # -30%
MAX_HOLD_HOURS = 6.0

POSITION_SIZE_PCT_OF_CASH = 0.10  # 10% des aktuellen Cash pro neuer Position
MAX_OPEN_POSITIONS = 5


def position_size_usd(cash: float) -> float:
    return cash * POSITION_SIZE_PCT_OF_CASH


def should_exit(entry_price: float, current_price: float, entry_time: dt.datetime,
                 now: dt.datetime) -> tuple[bool, str] | tuple[bool, None]:
    """Prüft die drei Exit-Bedingungen in fester Reihenfolge (TP, SL, Zeit)."""
    if entry_price <= 0:
        return False, None
    change = (current_price - entry_price) / entry_price
    if change >= TAKE_PROFIT_PCT:
        return True, f"take_profit ({change:+.1%})"
    if change <= -STOP_LOSS_PCT:
        return True, f"stop_loss ({change:+.1%})"
    held_hours = (now - entry_time).total_seconds() / 3600.0
    if held_hours >= MAX_HOLD_HOURS:
        return True, f"max_hold_time ({held_hours:.1f}h)"
    return False, None
