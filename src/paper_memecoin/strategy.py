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


# --------------------------------------------------------------------------
# ZWEITE, SEPARATE Strategie-Variante: explizite Nutzer-Hypothese
# "~20k USD Einstieg, ~80k USD Ziel" = 4x / +300% auf MARKETCAP-Basis, nicht
# auf Prozent-Preis-Basis. Läuft parallel zur generischen TP/SL-Baseline oben,
# auf denselben Filter-Kandidaten, aber als eigene, separat geloggte und
# auswertbare Position (Tag "mcap_hypothesis" in trades.csv/state.json) -
# damit im Bericht konkret beantwortet werden kann: "hätte 20k->80k Mcap
# funktioniert?", getrennt von der generischen %-Baseline.
#
# entry_mcap_range: nur Kandidaten, deren aktueller marketCap (Dexscreener)
#   in diesem Fenster liegt, werden unter dieser Strategie überhaupt als
#   Einstiegskandidat gewertet (20k ist die Mitte des Fensters, nicht exakt
#   erzwungen, weil ein Poll den Coin selten exakt bei 20.000 USD erwischt).
# target_mcap_multiple: Exit bei current_mcap >= entry_mcap * multiple.
#   4.0 entspricht exakt "20k -> 80k" (NICHT gerundet auf +200%/"verdoppelt").
MCAP_HYPOTHESIS_PARAMS = {
    "entry_mcap_range": (10_000.0, 30_000.0),
    "target_mcap_multiple": 4.0,
    "stop_loss_pct": STOP_LOSS_PCT,  # gleiche Kapitalschutz-Regel wie Baseline (-30%)
    "max_hold_hours": MAX_HOLD_HOURS,  # gleiches Zeitlimit wie Baseline (6h)
}


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


def mcap_hypothesis_entry_eligible(market_cap: float | None) -> bool:
    """True, wenn der aktuelle Marketcap im Einstiegsfenster der
    Nutzer-Hypothese liegt (siehe MCAP_HYPOTHESIS_PARAMS)."""
    if market_cap is None:
        return False
    lo, hi = MCAP_HYPOTHESIS_PARAMS["entry_mcap_range"]
    return lo <= market_cap <= hi


def should_exit_mcap(entry_mcap: float, current_mcap: float | None,
                      entry_price: float, current_price: float,
                      entry_time: dt.datetime, now: dt.datetime) -> tuple[bool, str] | tuple[bool, None]:
    """Exit-Regeln der Mcap-Hypothese: Take-Profit auf MARKETCAP-Basis
    (current_mcap >= entry_mcap * target_mcap_multiple), Stop-Loss weiterhin
    auf Preisbasis (Mcap und Preis bewegen sich bei konstantem Supply
    proportional, Stop-Loss bleibt daher auf dem robusteren Preisvergleich)."""
    params = MCAP_HYPOTHESIS_PARAMS
    if current_mcap is not None and entry_mcap > 0:
        target_mcap = entry_mcap * params["target_mcap_multiple"]
        if current_mcap >= target_mcap:
            return True, f"mcap_take_profit ({current_mcap:,.0f} >= {target_mcap:,.0f} = {entry_mcap:,.0f} x {params['target_mcap_multiple']:.0f})"

    if entry_price > 0:
        change = (current_price - entry_price) / entry_price
        if change <= -params["stop_loss_pct"]:
            return True, f"mcap_stop_loss ({change:+.1%})"

    held_hours = (now - entry_time).total_seconds() / 3600.0
    if held_hours >= params["max_hold_hours"]:
        return True, f"mcap_max_hold_time ({held_hours:.1f}h)"
    return False, None
