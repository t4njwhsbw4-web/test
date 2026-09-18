"""Pollt Dexscreener nach Solana-Tokens, die seit < MAX_AGE_MINUTES existieren.

Ein einzelner Aufruf von scan() entspricht einem "Poll" - das Skript ist kein
Dauerprozess, sondern für wiederholten Aufruf (z.B. via Cron alle 5-10 Minuten)
gedacht. Siehe dexscreener_client.py für die ausführliche Dokumentation der
Datenquelle und ihrer Grenzen.
"""
from __future__ import annotations

from .dexscreener_client import fetch_fresh_candidates
from .models import Candidate

MAX_AGE_MINUTES = 20


def scan(max_age_minutes: float = MAX_AGE_MINUTES) -> list[Candidate]:
    """Ein Poll: liefert alle aktuell auffindbaren Solana-Tokens, die jünger
    als max_age_minutes sind. Kann leer sein - das ist ein valides Ergebnis,
    kein Fehler (siehe Docstring in dexscreener_client.py: wir sehen nur einen
    kuratierten Ausschnitt aller Launches, nicht den vollständigen Feed)."""
    return fetch_fresh_candidates(max_age_seconds=max_age_minutes * 60)
