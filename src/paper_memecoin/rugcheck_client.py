"""Minimaler Client für die öffentliche RugCheck.xyz-API (api.rugcheck.xyz).

GEPRÜFTE ERREICHBARKEIT (2026-09-18, aus dieser Umgebung heraus):
- https://api.rugcheck.xyz/v1/tokens/{mint}/report          -> HTTP 200, ohne
  Key erreichbar, liefert u.a. mintAuthority, freezeAuthority, totalHolders,
  topHolders, creatorBalance, totalMarketLiquidity, rugged (bool), score,
  score_normalised, risks[] (menschenlesbare Warnungen wie "Mutable metadata",
  "High market cap per holder").
- Rate-Limit: der Response-Header `x-rate-limit-limit` zeigte in unseren
  Tests den Wert 15 (nicht die vom Auftrag genannten 20) - wir behandeln 15
  als beobachteten, KONSERVATIVEN Richtwert für das Tagesbudget ohne Key.
  Da unklar ist, ob sich das Fenster täglich oder rollierend zurücksetzt,
  wird hier zusätzlich ein lokaler Sicherheitsabstand eingehalten (siehe
  RUGCHECK_DAILY_BUDGET unten).
- WICHTIGE EINSCHRÄNKUNG aus den Live-Tests: `totalHolders` und `topHolders`
  waren bei mehreren ganz frischen (<~10h alten) Solana-Tokens leer/0 -
  RugCheck scheint Holder-Daten für brandneue Tokens (noch) nicht indiziert
  zu haben. Genau für unsere Zielgruppe (<20 Minuten) ist diese Kennzahl also
  oft NICHT verfügbar. Wir behandeln totalHolders==0 deshalb als "unbekannt",
  nicht als "0 Holder = hartes Fail".

Wegen des extrem knappen Tagesbudgets wird RugCheck NUR für Kandidaten
abgefragt, die den (kostenlosen) Dexscreener-Vorfilter (filter.py,
score_candidate) bereits bestanden haben - nicht für jeden Scan-Treffer.
"""
from __future__ import annotations

import datetime as dt
import json
import urllib.error
import urllib.request
from pathlib import Path

BASE_URL = "https://api.rugcheck.xyz/v1"
USER_AGENT = "paper-memecoin-scanner/0.1 (+forward-paper-trading-experiment)"
TIMEOUT_S = 15

# Beobachteter Header-Wert war 15; wir bleiben mit Absicht darunter, um nie
# versehentlich in die kostenpflichtige Stufe ($0.02/Request) zu rutschen.
RUGCHECK_DAILY_BUDGET = 12

QUOTA_PATH = Path(__file__).resolve().parents[2] / "artifacts" / "memecoin_paper" / "rugcheck_quota.json"


def _load_quota() -> dict:
    today = dt.date.today().isoformat()
    if QUOTA_PATH.exists():
        try:
            data = json.loads(QUOTA_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            data = {}
        if data.get("date") == today:
            return data
    return {"date": today, "used": 0}


def _save_quota(data: dict) -> None:
    QUOTA_PATH.parent.mkdir(parents=True, exist_ok=True)
    QUOTA_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def has_budget() -> bool:
    return _load_quota()["used"] < RUGCHECK_DAILY_BUDGET


def _record_call() -> None:
    data = _load_quota()
    data["used"] += 1
    _save_quota(data)


def fetch_report(mint: str) -> dict | None:
    """Voller RugCheck-Report für einen Mint. None bei Fehler, Budget-
    Erschöpfung oder wenn RugCheck den Token nicht kennt."""
    if not has_budget():
        return None
    _record_call()  # vor dem Request zählen - konservativ, zählt auch Fehlversuche
    req = urllib.request.Request(f"{BASE_URL}/tokens/{mint}/report", headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError):
        return None
