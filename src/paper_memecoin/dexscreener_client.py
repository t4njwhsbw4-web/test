"""Minimaler HTTP-Client für die öffentliche Dexscreener-API (kein API-Key nötig).

GEPRÜFTE ERREICHBARKEIT (2026-09-18, aus dieser Umgebung heraus):
- https://api.dexscreener.com/token-boosts/latest/v1      -> HTTP 200, funktioniert
- https://api.dexscreener.com/token-profiles/latest/v1    -> HTTP 200, funktioniert
- https://api.dexscreener.com/token-pairs/v1/solana/{adr} -> HTTP 200, ABER nur mit
  gesetztem User-Agent-Header (ohne UA -> 403 Forbidden)
- https://frontend-api.pump.fun/coins?...                 -> HTTP 530 / Cloudflare
  "error code: 1016" (Origin DNS error) - aus dieser Umgebung NICHT erreichbar.
  pump.fun scheint die inoffizielle Frontend-API auf Cloudflare-Ebene für
  Anfragen ausserhalb des Browser-Kontexts zu blocken.
- https://public-api.birdeye.so/...                       -> HTTP 401 ohne Key,
  also für dieses Projekt ohne (kostenpflichtigen) Key nicht nutzbar.

=> Dexscreener ist die einzige tatsächlich funktionierende, keyless Quelle.

WICHTIGE EINSCHRÄNKUNG (ehrlich dokumentiert, siehe auch filter.py):
Es gibt bei Dexscreener KEINEN öffentlichen "alle neuen Pairs in Echtzeit"-
Feed ohne Key. token-boosts/latest/v1 und token-profiles/latest/v1 liefern
stattdessen eine rotierende Liste von ~20-30 Tokens, die von Projekten aktiv
beworben ("boosted") bzw. mit einem Profil versehen wurden - NICHT jeder neue
pump.fun-Launch taucht dort auf, sondern nur ein (bezahlt kuratierter)
Ausschnitt. Wir filtern diese Liste anschliessend nach tatsächlichem Alter
(pairCreatedAt < 20 Minuten). Das heisst: Wir bekommen KEINEN vollständigen
Feed aller Launches, sondern eine Teilmenge, die zufällig sowohl beworben als
auch ganz frisch ist. Das begrenzt die Trefferquote pro Poll spürbar.
"""
from __future__ import annotations

import datetime as dt
import json
import urllib.error
import urllib.request

from .models import Candidate

BASE_URL = "https://api.dexscreener.com"
USER_AGENT = "paper-memecoin-scanner/0.1 (+forward-paper-trading-experiment)"
TIMEOUT_S = 15


def _get_json(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_boosted_solana_tokens() -> list[dict]:
    """Rotierende Liste beworbener Tokens, gefiltert auf chainId == solana."""
    try:
        data = _get_json(f"{BASE_URL}/token-boosts/latest/v1")
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError):
        return []
    if not isinstance(data, list):
        return []
    return [d for d in data if d.get("chainId") == "solana" and d.get("tokenAddress")]


def fetch_profiled_solana_tokens() -> list[dict]:
    """Rotierende Liste von Tokens mit Profil, gefiltert auf chainId == solana."""
    try:
        data = _get_json(f"{BASE_URL}/token-profiles/latest/v1")
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError):
        return []
    if not isinstance(data, list):
        return []
    return [d for d in data if d.get("chainId") == "solana" and d.get("tokenAddress")]


def fetch_pairs_for_token(token_address: str) -> list[dict]:
    """Alle bekannten Trading-Pairs für einen Solana-Token-Contract."""
    try:
        data = _get_json(f"{BASE_URL}/token-pairs/v1/solana/{token_address}")
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError):
        return []
    return data if isinstance(data, list) else []


def _best_pair(pairs: list[dict]) -> dict | None:
    """Wählt das Pair mit der jüngsten pairCreatedAt (bzw. dem höchsten Volumen,
    falls kein Zeitstempel vorhanden), falls ein Token mehrere Pools hat."""
    if not pairs:
        return None
    dated = [p for p in pairs if p.get("pairCreatedAt")]
    if dated:
        return max(dated, key=lambda p: p["pairCreatedAt"])
    return max(pairs, key=lambda p: (p.get("volume", {}) or {}).get("h24", 0) or 0)


def pair_to_candidate(pair: dict, source: str, now: dt.datetime | None = None) -> Candidate | None:
    if not pair:
        return None
    now = now or dt.datetime.now(dt.timezone.utc)
    base = pair.get("baseToken", {}) or {}
    created_ms = pair.get("pairCreatedAt")
    created_at = None
    age_seconds = None
    if created_ms:
        created_at = dt.datetime.fromtimestamp(created_ms / 1000, tz=dt.timezone.utc)
        age_seconds = (now - created_at).total_seconds()

    liquidity = pair.get("liquidity") or {}
    txns = pair.get("txns") or {}
    volume = pair.get("volume") or {}
    info = pair.get("info") or {}

    price_raw = pair.get("priceUsd")
    try:
        price_usd = float(price_raw) if price_raw is not None else None
    except (TypeError, ValueError):
        price_usd = None

    return Candidate(
        chain_id=pair.get("chainId", "solana"),
        dex_id=pair.get("dexId", "unknown"),
        token_address=base.get("address", ""),
        pair_address=pair.get("pairAddress", ""),
        symbol=base.get("symbol", "?"),
        name=base.get("name", "?"),
        price_usd=price_usd,
        liquidity_usd=liquidity.get("usd"),
        market_cap=pair.get("marketCap"),
        fdv=pair.get("fdv"),
        pair_created_at=created_at,
        age_seconds=age_seconds,
        volume_h1=(volume.get("h1")),
        volume_m5=(volume.get("m5")),
        txns_buys_h1=(txns.get("h1") or {}).get("buys"),
        txns_sells_h1=(txns.get("h1") or {}).get("sells"),
        has_website=bool(info.get("websites")),
        has_social=bool(info.get("socials")),
        source=source,
    )


def fetch_fresh_candidates(max_age_seconds: float) -> list[Candidate]:
    """Kombiniert boosts + profiles, holt für jeden Solana-Token die Pair-Daten
    und gibt nur Kandidaten zurück, die jünger als max_age_seconds sind."""
    seen_addresses: dict[str, str] = {}
    for entry in fetch_boosted_solana_tokens():
        seen_addresses.setdefault(entry["tokenAddress"], "token-boosts")
    for entry in fetch_profiled_solana_tokens():
        seen_addresses.setdefault(entry["tokenAddress"], "token-profiles")

    now = dt.datetime.now(dt.timezone.utc)
    fresh: list[Candidate] = []
    for token_address, source in seen_addresses.items():
        pairs = fetch_pairs_for_token(token_address)
        best = _best_pair(pairs)
        candidate = pair_to_candidate(best, source, now=now)
        if candidate is None:
            continue
        if candidate.age_seconds is None:
            continue  # kein Zeitstempel -> Alter unbekannt -> kann nicht als "frisch" gelten
        if candidate.age_seconds <= max_age_seconds:
            fresh.append(candidate)
    return fresh
