"""Minimaler JSON-RPC-Client für die öffentliche Solana-Mainnet-RPC (kein
API-Key nötig). Analog im Stil zu src/paper_memecoin/dexscreener_client.py
gehalten (urllib, kein externes HTTP-Paket, defensive try/except statt
Exceptions durchzureichen).

GEPRÜFTE ERREICHBARKEIT + DATENTIEFE (2026-09-18, aus dieser Umgebung heraus,
siehe Bericht des Bauauftrags für den vollständigen Quellenvergleich):
- https://api.mainnet-beta.solana.com                    -> HTTP 200, keylos
  erreichbar, JSON-RPC 2.0. getHealth, getSignaturesForAddress, getTransaction
  liefern brauchbare Daten.
- https://api.helius.xyz/...                              -> HTTP 401 ohne
  Key (auch der "kostenlose" Tier verlangt einen API-Key) - für später
  vorgemerkt, aktuell NICHT genutzt.
- https://public-api.solscan.io/...                       -> Basis-URL
  antwortet (200), alle getesteten Account-/Transaktions-Endpunkte lieferten
  404 - die alte keylose "public-api"-Generation scheint grösstenteils
  abgeschaltet/auf die kostenpflichtige pro-api.solscan.io (v2, HTTP 401 ohne
  Key) migriert. NICHT nutzbar ohne Key.
- https://api.solscan.io/...                              -> von dieser
  Umgebung aus per Egress-Policy blockiert (502 auf CONNECT, siehe
  /root/.ccr/__agentproxy/status "connect_rejected"). NICHT nutzbar.
- https://gmgn.ai/api/... bzw. /defi/quotation/...         -> HTTP 403 auch
  mit Browser-User-Agent - vermutlich Cloudflare-Bot-Schutz/JS-Challenge, aus
  einem reinen HTTP-Client heraus nicht umgehbar. NICHT nutzbar.

=> Die öffentliche Solana-RPC ist aktuell die EINZIGE tatsächlich
funktionierende, keylose Quelle für Wallet-Transaktionshistorie in dieser
Umgebung.

DATENTIEFE der Solana-RPC (ehrlich, kein Bluff):
- getSignaturesForAddress liefert: Signatur, Slot, blockTime (Unix-Sekunden,
  UTC), err (None bei Erfolg), KEINE Token-/Betrags-Information.
- getTransaction (encoding=jsonParsed) liefert zusätzlich: pre-/post-
  TokenBalances (pro Token-Account: owner, mint, Menge VOR und NACH der Tx)
  sowie pre-/postBalances (SOL, lamports, pro Account-Index). Daraus lässt
  sich pro Wallet ableiten: welcher Mint hat sich wie verändert (Kauf/Verkauf)
  und wie viel SOL im selben Signer-Kontext den Besitzer gewechselt hat.
- KEIN direkter USD-Wert, KEIN historischer Preis zum Tx-Zeitpunkt - siehe
  fetcher.py für die Näherung, die daraus gebaut wird, und deren Grenzen.

RATE LIMITS (empirisch + offizielle Solana-Docs-Faustregel):
Ein kurzer Burst von 25 Requests in ~6s wurde beim Testlauf anstandslos mit
HTTP 200 beantwortet (keine 429 beobachtet) - das ist aber KEIN Beleg für
einen sicheren Dauerdurchsatz, sondern nur ein einzelner kurzer Burst zu
einer ruhigen Zeit. Die öffentliche RPC dokumentiert selbst keine festen
Zahlen, in der Praxis wird häufig ein Richtwert von ca. 40 Requests/10s
(~4 req/s) pro IP als konservativ genannt - REQUESTS_PER_SECOND_BUDGET unten
orientiert sich daran und bleibt bewusst darunter. Für produktiven Dauerbetrieb
mit 1000 Wallets ist das der limitierende Faktor, siehe fetcher.py.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

BASE_URL = "https://api.mainnet-beta.solana.com"
USER_AGENT = "smart-money-wallet-tracker/0.1 (+read-only-research-prototype)"
TIMEOUT_S = 20

# Konservatives Dauerbudget (siehe Docstring oben) - fetcher.py schläft
# zwischen einzelnen RPC-Calls mindestens 1/REQUESTS_PER_SECOND_BUDGET
# Sekunden, um dieses Budget nicht zu überschreiten.
REQUESTS_PER_SECOND_BUDGET = 3.0
MIN_REQUEST_INTERVAL_S = 1.0 / REQUESTS_PER_SECOND_BUDGET

_last_request_ts: float = 0.0


def _throttle() -> None:
    global _last_request_ts
    now = time.monotonic()
    wait = MIN_REQUEST_INTERVAL_S - (now - _last_request_ts)
    if wait > 0:
        time.sleep(wait)
    _last_request_ts = time.monotonic()


def _rpc_call(method: str, params: list, retries: int = 2) -> dict | None:
    """Ein JSON-RPC-Call mit einfachem Retry bei 429/Timeout. Gibt das
    'result'-Feld zurück, oder None bei endgültigem Fehler (kein stiller
    Fehlschlag - Aufrufer muss None behandeln, analog zu rugcheck_client.py)."""
    payload = json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params}).encode("utf-8")
    for attempt in range(retries + 1):
        _throttle()
        req = urllib.request.Request(
            BASE_URL, data=payload,
            headers={"Content-Type": "application/json", "User-Agent": USER_AGENT},
        )
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                if "error" in body:
                    return None
                return body.get("result")
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < retries:
                time.sleep(2.0 * (attempt + 1))
                continue
            return None
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            return None
    return None


def get_signatures_for_address(address: str, until: str | None = None, limit: int = 25) -> list[dict]:
    """Neueste Signaturen für eine Adresse, neueste zuerst. `until`: nur
    Signaturen NACH dieser (bereits bekannten) Signatur zurückgeben - so
    lässt sich inkrementell pollen, ohne alte Transaktionen erneut zu
    verarbeiten (siehe fetcher.py State-Handling)."""
    params_opts: dict = {"limit": limit}
    if until:
        params_opts["until"] = until
    result = _rpc_call("getSignaturesForAddress", [address, params_opts])
    return result if isinstance(result, list) else []


def get_transaction(signature: str) -> dict | None:
    """Volle geparste Transaktion (encoding=jsonParsed) oder None bei Fehler/
    noch nicht verfügbar (z.B. sehr frisch, noch nicht propagiert)."""
    return _rpc_call(
        "getTransaction",
        [signature, {"encoding": "jsonParsed", "maxSupportedTransactionVersion": 0}],
    )
