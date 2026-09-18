"""Verknüpfungsmodul zwischen paper_memecoin und wallet_tracker.

Liest (rein LESEND, keine Schreibzugriffe - der wallet_tracker-Cron-Job
schreibt diese Dateien parallel selbst) die vom wallet_tracker erzeugten
CSVs:
    - artifacts/wallet_tracker/wallet_trades.csv       (fetcher.py)
    - artifacts/wallet_tracker/confluence_signals.csv  (confluence.py,
      existiert erst NACHDEM confluence.py mindestens einmal gelaufen ist)

und beantwortet für einen gegebenen Token-Mint die Frage: "Wurde dieser Mint
innerhalb der letzten `window_minutes` von mindestens `min_wallets` der vom
Nutzer als 'beste Trader' identifizierten Wallets aktiv GEKAUFT?"

DAS IST DIE TESTBARE NUTZER-HYPOTHESE, KEIN BEWIESENER FAKT:
Die generelle Rug-Rate bei frischen Sub-100k-Memecoins liegt laut zitierter
Studie bei ~99% (siehe filter.py-Docstring). Der Nutzer vermutet, dass die
BEDINGTE Erfolgswahrscheinlichkeit deutlich besser ist, WENN zusätzlich zum
Rug-Filter (score_candidate/apply_rugcheck) auch eine oder mehrere seiner
Tracked-Wallets den Coin aktiv kaufen - weil diese Wallets nachweislich gute
Trader sind. Das ist legitimes Bayes'sches Konditionieren auf ein zusätzliches
Signal, aber UNBELEGT, bis genug echte Trades unter dieser Bedingung
gesammelt und ausgewertet wurden (siehe strategy.WALLET_SIGNAL_HYPOTHESIS_PARAMS
und run_loop.py, Strategie-Tag "wallet_signal" in trades.csv).

ROBUSTHEIT (explizit gefordert): Beide CSVs können fehlen oder leer sein -
die wallet_tracker-Cron-Jobs laufen erst seit kurzem. In jedem dieser Fälle
liefert check_wallet_signal() sauber "kein Signal" (matched=False,
wallet_count=0, ...) zurück, niemals eine Exception.
"""
from __future__ import annotations

import csv
import dataclasses
import datetime as dt
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
WALLET_TRACKER_ARTIFACT_DIR = REPO_ROOT / "artifacts" / "wallet_tracker"
WALLET_TRADES_CSV_PATH = WALLET_TRACKER_ARTIFACT_DIR / "wallet_trades.csv"
CONFLUENCE_SIGNALS_CSV_PATH = WALLET_TRACKER_ARTIFACT_DIR / "confluence_signals.csv"

DEFAULT_WINDOW_MINUTES = 30.0
DEFAULT_MIN_WALLETS = 1


@dataclasses.dataclass
class WalletSignalResult:
    """Ergebnis der Prüfung für EINEN Token-Mint."""

    token_mint: str
    matched: bool  # True, wenn min_wallets erreicht wurde
    wallet_count: int  # Anzahl unterschiedlicher Tracked-Wallets mit Buy im Fenster
    wallet_addresses: list[str]
    wallet_labels: list[str | None]
    most_recent_buy_age_seconds: float | None  # None, wenn wallet_count == 0
    oldest_buy_age_seconds: float | None
    min_wallets_required: int
    window_minutes: float
    source: str  # "wallet_trades" oder "wallet_trades+confluence_signals" (informativ)


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _read_csv_rows(path: Path) -> list[dict]:
    """Liest eine CSV robust ein. Fehlende Datei, leere Datei oder eine mit
    nur Header -> leere Liste, NIE eine Exception."""
    if not path.exists():
        return []
    try:
        with open(path, "r", newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except (OSError, csv.Error):
        return []


def _parse_dt(value: str | None) -> dt.datetime | None:
    if not value:
        return None
    try:
        parsed = dt.datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed


def _buys_from_wallet_trades(token_mint: str, window_start: dt.datetime, now: dt.datetime) -> dict[str, tuple[str | None, dt.datetime]]:
    """Liest wallet_trades.csv (fetcher.py-Schema) und liefert pro
    Wallet-Adresse den JÜNGSTEN Buy-Zeitpunkt (block_time) für diesen Mint
    innerhalb des Fensters. Zeitbasis: block_time (on-chain-Zeitpunkt, analog
    zu confluence.py), nicht detected_at."""
    result: dict[str, tuple[str | None, dt.datetime]] = {}
    for row in _read_csv_rows(WALLET_TRADES_CSV_PATH):
        if row.get("token_mint") != token_mint:
            continue
        if row.get("action") != "buy":
            continue
        block_time = _parse_dt(row.get("block_time"))
        if block_time is None:
            continue
        if block_time < window_start or block_time > now:
            continue
        address = row.get("wallet_address")
        if not address:
            continue
        label = row.get("wallet_label") or None
        existing = result.get(address)
        if existing is None or block_time > existing[1]:
            result[address] = (label, block_time)
    return result


def _buys_from_confluence_signals(token_mint: str, window_start: dt.datetime, now: dt.datetime) -> dict[str, tuple[str | None, dt.datetime]]:
    """Ergänzend: confluence_signals.csv (confluence.py-Schema, bereits
    aggregierte Mehr-Wallet-Signale). Nur "buy"-Action-Zeilen, nur für diesen
    Mint, Fenster über window_end (spätester on-chain-Zeitpunkt im Signal).
    Labels sind in dieser CSV nicht enthalten (nur Adressen) -> None."""
    result: dict[str, tuple[str | None, dt.datetime]] = {}
    for row in _read_csv_rows(CONFLUENCE_SIGNALS_CSV_PATH):
        if row.get("token_mint") != token_mint:
            continue
        if row.get("action") != "buy":
            continue
        window_end = _parse_dt(row.get("window_end"))
        if window_end is None:
            continue
        if window_end < window_start or window_end > now:
            continue
        addresses = (row.get("wallet_addresses") or "").split("|")
        for address in addresses:
            address = address.strip()
            if not address:
                continue
            existing = result.get(address)
            if existing is None or window_end > existing[1]:
                result[address] = (None, window_end)
    return result


def check_wallet_signal(
    token_mint: str,
    min_wallets: int = DEFAULT_MIN_WALLETS,
    window_minutes: float = DEFAULT_WINDOW_MINUTES,
    now: dt.datetime | None = None,
) -> WalletSignalResult:
    """Prüft, ob für `token_mint` innerhalb von `window_minutes` mindestens
    `min_wallets` unterschiedliche Tracked-Wallets einen Kauf (action="buy")
    verzeichnet haben - kombiniert aus wallet_trades.csv (Einzel-Trades) und,
    falls vorhanden, confluence_signals.csv (bereits aggregierte
    Mehr-Wallet-Fenster). Beide Quellen werden über die Wallet-Adresse
    dedupliziert (dieselbe Wallet zählt nur einmal).

    Robust gegen fehlende/leere CSVs (Cron-Jobs laufen ggf. erst seit kurzem):
    liefert in diesem Fall matched=False, wallet_count=0 - kein Crash."""
    now = now or _now()
    window_start = now - dt.timedelta(minutes=window_minutes)

    combined: dict[str, tuple[str | None, dt.datetime]] = {}
    combined.update(_buys_from_wallet_trades(token_mint, window_start, now))

    confluence_matches = _buys_from_confluence_signals(token_mint, window_start, now)
    used_confluence = bool(confluence_matches)
    for address, (label, ts) in confluence_matches.items():
        existing = combined.get(address)
        if existing is None:
            combined[address] = (label, ts)
        elif existing[0] is None and label is not None:
            combined[address] = (label, existing[1])

    wallet_count = len(combined)
    wallet_addresses = sorted(combined.keys())
    wallet_labels = [combined[a][0] for a in wallet_addresses]

    if wallet_count > 0:
        timestamps = [combined[a][1] for a in wallet_addresses]
        most_recent_age = (now - max(timestamps)).total_seconds()
        oldest_age = (now - min(timestamps)).total_seconds()
    else:
        most_recent_age = None
        oldest_age = None

    return WalletSignalResult(
        token_mint=token_mint,
        matched=wallet_count >= min_wallets,
        wallet_count=wallet_count,
        wallet_addresses=wallet_addresses,
        wallet_labels=wallet_labels,
        most_recent_buy_age_seconds=most_recent_age,
        oldest_buy_age_seconds=oldest_age,
        min_wallets_required=min_wallets,
        window_minutes=window_minutes,
        source="wallet_trades+confluence_signals" if used_confluence else "wallet_trades",
    )
