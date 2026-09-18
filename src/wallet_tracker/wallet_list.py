"""Lädt die Liste beobachteter Wallets aus einer einfachen Textdatei.

DATEIFORMAT (eine Adresse pro Zeile):
    <solana_adresse>
    <solana_adresse>,<optionales label>

    - Leerzeilen und Zeilen, die mit "#" beginnen, werden ignoriert (Kommentare).
    - Whitespace um Adresse/Label wird getrimmt.
    - Das Label ist rein informativ (z.B. "axiom_rank_12" oder ein Alias)
      und darf Kommas NICHT enthalten (kein CSV-Quoting implementiert - bei
      Bedarf später auf csv.reader umstellen, aktuell bewusst simpel gehalten
      analog zum Rest des Repos).
    - Duplikate (gleiche Adresse mehrfach) werden beim Laden entfernt, das
      erste Vorkommen (inkl. dessen Label) gewinnt; eine Warnung wird
      zurückgegeben, kein Exception.

Das ist EXAKT das Format, in dem die echte ~1000-Adressen-Liste vom Nutzer
später eingespeist werden soll: einfach die Datei unter dem in
DEFAULT_WALLET_LIST_PATH genannten Pfad ablegen (oder einen eigenen Pfad an
load_watched_wallets() übergeben) - kein Format-Wechsel, kein Code-Change
nötig.

Keine Adress-Validierung gegen die Solana-Kurve (kein base58-Decode/Length-
Check) - eine ungültige Adresse fällt beim ersten Poll-Versuch in fetcher.py
schlicht mit einem RPC-Fehler für genau diese Wallet auf, alle anderen laufen
unbeeinflusst weiter.
"""
from __future__ import annotations

from pathlib import Path

from .models import WatchedWallet

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = REPO_ROOT / "artifacts" / "wallet_tracker"
DEFAULT_WALLET_LIST_PATH = ARTIFACT_DIR / "watched_wallets.txt"
EXAMPLE_WALLET_LIST_PATH = ARTIFACT_DIR / "watched_wallets.example.txt"


def load_watched_wallets(path: Path | str | None = None) -> tuple[list[WatchedWallet], list[str]]:
    """Lädt Wallets aus `path` (Default: DEFAULT_WALLET_LIST_PATH - die ECHTE
    Liste, sobald sie existiert). Gibt (wallets, warnings) zurück; warnings
    enthält z.B. Hinweise zu übersprungenen Duplikaten. Existiert die Datei
    nicht, wird eine leere Liste zurückgegeben (kein Absturz), damit
    fetcher.py auch VOR Ankunft der echten Liste sauber "0 Wallets" meldet
    statt zu crashen."""
    target = Path(path) if path is not None else DEFAULT_WALLET_LIST_PATH
    if not target.exists():
        return [], [f"Wallet-Liste {target} existiert nicht - noch keine Adressen zu tracken."]

    wallets: list[WatchedWallet] = []
    warnings: list[str] = []
    seen: set[str] = set()

    for lineno, raw_line in enumerate(target.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(",", 1)
        address = parts[0].strip()
        label = parts[1].strip() if len(parts) > 1 and parts[1].strip() else None
        if not address:
            warnings.append(f"Zeile {lineno}: leere Adresse übersprungen.")
            continue
        if address in seen:
            warnings.append(f"Zeile {lineno}: Duplikat von '{address}' übersprungen.")
            continue
        seen.add(address)
        wallets.append(WatchedWallet(address=address, label=label))

    return wallets, warnings
