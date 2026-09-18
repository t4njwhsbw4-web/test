"""Methodische Korrektur der bisherigen Alters-Klassifikation im historischen
Wallet-Backtest (siehe wallet_backtest_report.py).

PROBLEM DER VORGAENGER-ANALYSE (Nutzerkritik, berechtigt): `age_at_buy_hours`
in historical_entries.json basiert auf Dexscreener `pairCreatedAt`. Das ist
(a) oft NULL fuer tote/delistete Coins - eine zirkulaere Verzerrung, denn "kein
Pool mehr auffindbar" heisst "keine Alters-Info mehr", VOELLIG UNABHAENGIG vom
echten Alter zum Kaufzeitpunkt. Und (b) selbst wenn vorhanden, ist es das
Datum der AMM-Migration (Bonding-Curve -> Raydium/PumpSwap/Meteora), NICHT das
echte Mint-Erstellungsdatum auf der pump.fun-Bonding-Curve. Ein Coin kann also
Tage/Wochen auf der Bonding-Curve gelebt haben, bevor er zu einem Dexscreener-
"Pair" wurde - `age_at_buy_hours` waere dann fuer einen laengst nicht mehr
"frischen" Kauf trotzdem klein oder sogar negativ (siehe die stark negativen
Werte in historical_entries.json, z.B. -1052h - das ist genau dieses Symptom,
nicht ein Datenfehler).

DIESES SKRIPT ERSETZT DIE ALTERS-BESTIMMUNG DURCH DIE BLOCKCHAIN SELBST:
Fuer jeden Mint aus historical_entries.json wird über die oeffentliche Solana-
RPC (solana_rpc_client.py, UNVERAENDERT wiederverwendet) die AELTESTE bekannte
Signatur fuer die MINT-ADRESSE selbst gesucht (getSignaturesForAddress mit der
Mint-Adresse, rueckwaerts per `before`-Cursor paginiert, bis entweder das Ende
der Historie erreicht ist ODER PAGES_CAP erreicht wird). Deren blockTime ist
ein Proxy fuer den Erstellungszeitpunkt - bei pump.fun-Mints i.d.R. die
Initialize-/Mint-Transaktion des Bonding-Curve-Programms, die praktisch immer
die erste Transaktion ist, die je gegen diese Adresse ausgefuehrt wurde.

EHRLICHKEITSREGEL (laut Auftrag, hart): Wird PAGES_CAP erreicht, BEVOR das
Ende der Signaturliste (kuerzere Page als SIGNATURES_PAGE_LIMIT oder leere
Page) erreicht ist, gilt das Erstellungsdatum als NICHT SICHER ERMITTELBAR -
es wird KEINE Schaetzung aus der aeltesten GEFUNDENEN Signatur geraten
(die waere ja per Definition juenger als die echte Erstellung und wuerde die
Alters-Klassifikation in Richtung "frischer" verzerren - dieselbe Art Bias wie
das urspruengliche Problem, nur an anderer Stelle).

WAS DIESES SKRIPT NICHT TUT (bewusst, siehe Auftrag):
- Keine neue Wallet-Abfrage. Die 23 Kauf-Eintraege aus historical_entries.json
  (Ergebnis von wallet_backtest_fetch.py + wallet_backtest_valuation.py)
  werden 1:1 wiederverwendet.
- Keine Neubewertung der aktuellen Coin-Performance. multiple_since_buy /
  return_pct_since_buy / dead_or_delisted aus historical_entries.json bleiben
  unveraendert - die Neuermittlung des Erstellungsdatums aendert nichts an der
  bereits ermittelten AKTUELLEN Bewertung, nur an der Alters-KLASSIFIKATION
  des Kaufs.
- Keine rueckwirkende Anwendung des RugCheck-Filters (src/paper_memecoin/
  filter.py). RugCheck liefert nur den AKTUELLEN Zustand eines Tokens (Mint-/
  Freeze-Authority, Holder-Konzentration, Liquiditaet, ...), nicht dessen
  historischen Zustand zum Kaufzeitpunkt vor Tagen/Wochen. Es ist schlicht
  nicht moeglich, nachtraeglich zu wissen, ob ein Token vor N Tagen zum
  Kaufzeitpunkt unseren Filter bestanden haette. Diese Analyse beantwortet
  daher weiterhin nur: "kauften diese Wallets tatsaechlich frische Coins, und
  was wurde daraus" - NICHT "was waere passiert, wenn zusaetzlich unser
  Rug-Filter bestanden haette". Das ist eine strukturelle Grenze jeder
  retrospektiven Analyse ohne historischen Snapshot-Datenspeicher, kein Fehler
  dieses Skripts, und wird im Bericht explizit wiederholt.

Nur Lesezugriff (RPC-Reads), keine Transaktionen. Schreibt NUR
artifacts/wallet_tracker/historical_age_refined_report.txt (und zum
Nachvollziehen ein Zwischen-Cache historical_mint_creation.json). Aendert
KEINE bestehende Datei.
"""
from __future__ import annotations

import datetime as dt
import json
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.wallet_tracker import solana_rpc_client  # noqa: E402

ARTIFACT_DIR = REPO_ROOT / "artifacts" / "wallet_tracker"
ENTRIES_PATH = ARTIFACT_DIR / "historical_entries.json"
MINT_CREATION_CACHE_PATH = ARTIFACT_DIR / "historical_mint_creation.json"
REPORT_PATH = ARTIFACT_DIR / "historical_age_refined_report.txt"

SIGNATURES_PAGE_LIMIT = 1000  # Solana-RPC-Maximum pro getSignaturesForAddress-Call
PAGES_CAP = 10  # Obergrenze Signaturen-Seiten (=RPC-Calls) pro Mint fuer das rueckwaertige Paging
MIN_ENTRIES_FOR_RELIABLE_BUCKET = 5  # siehe Auftrag TEIL 3


def _load_json(path: Path, default):
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default


def _save_json(path: Path, data) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


def find_mint_creation_time(mint: str) -> dict:
    """Paginiert rueckwaerts durch getSignaturesForAddress fuer die Mint-
    Adresse, bis das echte Ende der Historie erreicht ist (Page kuerzer als
    SIGNATURES_PAGE_LIMIT oder leer) oder PAGES_CAP erreicht wird. Gibt die
    aelteste GEFUNDENE Signatur zurueck, mit einem expliziten Flag, ob das
    wirklich das Ende der Historie war (=sicher) oder nicht (=PAGES_CAP
    erreicht, Erstellungsdatum NICHT sicher ermittelbar)."""
    before: str | None = None
    pages = 0
    oldest_sig: dict | None = None
    reached_end_of_history = False
    while pages < PAGES_CAP:
        params: dict = {"limit": SIGNATURES_PAGE_LIMIT}
        if before:
            params["before"] = before
        page = solana_rpc_client._rpc_call("getSignaturesForAddress", [mint, params])
        pages += 1
        if not isinstance(page, list) or not page:
            reached_end_of_history = True
            break
        oldest_sig = page[-1]
        before = page[-1].get("signature")
        if len(page) < SIGNATURES_PAGE_LIMIT:
            reached_end_of_history = True
            break

    creation_certain = reached_end_of_history and oldest_sig is not None and oldest_sig.get("blockTime")
    return {
        "mint": mint,
        "pages_fetched": pages,
        "reached_end_of_history": reached_end_of_history,
        "oldest_signature": oldest_sig.get("signature") if oldest_sig else None,
        "oldest_block_time": oldest_sig.get("blockTime") if oldest_sig else None,
        "creation_time_certain": bool(creation_certain),
        "creation_time_iso": (
            dt.datetime.fromtimestamp(oldest_sig["blockTime"], tz=dt.timezone.utc).isoformat()
            if creation_certain else None
        ),
    }


def _age_group(minutes: float | None) -> str:
    if minutes is None:
        return "Erstellungsdatum nicht ermittelbar"
    if minutes < 0:
        # Kauf-Zeitstempel liegt VOR der gefundenen "aeltesten" Mint-Signatur -
        # kann bei creation_time_certain=True eigentlich nicht vorkommen, ist
        # hier defensiv trotzdem abgefangen und wird nicht stillschweigend
        # einsortiert.
        return "Erstellungsdatum nicht ermittelbar"
    if minutes <= 15:
        return "<= 15 Min. nach Mint-Erstellung (\"echt frisch\")"
    if minutes <= 60:
        return "15 Min. - 1 Std. nach Mint-Erstellung"
    return "> 1 Std. nach Mint-Erstellung"


GROUP_ORDER = [
    "<= 15 Min. nach Mint-Erstellung (\"echt frisch\")",
    "15 Min. - 1 Std. nach Mint-Erstellung",
    "> 1 Std. nach Mint-Erstellung",
    "Erstellungsdatum nicht ermittelbar",
]


def _pct(n: int, total: int) -> str:
    return f"{100.0 * n / total:.1f}%" if total else "n/a"


def build_report(entries: list[dict], creation_cache: dict[str, dict]) -> tuple[str, dict]:
    refined = []
    for e in entries:
        mint = e["token_mint"]
        creation = creation_cache.get(mint, {})
        block_time = dt.datetime.fromisoformat(e["block_time"])
        minutes_since_creation = None
        if creation.get("creation_time_certain") and creation.get("oldest_block_time"):
            creation_dt = dt.datetime.fromtimestamp(creation["oldest_block_time"], tz=dt.timezone.utc)
            delta_min = (block_time - creation_dt).total_seconds() / 60.0
            if delta_min >= 0:
                minutes_since_creation = delta_min
        refined.append({**e, "minutes_since_mint_creation": minutes_since_creation,
                         "age_group": _age_group(minutes_since_creation),
                         "creation_time_certain": bool(creation.get("creation_time_certain"))})

    lines: list[str] = []
    lines.append("=" * 78)
    lines.append("KORRIGIERTE ALTERS-ANALYSE - ECHTES MINT-ERSTELLUNGSDATUM UEBER SOLANA-RPC")
    lines.append(f"Erzeugt: {dt.datetime.now(dt.timezone.utc).isoformat()}")
    lines.append("=" * 78)
    lines.append("")
    lines.append("KORRIGIERTE METHODIK GEGENUEBER DER VORGAENGER-ANALYSE:")
    lines.append("Die vorherige Alters-Klassifikation nutzte Dexscreener `pairCreatedAt`. Das")
    lines.append("ist (a) oft NULL fuer tote/delistete Coins - eine zirkulaere Verzerrung,")
    lines.append("denn 'kein Pool mehr' bedeutet 'keine Alters-Info mehr', unabhaengig vom")
    lines.append("echten Alter beim Kauf - und (b) selbst wenn vorhanden, das Datum der AMM-")
    lines.append("Migration, nicht der eigentlichen Mint-Erstellung auf der pump.fun-Bonding-")
    lines.append("Curve. Diese Analyse ersetzt das durch die AELTESTE bekannte On-Chain-")
    lines.append("Signatur der Mint-Adresse selbst (Solana Public RPC, getSignaturesForAddress,")
    lines.append("rueckwaerts paginiert bis zum nachweislichen Ende der Historie). Konnte das")
    lines.append("Ende der Historie innerhalb des Paging-Budgets NICHT erreicht werden, gilt")
    lines.append("das Erstellungsdatum als NICHT SICHER ERMITTELBAR - es wird KEINE Schaetzung")
    lines.append("aus der aeltesten GEFUNDENEN (aber moeglicherweise nicht aeltesten echten)")
    lines.append("Signatur geraten, um nicht denselben Bias-Fehler an anderer Stelle zu")
    lines.append("wiederholen.")
    lines.append("")
    lines.append(f"Basis: die bereits vorhandenen {len(entries)} Kauf-Eintraege aus")
    lines.append("historical_entries.json (keine neue Wallet-Abfrage). Die dort bereits")
    lines.append("ermittelte AKTUELLE Bewertung (multiple_since_buy, dead_or_delisted) bleibt")
    lines.append("unveraendert - nur die Alters-KLASSIFIKATION des Kaufs wird neu bestimmt.")
    lines.append("")

    lines.append("-" * 78)
    lines.append("ERGEBNIS PRO ALTERS-GRUPPE")
    lines.append("-" * 78)
    groups: dict[str, list[dict]] = {g: [] for g in GROUP_ORDER}
    for e in refined:
        groups[e["age_group"]].append(e)

    for g in GROUP_ORDER:
        members = groups[g]
        lines.append(f"\n{g}: n={len(members)}")
        if not members:
            lines.append("  (keine Eintraege in dieser Gruppe)")
            continue
        n_dead = sum(1 for e in members if e["dead_or_delisted"])
        returns = [e["return_pct_since_buy"] for e in members]
        multiples = [e["multiple_since_buy"] for e in members]
        n_15x = sum(1 for m in multiples if m >= 1.5)
        n_2x = sum(1 for m in multiples if m >= 2.0)
        lines.append(f"  tot/delisted: {n_dead}/{len(members)} ({_pct(n_dead, len(members))})")
        lines.append(f"  Median-Rendite seit Kauf: {statistics.median(returns):+.1%}")
        lines.append(f"  Anteil >= 1.5x seit Kauf: {n_15x}/{len(members)} ({_pct(n_15x, len(members))})")
        lines.append(f"  Anteil >= 2x seit Kauf:   {n_2x}/{len(members)} ({_pct(n_2x, len(members))})")
        for e in members:
            mins = e["minutes_since_mint_creation"]
            mins_str = f"{mins:.1f}min" if mins is not None else "n/a"
            lines.append(f"    - mint={e['token_mint'][:12]}... kauf={e['block_time']} "
                          f"alter_bei_kauf={mins_str} multiple={e['multiple_since_buy']:.3f} "
                          f"tot={e['dead_or_delisted']}")

    lines.append("")
    lines.append("-" * 78)
    lines.append("TEIL 3 - IST DIE '<=15MIN'-GRUPPE GROSS GENUG FUER EINE AUSSAGE?")
    lines.append("-" * 78)
    fresh = groups["<= 15 Min. nach Mint-Erstellung (\"echt frisch\")"]
    n_fresh = len(fresh)
    if n_fresh >= MIN_ENTRIES_FOR_RELIABLE_BUCKET:
        lines.append(f"n={n_fresh} >= Minimum ({MIN_ENTRIES_FOR_RELIABLE_BUCKET}) - eine vorsichtige Aussage")
        lines.append("ist grundsaetzlich moeglich, siehe Kennzahlen oben.")
    else:
        lines.append(f"n={n_fresh} < Minimum ({MIN_ENTRIES_FOR_RELIABLE_BUCKET}) fuer eine belastbare Aussage.")
        lines.append("ZU WENIG DATEN FUER EINE VERLAESSLICHE AUSSAGE ZU <=15MIN-KAEUFEN.")
        lines.append("Das ist ein valides, ehrliches Ergebnis und keine Ausrede: die")
        lines.append("Ausgangsstichprobe hatte insgesamt nur 23 Eintraege total, verteilt auf")
        lines.append("vier Alters-Gruppen - selbst wenn alle 23 in dieser einen Gruppe laegen,")
        lines.append("waere das noch eine kleine Stichprobe; mit einer Teilmenge davon ist eine")
        lines.append("Interpretation aus 1-4 Datenpunkten NICHT vertretbar (ein einzelner")
        lines.append("Ausreisser-Trade wuerde das Bild komplett kippen). Es wird hier bewusst")
        lines.append("KEINE Tendenzaussage ('sieht eher schlecht/gut aus') aus dieser Gruppe")
        lines.append("abgeleitet.")
    lines.append("")

    lines.append("=" * 78)
    lines.append("EHRLICHE EINORDNUNG - ANTWORT AUF DIE NUTZERFRAGE")
    lines.append("=" * 78)
    lines.append("Frage: 'Sind wirklich frisch gekaufte, gut gefilterte Coins auch tot?'")
    lines.append("")
    lines.append("Diese Analyse kann NUR den ersten Teil der Frage beantworten (frisch")
    lines.append("gekauft), und selbst das nur mit der oben dokumentierten kleinen")
    lines.append("Stichprobe. Der zweite Teil ('gut gefiltert') ist STRUKTURELL NICHT")
    lines.append("rueckwirkend pruefbar: der Rug-Filter (RugCheck: Mint-/Freeze-Authority,")
    lines.append("Holder-Konzentration, Liquiditaet, siehe src/paper_memecoin/filter.py)")
    lines.append("wurde NICHT rueckwirkend auf diese historischen Kaeufe angewendet, weil")
    lines.append("RugCheck nur den AKTUELLEN Zustand eines Tokens liefert, nicht dessen")
    lines.append("historischen Zustand zum Kaufzeitpunkt vor Tagen oder Wochen. Es ist nicht")
    lines.append("moeglich, nachtraeglich zu wissen, ob ein Token vor 40 Tagen zum")
    lines.append("Kaufzeitpunkt unseren Filter bestanden haette (Mint-/Freeze-Authority koennen")
    lines.append("seither widerrufen worden sein, Holder-Konzentration und Liquiditaet aendern")
    lines.append("sich staendig). Das ist eine strukturelle Grenze jeder retrospektiven Analyse")
    lines.append("ohne historischen Snapshot-Datenspeicher, kein Fehler dieses Skripts.")
    lines.append("")
    lines.append("Diese Analyse beantwortet also weiterhin nur die engere Frage: 'Kauften")
    lines.append("diese Wallets nachweislich frische Coins (jetzt ueber die Blockchain")
    lines.append("verifiziert, nicht ueber Dexscreener-Proxy), und was wurde in der Folge")
    lines.append("daraus?' - NICHT 'was waere passiert, wenn zusaetzlich unser Rug-Filter zum")
    lines.append("Kaufzeitpunkt bestanden haette'. Fuer die zweite Frage braeuchte es einen")
    lines.append("FORWARD-Test (RugCheck/Filter live auf neue Kaeufe anwenden und danach")
    lines.append("beobachten), keinen retrospektiven.")
    lines.append("")
    if n_fresh < MIN_ENTRIES_FOR_RELIABLE_BUCKET:
        lines.append(f"Zur reinen 'frisch gekauft'-Frage: die '<=15min'-Gruppe hat mit n={n_fresh}")
        lines.append("schlicht zu wenige Datenpunkte fuer eine verlaessliche Aussage - weder")
        lines.append("'die meisten davon sind tot' noch 'die meisten davon liefen gut' laesst sich")
        lines.append("aus dieser Stichprobengroesse seriös behaupten. Belastbarer sind allenfalls")
        lines.append("die groesseren Alters-Gruppen (>1h) oben, aber auch die sind mit n<=23 total")
        lines.append("statistisch klein.")
    else:
        n_dead_fresh = sum(1 for e in fresh if e["dead_or_delisted"])
        lines.append(f"Zur reinen 'frisch gekauft'-Frage: von {n_fresh} nachweislich innerhalb von")
        lines.append(f"15 Minuten nach echter Mint-Erstellung gekauften Coins sind {n_dead_fresh}")
        lines.append(f"({_pct(n_dead_fresh, n_fresh)}) heute tot/delisted.")
    lines.append("")
    n_uncertain = len(groups["Erstellungsdatum nicht ermittelbar"])
    lines.append(f"Zusaetzlicher Hinweis: bei {n_uncertain}/{len(entries)} Eintraegen liess sich das")
    lines.append("echte Erstellungsdatum innerhalb des Paging-Budgets (PAGES_CAP="
                  f"{PAGES_CAP} Seiten) nicht sicher bestimmen (Mint-Adresse hat mehr als")
    lines.append(f"{PAGES_CAP * SIGNATURES_PAGE_LIMIT} Signaturen, oder RPC-Fehler) - fuer diese")
    lines.append("wird bewusst keine Alters-Schaetzung geraten, siehe Methodik oben.")

    return "\n".join(lines), {"entries_refined": refined}


def main() -> None:
    data = _load_json(ENTRIES_PATH, None)
    if data is None:
        print("[age-refined] Keine historical_entries.json gefunden - erst wallet_backtest_valuation.py laufen lassen.")
        return
    entries = data.get("entries", [])
    mints = sorted({e["token_mint"] for e in entries})
    print(f"[age-refined] {len(entries)} Entries, {len(mints)} eindeutige Mints. Ermittle Mint-Erstellungsdaten ueber Solana RPC...", flush=True)

    creation_cache = _load_json(MINT_CREATION_CACHE_PATH, {})
    todo = [m for m in mints if m not in creation_cache]
    for i, mint in enumerate(todo, start=1):
        result = find_mint_creation_time(mint)
        creation_cache[mint] = result
        _save_json(MINT_CREATION_CACHE_PATH, creation_cache)
        status = "sicher" if result["creation_time_certain"] else "NICHT sicher (PAGES_CAP erreicht)"
        print(f"[age-refined] ({i}/{len(todo)}) {mint}: {result['pages_fetched']} Seiten, "
              f"Erstellung={result.get('creation_time_iso')} [{status}]", flush=True)
        time.sleep(0.05)  # zusaetzlich zum internen Throttling von solana_rpc_client

    report, _details = build_report(entries, creation_cache)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(report)
    print(f"\n[age-refined] gespeichert unter {REPORT_PATH}")


if __name__ == "__main__":
    main()
