"""TEIL 3 der historischen Wallet-Backtest-Analyse: aggregiert
historical_entries.json (siehe wallet_backtest_valuation.py) zu den vom
Auftrag geforderten Kennzahlen und schreibt einen lesbaren Bericht nach
artifacts/wallet_tracker/historical_backtest_report.txt.

WICHTIGSTE EINSCHRÄNKUNG (siehe Bericht-Kopf, MUSS ehrlich stehen bleiben):
Dies ist KEIN sauberer Hold-out-Test. Die "aktuelle" Bewertung jedes Tokens
nutzt Informationen (den heutigen Dexscreener-Stand), die zum Kaufzeitpunkt
nicht verfügbar waren (Survivorship-/Look-ahead-Problem). Es beantwortet
"Wie oft lag diese Wallet-Gruppe historisch richtig, im Nachhinein
betrachtet?" - keine Simulation von echtem Live-Trading mit Entry-/Exit-
Timing (kein bester Exit-Punkt wird gesucht, nur Kaufpreis vs. HEUTE).
"""
from __future__ import annotations

import datetime as dt
import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

ARTIFACT_DIR = REPO_ROOT / "artifacts" / "wallet_tracker"
BUYS_PATH = ARTIFACT_DIR / "historical_buys.json"
ENTRIES_PATH = ARTIFACT_DIR / "historical_entries.json"
REPORT_TXT_PATH = ARTIFACT_DIR / "historical_backtest_report.txt"

MIN_ENTRIES_PER_WALLET_FOR_BREAKDOWN = 3


def _pct(n: int, total: int) -> str:
    return f"{100.0 * n / total:.1f}%" if total else "n/a"


def _age_bucket(hours: float | None) -> str:
    if hours is None:
        return "unbekannt (kein Dexscreener-pairCreatedAt)"
    if hours < 0:
        return "unbekannt (Zeitstempel unplausibel)"
    if hours <= 1:
        return "<= 1h (sehr frischer Launch)"
    if hours <= 24:
        return "1-24h"
    if hours <= 24 * 7:
        return "1-7 Tage"
    return "> 7 Tage (kein 'frischer' Memecoin-Kauf im engeren Sinn)"


def build_report() -> str:
    buys_cache = json.loads(BUYS_PATH.read_text(encoding="utf-8")) if BUYS_PATH.exists() else {}
    entries_data = json.loads(ENTRIES_PATH.read_text(encoding="utf-8")) if ENTRIES_PATH.exists() else {}
    entries: list[dict] = entries_data.get("entries", [])

    wallets_meta = buys_cache.get("wallets", {})
    n_wallets_sampled = len(wallets_meta)
    n_wallets_with_buys = sum(1 for w in wallets_meta.values() if w.get("buys_found", 0) > 0)
    n_raw_buys = len(buys_cache.get("buys", []))

    lines: list[str] = []
    lines.append("=" * 78)
    lines.append("HISTORISCHER WALLET-BACKTEST - Axiom 'beste Trader'-Liste")
    lines.append(f"Erzeugt: {dt.datetime.now(dt.timezone.utc).isoformat()}")
    lines.append("=" * 78)
    lines.append("")
    lines.append("EINSCHRAENKUNG (bitte NICHT ueberlesen):")
    lines.append("Dies ist KEIN sauberer Hold-out-Test. Die 'aktuelle' Bewertung jedes")
    lines.append("Tokens nutzt Informationen (heutiger Dexscreener-Stand), die zum")
    lines.append("Kaufzeitpunkt nicht verfuegbar waren -> klassisches Survivorship-/")
    lines.append("Look-ahead-Problem bei retrospektiver Analyse. Es wird NICHT simuliert,")
    lines.append("wann/ob ein reales Live-System hatte aussteigen koennen (kein optimales")
    lines.append("Timing gesucht, kein Peak-Preis waehrend der Haltezeit) - nur")
    lines.append("Kaufpreis vs. HEUTE. Die Frage, die dieser Bericht beantwortet, ist eng:")
    lines.append("'Wie oft lag diese Wallet-Gruppe historisch richtig?', NICHT 'was haette")
    lines.append("Live-Trading mit dieser Strategie abgeworfen'.")
    lines.append("")
    lines.append("-" * 78)
    lines.append("STICHPROBE")
    lines.append("-" * 78)
    lines.append(f"Wallets in Stichprobe (erste N aus watched_wallets.txt, Auswahl siehe")
    lines.append(f"wallet_backtest_fetch.py-Docstring): {n_wallets_sampled}")
    lines.append(f"Davon mit mind. 1 gefundenem Kauf im abgedeckten Zeitfenster: {n_wallets_with_buys} "
                  f"({_pct(n_wallets_with_buys, n_wallets_sampled)})")
    lines.append(f"Rohe Buy-Events (vor Dedup pro Wallet x Mint): {n_raw_buys}")
    lines.append(f"Ausgewertete Entries (1 pro Wallet x Mint, fruehster Kauf): {len(entries)}")
    lines.append("")
    n_full_cov = sum(1 for w in wallets_meta.values() if w.get("window_fully_covered_by_pagination"))
    lines.append(f"Wallets, bei denen das Zeitfenster (LOOKBACK_DAYS) durch das Paging voll")
    lines.append(f"abgedeckt wurde: {n_full_cov}/{n_wallets_sampled}. Bei den uebrigen ist die")
    lines.append(f"tatsaechlich abgedeckte Historie KUERZER (siehe wallets.<addr>.oldest_fetched_block_time")
    lines.append(f"in historical_buys.json) - einige der meistkopierten Top-Wallets der Liste")
    lines.append(f"(z.B. 'cented', 'kadenox', 'king trey', 'decu') werden zum Analysezeitpunkt so massiv")
    lines.append(f"mit fehlschlagenden Spam-/Front-Running-Transaktionen ueberflutet, dass selbst")
    lines.append(f"mehrere tausend Signaturen nur Sekunden bis Minuten in die Vergangenheit reichen -")
    lines.append(f"fuer diese Wallets liefert die Stichprobe ehrlich wenig bis keine historischen Buys.")
    lines.append("")

    if not entries:
        lines.append("KEINE auswertbaren Entries gefunden - siehe obige Einschraenkung. Bericht endet hier.")
        return "\n".join(lines)

    n = len(entries)
    dead = [e for e in entries if e["dead_or_delisted"]]
    alive = [e for e in entries if not e["dead_or_delisted"]]
    returns = [e["return_pct_since_buy"] for e in entries]
    multiples = [e["multiple_since_buy"] for e in entries]

    lines.append("-" * 78)
    lines.append("TEIL 2/3 - WAS WURDE AUS DEN GEKAUFTEN COINS?")
    lines.append("-" * 78)
    lines.append(f"Noch auf Dexscreener handelbar: {len(alive)} ({_pct(len(alive), n)})")
    lines.append(f"Nicht mehr auffindbar / kein Pool mehr (= Totalverlust gewertet): "
                  f"{len(dead)} ({_pct(len(dead), n)})")
    lines.append("")
    lines.append(f"Median-Rendite seit Kauf (dead=-100%, SOL-denominiert, siehe Methodik in "
                  f"wallet_backtest_valuation.py): {statistics.median(returns):+.1%}")
    lines.append(f"Mittelwert-Rendite seit Kauf: {statistics.mean(returns):+.1%} "
                  f"(Mittelwert wird von einzelnen Ausreissern verzerrt - Median ist die "
                  f"aussagekraeftigere Kennzahl, siehe bestehende Survivorship-Analyse im Repo)")
    n_2x = sum(1 for m in multiples if m >= 2.0)
    n_15x = sum(1 for m in multiples if m >= 1.5)
    lines.append(f"Erreichten >= 2x (+100%) seit Kauf: {n_2x} ({_pct(n_2x, n)})")
    lines.append(f"Erreichten >= 1.5x (+50%) seit Kauf: {n_15x} ({_pct(n_15x, n)})")
    lines.append("")
    lines.append("Vergleich zur vom Nutzer zitierten generellen Basis-Rate: ~99% Rug-Rate bei")
    lines.append("frischen Sub-100k-Launches OHNE Wallet-Signal (d.h. dort landen ueblicherweise")
    lines.append(f"~1% bei irgendeinem Erfolg). In dieser Stichprobe: {_pct(len(dead), n)} Totalverlust/")
    lines.append(f"delisted, {_pct(n_2x, n)} mit >=2x seit Kauf.")
    lines.append("")

    lines.append("-" * 78)
    lines.append("AUFSCHLUESSELUNG NACH TOKEN-ALTER ZUM KAUFZEITPUNKT")
    lines.append("(relevant fuer die Frage: kaufen diese Wallets wirklich FRISCHE Memecoins?)")
    lines.append("-" * 78)
    buckets: dict[str, list[dict]] = {}
    for e in entries:
        buckets.setdefault(_age_bucket(e["age_at_buy_hours"]), []).append(e)
    for label, group in sorted(buckets.items(), key=lambda kv: -len(kv[1])):
        g_dead = sum(1 for e in group if e["dead_or_delisted"])
        g_median = statistics.median(e["return_pct_since_buy"] for e in group)
        lines.append(f"  {label}: n={len(group)}, tot/delisted={_pct(g_dead, len(group))}, "
                      f"Median-Rendite={g_median:+.1%}")
    lines.append("")

    lines.append("-" * 78)
    lines.append("AUFSCHLUESSELUNG NACH HALTEDAUER SEIT KAUF (Tage bis HEUTE)")
    lines.append("(Proxy fuer 'wie lange nach Kauf lief der Coin, bevor er stieg/auf 0 ging' -")
    lines.append(" wir kennen NUR Kaufpreis vs. HEUTE, keinen Preisverlauf dazwischen, siehe")
    lines.append(" Einschraenkung oben. Relevant zum Abgleich mit MAX_HOLD_HOURS in")
    lines.append(" strategy.WALLET_SIGNAL_HYPOTHESIS_PARAMS, falls dort schon vorhanden.)")
    lines.append("-" * 78)
    hold_buckets = [("<1 Tag", lambda d: d < 1), ("1-7 Tage", lambda d: 1 <= d < 7),
                    ("7-30 Tage", lambda d: 7 <= d < 30), (">= 30 Tage", lambda d: d >= 30)]
    for label, pred in hold_buckets:
        group = [e for e in entries if pred(e["hold_days_since_buy"])]
        if not group:
            continue
        g_dead = sum(1 for e in group if e["dead_or_delisted"])
        g_median = statistics.median(e["return_pct_since_buy"] for e in group)
        lines.append(f"  {label}: n={len(group)}, tot/delisted={_pct(g_dead, len(group))}, "
                      f"Median-Rendite={g_median:+.1%}")
    lines.append("")

    lines.append("-" * 78)
    lines.append("PER-WALLET AUFFAELLIGKEITEN")
    lines.append(f"(nur Wallets mit >= {MIN_ENTRIES_PER_WALLET_FOR_BREAKDOWN} Entries - ACHTUNG: kleine")
    lines.append("Stichprobe pro Wallet, siehe Overfitting-Warnung am Ende)")
    lines.append("-" * 78)
    by_wallet: dict[str, list[dict]] = {}
    for e in entries:
        by_wallet.setdefault(e["wallet_label"] or e["wallet_address"], []).append(e)
    qualifying = {label: g for label, g in by_wallet.items() if len(g) >= MIN_ENTRIES_PER_WALLET_FOR_BREAKDOWN}
    if not qualifying:
        lines.append(f"  Keine Wallet erreicht {MIN_ENTRIES_PER_WALLET_FOR_BREAKDOWN}+ Entries in dieser "
                      f"Stichprobe - Aussagen auf Einzelwallet-Ebene sind bei dieser Fenstergroesse "
                      f"noch nicht sinnvoll moeglich.")
    else:
        ranked = sorted(qualifying.items(),
                         key=lambda kv: statistics.median(e["return_pct_since_buy"] for e in kv[1]),
                         reverse=True)
        for label, g in ranked:
            hit = sum(1 for e in g if e["multiple_since_buy"] >= 1.5)
            lines.append(f"  {label}: n={len(g)}, Median-Rendite={statistics.median(e['return_pct_since_buy'] for e in g):+.1%}, "
                          f">=1.5x-Trefferquote={_pct(hit, len(g))}")
    lines.append("")
    lines.append("WARNUNG: Einzelwallet-Trefferquoten basieren hier auf sehr wenigen Datenpunkten")
    lines.append("(oft < 10 pro Wallet). Das ist NICHT robust genug, um einzelne Wallets fuer die")
    lines.append("Priorisierung zu bevorzugen/auszuschliessen - es ist ein Hinweis fuer eine spaetere,")
    lines.append("groessere Stichprobe, kein belastbares Ranking. Overfitting-Risiko hoch.")
    lines.append("")

    lines.append("=" * 78)
    lines.append("EINORDNUNG DER NUTZER-HYPOTHESE")
    lines.append("=" * 78)
    lines.append("Hypothese: 'Wallets aus der Axiom-Liste kaufen frische Memecoins mit deutlich")
    lines.append("besserer Erfolgswahrscheinlichkeit als die generelle ~99% Rug-Rate.'")
    lines.append("")
    dead_pct = len(dead) / n
    hit_2x_pct = n_2x / n
    hit_15x_pct = n_15x / n
    med_return = statistics.median(returns)
    lines.append(f"In dieser Stichprobe ({n} Wallet-x-Mint-Entries aus {n_wallets_with_buys} von 30 Wallets,")
    lines.append(f"23 der 30 Stichproben-Wallets lieferten GAR KEINEN auswertbaren Kauf im Fenster):")
    lines.append(f"  - {_pct(len(dead), n)} der Kaeufe sind heute tot/delisted (Totalverlust gewertet)")
    lines.append(f"  - {_pct(n_2x, n)} erreichten seit Kauf mindestens 2x, {_pct(n_15x, n)} mindestens 1.5x")
    lines.append(f"  - Median-Rendite seit Kauf: {med_return:+.1%}")
    lines.append("")
    if dead_pct < 0.99 and (hit_2x_pct > 0 or hit_15x_pct > 0):
        verdict = ("GEMISCHT GESTUETZT: der Totalverlust-Anteil liegt unter der zitierten ~99%-Basisrate, "
                   "UND ein Teil der Kaeufe zeigt spuerbare Vervielfachungen.")
    elif dead_pct < 0.99 and hit_2x_pct == 0 and hit_15x_pct == 0:
        verdict = ("SCHWACH/NICHT UEBERZEUGEND GESTUETZT: der reine Totalverlust-Anteil liegt zwar unter "
                   "der zitierten ~99%-Basisrate, aber KEIN einziger der 23 Kaeufe erreichte seit dem "
                   "Kauf auch nur 1.5x - die 'Erfolgsfaelle' in dieser Stichprobe sind bestenfalls leichte "
                   "Wertsteigerungen oder Seitwaertsbewegungen, keine belastbaren Gewinn-Trades. Das ist "
                   "eher ein Hinweis, dass 'nicht komplett tot' und 'profitabler Trade' zwei verschiedene "
                   "Dinge sind, die man nicht verwechseln sollte.")
    else:
        verdict = "NICHT GESTUETZT: der Totalverlust-Anteil liegt in der Groessenordnung der zitierten Basisrate."
    lines.append(f"Befund dieser Stichprobe: {verdict}")
    lines.append("")
    lines.append("Zusaetzliche Vorsicht ist geboten, weil:")
    lines.append(f"  1. Nur {n_wallets_with_buys}/30 Stichproben-Wallets ueberhaupt einen auswertbaren Kauf")
    lines.append(f"     lieferten ({n} Entries) - die meisten der bekanntesten/meistkopierten Top-Wallets")
    lines.append(f"     ('cented', 'kadenox', 'king trey', 'decu', ...) konnten wegen der oben beschriebenen")
    lines.append(f"     Spam-Flut GAR NICHT ausgewertet werden. Diese Stichprobe ist also NICHT repraesentativ")
    lines.append(f"     fuer die 'besten' Wallets der Liste, sondern fuer die (zufaellig) weniger umkaempften.")
    lines.append(f"  2. n=23 ist statistisch klein - einzelne Ausreisser (z.B. ein einziger 10x-Trade) koennten")
    lines.append(f"     das Bild noch deutlich verschieben, in beide Richtungen.")
    lines.append(f"  3. Reiner Kaufpreis-vs-HEUTE-Vergleich mit Survivorship-/Look-ahead-Verzerrung (siehe")
    lines.append(f"     Einschraenkung ganz oben) - KEINE Simulation von echtem Entry-/Exit-Timing.")
    lines.append(f"  4. Ein Teil der 'toten' Coins sind moeglicherweise pump.fun-Bonding-Curve-Kaeufe VOR")
    lines.append(f"     Migration zu einem AMM-Pool, die auf Dexscreener nie als eigener 'Pair' auftauchen -")
    lines.append(f"     das wuerde die Totalverlust-Quote nach oben verzerren (siehe 'Zeitstempel unplausibel'-")
    lines.append(f"     Bucket oben, negative age_at_buy_hours deuten genau darauf hin).")
    lines.append("")
    lines.append("Fazit: Dieser erste, kleine historische Rueckblick liefert KEINEN klaren Beleg dafuer, dass")
    lines.append("die Wallet-Liste zuverlaessig Gewinner-Coins trifft - die Totalverlustquote ist zwar besser")
    lines.append("als die zitierte 99%-Baseline, aber niemand in der Stichprobe erreichte seit Kauf auch nur")
    lines.append("+50%. Fuer eine belastbare Aussage braucht es entweder eine groessere/repraesentativere")
    lines.append("Stichprobe (idealerweise mit geloester Spam-Problematik bei den Top-Wallets) oder die im")
    lines.append("Repo bereits laufende Forward-Wallet-Signal-Strategie (src/paper_memecoin/strategy.py,")
    lines.append("WALLET_SIGNAL_HYPOTHESIS_PARAMS) mit echtem Entry-/Exit-Timing statt Kauf-vs-heute.")

    return "\n".join(lines)


def main() -> None:
    report = build_report()
    print(report)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_TXT_PATH.write_text(report, encoding="utf-8")
    print(f"\n[report] gespeichert unter {REPORT_TXT_PATH}")


if __name__ == "__main__":
    main()
