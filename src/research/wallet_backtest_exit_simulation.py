"""TEIL 3 der historischen Wallet-Backtest-Analyse: KORREKTUR eines methodischen
Fehlers in wallet_backtest_valuation.py.

DER FEHLER (vom Nutzer erkannt): wallet_backtest_valuation.py vergleicht den
historischen Kaufpreis mit dem HEUTIGEN Preis (Wochen später) - das beantwortet
"was ist der Coin heute wert", nicht "war der TRADE gut". Der Nutzer tradet
die ersten ~15 Minuten nach Launch mit klaren Exit-Regeln (siehe
src/paper_memecoin/user_filter_strategy.py::should_exit_user_filter). Fast
jeder Memecoin geht irgendwann auf 0 - das sagt nichts über die Qualität des
TATSÄCHLICHEN Trades, wenn man rechtzeitig wieder rausgegangen wäre.

DIESES SKRIPT: rekonstruiert für jeden historischen Kauf (aus
historical_entries.json) den echten on-chain-Preisverlauf der MINT-ADRESSE
in den ersten 8 Stunden nach dem Kauf (max_hold=6h + 2h Puffer) über die
öffentliche Solana-RPC (src/wallet_tracker/solana_rpc_client.py, gleiche
Anbindung wie in den Vorgänger-Analysen) und simuliert darauf exakt die
Exit-Regeln aus user_filter_strategy.USER_FILTER_PARAMS /
should_exit_user_filter.

METHODIK PREISREKONSTRUKTION (analog zu fetcher.py/wallet_backtest_fetch.py,
aber generalisiert von "eine Wallet" auf "alle Trader dieses Mints"):
Für jede Transaktion, die den Mint referenziert, werden aus preTokenBalances/
postTokenBalances alle Owner mit einem Balance-Delta bei diesem Mint
extrahiert. Für jeden Owner wird über dessen Position in accountKeys der
korrespondierende SOL-Lamport-Delta (pre/postBalances) gesucht. Preis =
|sol_delta| / |token_delta|. Bei mehreren Deltas in einer Tx (typischerweise
Trader-Seite UND Pool/Bonding-Curve-Seite, die bei einem einfachen Swap
näherungsweise denselben impliziten Preis ergeben) wird der Median der
gefundenen Preise dieser Tx verwendet.

EHRLICHE EINSCHRÄNKUNGEN (siehe auch Abschlussbericht):
1. KEIN Wallet-Sell-Signal-Exit simulierbar: wir haben keine vollständige
   historische Sell-Historie ALLER 722 beobachteten Wallets für die
   jeweiligen Zeitfenster (wallet_trades.csv wird erst seit dem produktiven
   Fetcher-Start gefüllt, nicht rückwirkend für beliebige historische
   Fenster). => Simulation ist tendenziell OPTIMISTISCHER als ein echtes
   Live-System (das zusätzlich früher aussteigen könnte).
2. KEIN Mcap-Floor-Exit simulierbar: wir rekonstruieren nur den SOL-Preis
   pro Token aus Trade-Deltas, nicht die Total-Supply/Marketcap zu jedem
   Zeitpunkt (dafür bräuchte es einen historischen Supply-Feed, den wir
   keylos nicht haben). Der $10.000-Mcap-Floor aus USER_FILTER_PARAMS wird
   deshalb in dieser Simulation NICHT geprüft - ebenfalls tendenziell
   optimistisch (ein Exit, der in der Realität zusätzlich früher hätte
   greifen können, greift hier nicht).
3. Preis aus einzelnen Trade-Tx ist volatiler/rauschiger als ein echter
   OHLC-Chart - ein einzelner Ausreisser-Trade (z.B. winziger Betrag mit
   schlechtem Slippage) kann fälschlich einen Trailing-Stop auslösen oder
   einen Peak markieren, der so nie "sichtbar" gewesen wäre. Nicht
   überinterpretieren bei wenigen Preispunkten.
4. Aus RPC-Budgetgründen (3 req/s, siehe solana_rpc_client.py) werden pro
   Mint höchstens MAX_TX_FETCHED_PER_MINT Transaktionen tatsächlich per
   getTransaction geladen (gleichmässig über die im Fenster gelisteten,
   bereits auf MAX_SIGNATURES_LISTED gecappten Signaturen verteilt) - nicht
   jede einzelne Signatur. Das ist eine zusätzliche, über die Auftrags-
   Vorgabe (Cap bei ~1500 gelisteten Signaturen) hinausgehende Stichproben-
   Reduktion, um die Laufzeit praktikabel zu halten. Wird pro Mint
   dokumentiert.
5. Kleine Stichprobe (<=23 Mints, meist weniger mit genug Preispunkten).

Schreibt NUR neue Artefakte, ändert KEINE bestehende Datei:
  artifacts/wallet_tracker/historical_exit_simulation_raw.json   (Preisserien + Zwischenstand, inkrementell gecacht)
  artifacts/wallet_tracker/historical_exit_simulation_report.txt (Endbericht)
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
from src.paper_memecoin.user_filter_strategy import (  # noqa: E402
    USER_FILTER_PARAMS,
    should_exit_user_filter,
)

ARTIFACT_DIR = REPO_ROOT / "artifacts" / "wallet_tracker"
ENTRIES_PATH = ARTIFACT_DIR / "historical_entries.json"
MINT_CREATION_PATH = ARTIFACT_DIR / "historical_mint_creation.json"
RAW_OUT_PATH = ARTIFACT_DIR / "historical_exit_simulation_raw.json"
REPORT_OUT_PATH = ARTIFACT_DIR / "historical_exit_simulation_report.txt"

WINDOW_HOURS = 8.0  # max_hold=6h (Sicherheitsnetz) + 2h Puffer, siehe Auftrag
MAX_SIGNATURES_LISTED = 1500  # Cap der im Fenster GELISTETEN Signaturen (Auftrags-Vorgabe)
MAX_TX_FETCHED_PER_MINT = 220  # praktisches RPC-Budget: tatsächlich per getTransaction geladene Tx pro Mint (siehe Docstring Punkt 4)
PAGES_CAP_MINT = 40  # Sicherheitsnetz fürs rückwärtige before-Paging (40 * 1000 = 40000 gescannte Signaturen max.)
SIGNATURES_PAGE_LIMIT = 1000
VERIFIED_15MIN_MAX_MINUTES = 15.0


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


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


def _evenly_spaced_indices(n: int, k: int) -> list[int]:
    """Wie in wallet_backtest_fetch.py: k gleichmässig über [0, n) verteilte
    Indizes (chronologisch, älteste zuerst) - Punkte mit hoher Tx-Dichte
    (typischerweise kurz nach Launch) bekommen dadurch automatisch mehr
    Auflösung als ruhige Phasen später."""
    if n <= 0:
        return []
    if k >= n:
        return list(range(n))
    if k <= 1:
        return [0]
    step = (n - 1) / (k - 1)
    return sorted({round(i * step) for i in range(k)})


def determine_verified_15min_mints(entries: list[dict], mint_creation: dict) -> set[str]:
    """Mints, deren Kauf laut historical_mint_creation.json (creation_time_certain=True)
    <= VERIFIED_15MIN_MAX_MINUTES nach der Mint-Erstellung stattfand - die vom
    Auftrag bevorzugte Teilmenge."""
    verified = set()
    for e in entries:
        mint = e["token_mint"]
        info = mint_creation.get(mint)
        if not info or not info.get("creation_time_certain") or not info.get("creation_time_iso"):
            continue
        creation = dt.datetime.fromisoformat(info["creation_time_iso"])
        buy = dt.datetime.fromisoformat(e["block_time"])
        diff_minutes = (buy - creation).total_seconds() / 60.0
        if 0 <= diff_minutes <= VERIFIED_15MIN_MAX_MINUTES:
            verified.add(mint)
    return verified


def _fetch_signatures_in_window(mint: str, start: dt.datetime, end: dt.datetime) -> dict:
    """Rückwärtiges before-Paging über getSignaturesForAddress für die
    MINT-Adresse selbst (nicht eine einzelne Wallet), analog zu
    wallet_backtest_fetch.py::_fetch_signature_pages, aber generalisiert.
    Sammelt alle erfolgreichen (err=None) Signaturen mit blockTime in
    [start, end], bricht ab sobald eine Seite komplett vor `start` liegt
    oder PAGES_CAP_MINT erreicht ist."""
    in_window: list[dict] = []
    before: str | None = None
    pages = 0
    total_scanned = 0
    hit_pages_cap = False
    reached_before_start = False
    start_ts = start.timestamp()
    end_ts = end.timestamp()

    while pages < PAGES_CAP_MINT:
        params: dict = {"limit": SIGNATURES_PAGE_LIMIT}
        if before:
            params["before"] = before
        page = solana_rpc_client._rpc_call("getSignaturesForAddress", [mint, params])
        pages += 1
        if not isinstance(page, list) or not page:
            break
        total_scanned += len(page)
        for s in page:
            bt = s.get("blockTime")
            if bt is None or s.get("err") is not None:
                continue
            if start_ts <= bt <= end_ts:
                in_window.append(s)
        oldest_bt = page[-1].get("blockTime")
        before = page[-1].get("signature")
        if oldest_bt is not None and oldest_bt < start_ts:
            reached_before_start = True
            break
        if len(page) < SIGNATURES_PAGE_LIMIT:
            break
    else:
        hit_pages_cap = True

    in_window.sort(key=lambda s: s["blockTime"])
    listed_capped = len(in_window) > MAX_SIGNATURES_LISTED
    if listed_capped:
        in_window = in_window[:MAX_SIGNATURES_LISTED]

    return {
        "mint": mint,
        "signatures_scanned_total": total_scanned,
        "pages_fetched": pages,
        "hit_pages_cap": hit_pages_cap,
        "reached_before_window_start": reached_before_start,
        "in_window_signature_count": len(in_window),
        "listed_capped_at_1500": listed_capped,
        "signatures": in_window,
    }


def _account_keys(tx: dict) -> list[str]:
    keys = ((tx.get("transaction") or {}).get("message") or {}).get("accountKeys") or []
    return [k.get("pubkey") if isinstance(k, dict) else k for k in keys]


def extract_mint_price_points(tx: dict, target_mint: str) -> list[dict]:
    """Generalisierte Variante von fetcher.py::extract_wallet_trades: statt
    auf EINE Wallet zu filtern, werden ALLE Owner mit einem Balance-Delta bei
    target_mint in dieser Tx betrachtet (Trader- UND Pool-/Bonding-Curve-
    Seite - beide ergeben näherungsweise denselben impliziten Preis, siehe
    Moduldocstring)."""
    meta = tx.get("meta") or {}
    pre = {b["accountIndex"]: b for b in (meta.get("preTokenBalances") or [])}
    post = {b["accountIndex"]: b for b in (meta.get("postTokenBalances") or [])}
    account_keys = _account_keys(tx)
    pre_bal = meta.get("preBalances") or []
    post_bal = meta.get("postBalances") or []

    owner_deltas: dict[str, float] = {}
    for idx in set(pre) | set(post):
        pre_b, post_b = pre.get(idx), post.get(idx)
        mint = (post_b or pre_b or {}).get("mint")
        if mint != target_mint:
            continue
        owner = (post_b or pre_b or {}).get("owner")
        if not owner:
            continue
        pre_amt = float(((pre_b or {}).get("uiTokenAmount") or {}).get("uiAmount") or 0.0)
        post_amt = float(((post_b or {}).get("uiTokenAmount") or {}).get("uiAmount") or 0.0)
        owner_deltas[owner] = owner_deltas.get(owner, 0.0) + (post_amt - pre_amt)

    points = []
    for owner, delta in owner_deltas.items():
        if abs(delta) < 1e-9:
            continue
        try:
            oidx = account_keys.index(owner)
        except ValueError:
            continue
        if oidx >= len(pre_bal) or oidx >= len(post_bal):
            continue
        sol_delta = (post_bal[oidx] - pre_bal[oidx]) / 1_000_000_000.0
        if abs(sol_delta) < 1e-9:
            continue
        price = abs(sol_delta) / abs(delta)
        if price <= 0:
            continue
        points.append({
            "owner": owner, "action": "buy" if delta > 0 else "sell",
            "token_delta": delta, "sol_delta": sol_delta, "price_sol": price,
        })
    return points


def build_price_series_for_mint(mint: str, buy_dt: dt.datetime, log_prefix: str = "") -> dict:
    end_dt = buy_dt + dt.timedelta(hours=WINDOW_HOURS)
    window_end = min(end_dt, _now())
    listing = _fetch_signatures_in_window(mint, buy_dt, end_dt)
    sigs = listing["signatures"]

    idxs = _evenly_spaced_indices(len(sigs), MAX_TX_FETCHED_PER_MINT)
    selected = [sigs[i] for i in idxs]
    tx_fetch_capped = len(sigs) > len(selected)

    series: list[dict] = []
    tx_fetch_failures = 0
    for j, s in enumerate(selected, start=1):
        sig = s["signature"]
        tx = solana_rpc_client.get_transaction(sig)
        if tx is None:
            tx_fetch_failures += 1
            continue
        points = extract_mint_price_points(tx, mint)
        if not points:
            continue
        prices = [p["price_sol"] for p in points]
        block_time = dt.datetime.fromtimestamp(s["blockTime"], tz=dt.timezone.utc)
        series.append({
            "block_time": block_time.isoformat(),
            "price_sol": statistics.median(prices),
            "n_deltas_in_tx": len(points),
            "tx_signature": sig,
        })
        if j % 40 == 0:
            print(f"{log_prefix}  ... {j}/{len(selected)} Tx geladen, {len(series)} Preispunkte bisher", flush=True)

    series.sort(key=lambda p: p["block_time"])
    return {
        "mint": mint,
        "buy_time": buy_dt.isoformat(),
        "window_end_time": end_dt.isoformat(),
        "window_end_capped_by_now": end_dt > _now(),
        "signatures_scanned_total": listing["signatures_scanned_total"],
        "pages_fetched": listing["pages_fetched"],
        "hit_pages_cap": listing["hit_pages_cap"],
        "reached_before_window_start": listing["reached_before_window_start"],
        "in_window_signature_count_before_cap": listing["in_window_signature_count"],
        "listed_capped_at_1500": listing["listed_capped_at_1500"],
        "tx_fetched": len(selected),
        "tx_fetch_capped_from_listed": tx_fetch_capped,
        "tx_fetch_failures": tx_fetch_failures,
        "price_points_found": len(series),
        "price_series": series,
    }


PRICE_SANITY_BAND = 100.0  # siehe simulate_exit: Preispunkte ausserhalb [entry/BAND, entry*BAND] werden verworfen


def simulate_exit(entry_price_sol: float, buy_dt: dt.datetime, price_series: list[dict]) -> dict:
    """Wendet should_exit_user_filter() punktweise auf die rekonstruierte
    Preisserie an (chronologisch), token_mint=None (kein Wallet-Sell-Signal,
    siehe Einschränkung 1), current_mcap=None immer (siehe Einschränkung 2).
    peak_price wird als laufendes Maximum ALLER bisherigen current_price-Werte
    mitgeführt (should_exit_user_filter bezieht den jeweils aktuellen Preis
    selbst mit in effective_peak ein).

    SANITY-FILTER (empirisch entdeckt, siehe Moduldocstring Punkt 3): manche
    Tx sind Token-zu-Token-Multi-Hop-Swaps (Route über einen anderen Token,
    nicht direkt SOL<->Ziel-Mint) - dort entspricht der SOL-Lamport-Delta des
    Owners NICHT dem Zahlungsbetrag für diesen Mint, sondern z.B. nur einer
    Nebengebühr/Rent-Bewegung, was einen um Grössenordnungen falschen
    "Preis" erzeugt (beobachtet: ein Preispunkt 580x unter dem echten
    Kaufpreis in derselben Sekunde wie der Kauf selbst, der einen reinen
    Daten-Artefakt-Stop-Loss ausgelöst hätte). Preispunkte ausserhalb von
    [entry_price/PRICE_SANITY_BAND, entry_price*PRICE_SANITY_BAND] werden
    deshalb als Daten-Artefakt verworfen, NICHT als echte Kursbewegung
    gewertet - ein 100x-Band ist grosszügig genug, um echte Memecoin-Pumps
    (die durchaus 10-50x erreichen können) nicht fälschlich zu verwerfen."""
    filtered_count = 0
    if price_series:
        lo, hi = entry_price_sol / PRICE_SANITY_BAND, entry_price_sol * PRICE_SANITY_BAND
        clean_series = [p for p in price_series if lo <= p["price_sol"] <= hi]
        filtered_count = len(price_series) - len(clean_series)
        price_series = clean_series

    if not price_series:
        return {
            "exit_reason": "kein_preisdatenpunkt_bis_fensterende",
            "exit_time": None, "exit_price_sol": None, "simulated_return_pct": None,
            "n_price_points_used": 0, "n_points_filtered_as_outlier": filtered_count,
            "note": "Keine einzige on-chain-Transaktion des Mints im 8h-Fenster nach dem Kauf gefunden "
                    "(oder RPC-Fehler, oder alle gefundenen Punkte als Daten-Artefakt verworfen) - selbst ein "
                    "Datenpunkt: sehr wahrscheinlich sofortiger Totalverlust/toter Coin, aber NICHT als -100% "
                    "behauptet, da nicht direkt beobachtet.",
        }

    peak = entry_price_sol
    for pt in price_series:
        t = dt.datetime.fromisoformat(pt["block_time"])
        p = pt["price_sol"]
        exited, reason = should_exit_user_filter(
            entry_price=entry_price_sol, current_price=p,
            entry_mcap=None, current_mcap=None,
            entry_time=buy_dt, now=t, peak_price=peak, token_mint=None,
        )
        peak = max(peak, p)
        if exited:
            category = (
                "stop_loss" if "stop_loss" in reason else
                "trailing_stop" if "trailing_stop" in reason else
                "max_hold_time" if "max_hold_time" in reason else
                "mcap_floor" if "mcap_floor" in reason else "other"
            )
            return {
                "exit_reason": category, "exit_reason_detail": reason,
                "exit_time": t.isoformat(), "exit_price_sol": p,
                "simulated_return_pct": (p / entry_price_sol) - 1.0,
                "n_price_points_used": len(price_series), "n_points_filtered_as_outlier": filtered_count,
            }

    last = price_series[-1]
    last_t = dt.datetime.fromisoformat(last["block_time"])
    held_hours = (last_t - buy_dt).total_seconds() / 3600.0
    if held_hours >= WINDOW_HOURS - 0.05:
        # Preisdaten decken das volle Fenster ab, aber weder SL/Trail noch
        # (aus Datengründen nicht simulierbarer) Mcap-Floor/Wallet-Sell haben
        # ausgelöst UND aus irgendeinem Grund auch nicht max_hold (sollte bei
        # 8h Fenster vs. 6h Limit eigentlich nicht vorkommen) - Sicherheitsnetz:
        p = last["price_sol"]
        return {
            "exit_reason": "max_hold_time_fallback", "exit_reason_detail": f"Fenster endet bei {held_hours:.1f}h, kein Exit-Signal ausgelöst - letzter bekannter Preis verwendet.",
            "exit_time": last_t.isoformat(), "exit_price_sol": p,
            "simulated_return_pct": (p / entry_price_sol) - 1.0,
            "n_price_points_used": len(price_series), "n_points_filtered_as_outlier": filtered_count,
        }

    return {
        "exit_reason": "kein_preisdatenpunkt_bis_fensterende",
        "exit_time": last_t.isoformat(), "exit_price_sol": last["price_sol"],
        "simulated_return_pct": (last["price_sol"] / entry_price_sol) - 1.0,
        "n_price_points_used": len(price_series), "n_points_filtered_as_outlier": filtered_count,
        "note": f"Preisdaten brechen bei {held_hours:.1f}h nach Kauf ab (letzte gefundene Tx des Mints im "
                "Fenster) - keine weiteren Transaktionen mehr gefunden, bevor das 8h-Fenster oder max_hold "
                "(6h) erreicht wurde. Selbst ein Datenpunkt: wahrscheinlich Totalverlust/toter Coin "
                "(keine Liquidität/kein Interesse mehr), aber nicht als exakter Exit-Preis zu interpretieren "
                "- letzter beobachteter Preis wird informativ als 'letzter bekannter Stand' berichtet.",
    }


def main() -> None:
    entries_data = _load_json(ENTRIES_PATH, None)
    mint_creation = _load_json(MINT_CREATION_PATH, {})
    if entries_data is None:
        print("[exit_sim] historical_entries.json nicht gefunden.")
        return
    entries = entries_data["entries"]
    verified = determine_verified_15min_mints(entries, mint_creation)
    print(f"[exit_sim] {len(entries)} Entries total, {len(verified)} als <=15min-Kauf verifiziert "
          f"(werden zuerst prozessiert).", flush=True)

    # Reihenfolge: verifizierte 11 zuerst, Rest danach (Best-Effort je nach Budget).
    ordered = sorted(entries, key=lambda e: (e["token_mint"] not in verified, e["block_time"]))

    raw = _load_json(RAW_OUT_PATH, {"generated_at": None, "mints": {}})
    t0 = time.monotonic()

    for i, e in enumerate(ordered, start=1):
        mint = e["token_mint"]
        if mint in raw["mints"]:
            print(f"[exit_sim] ({i}/{len(ordered)}) {mint} - bereits gecacht, überspringe Fetch.", flush=True)
            continue
        buy_dt = dt.datetime.fromisoformat(e["block_time"])
        tag = "VERIFIZIERT<=15min" if mint in verified else "unverifiziert"
        elapsed = time.monotonic() - t0
        print(f"[exit_sim] ({i}/{len(ordered)}) {mint} [{tag}] Kauf={e['block_time']} [t={elapsed:.0f}s]", flush=True)
        series_info = build_price_series_for_mint(mint, buy_dt, log_prefix=f"[exit_sim]   {mint[:8]}")
        sim = simulate_exit(e["entry_price_sol"], buy_dt, series_info["price_series"])
        raw["mints"][mint] = {
            "verified_15min_buy": mint in verified,
            "wallet_label": e.get("wallet_label"),
            "block_time": e["block_time"],
            "entry_price_sol": e["entry_price_sol"],
            "old_method_return_pct_vs_today": e["return_pct_since_buy"],
            "old_method_dead_or_delisted": e["dead_or_delisted"],
            "series_info": {k: v for k, v in series_info.items() if k != "price_series"},
            "price_series": series_info["price_series"],
            "simulation": sim,
        }
        raw["generated_at"] = _now().isoformat()
        _save_json(RAW_OUT_PATH, raw)
        print(f"[exit_sim]   -> {series_info['price_points_found']} Preispunkte, "
              f"Exit-Grund={sim['exit_reason']}, sim.Rendite="
              f"{('%.1f%%' % (sim['simulated_return_pct']*100)) if sim.get('simulated_return_pct') is not None else 'n/a'}",
              flush=True)

    write_report(raw, entries)
    print(f"[exit_sim] fertig. Bericht: {REPORT_OUT_PATH}", flush=True)


def write_report(raw: dict, entries: list[dict]) -> None:
    rows = []
    for e in entries:
        mint = e["token_mint"]
        m = raw["mints"].get(mint)
        if not m:
            continue
        sim = m["simulation"]
        rows.append({
            "mint": mint, "wallet_label": m["wallet_label"], "block_time": m["block_time"],
            "verified_15min_buy": m["verified_15min_buy"],
            "old_return_pct": m["old_method_return_pct_vs_today"],
            "old_dead": m["old_method_dead_or_delisted"],
            "n_price_points": m["series_info"]["price_points_found"],
            "sim_exit_reason": sim["exit_reason"],
            "sim_return_pct": sim.get("simulated_return_pct"),
        })

    usable = [r for r in rows if r["n_price_points"] > 0 and r["sim_return_pct"] is not None]
    verified_rows = [r for r in rows if r["verified_15min_buy"]]
    verified_usable = [r for r in verified_rows if r["n_price_points"] > 0 and r["sim_return_pct"] is not None]

    def stats(vals):
        if not vals:
            return None, None
        return statistics.median(vals), statistics.mean(vals)

    sim_all_med, sim_all_mean = stats([r["sim_return_pct"] for r in usable])
    old_all_med, old_all_mean = stats([r["old_return_pct"] for r in usable])
    sim_ver_med, sim_ver_mean = stats([r["sim_return_pct"] for r in verified_usable])
    old_ver_med, old_ver_mean = stats([r["old_return_pct"] for r in verified_usable])

    profitable_sim = sum(1 for r in usable if r["sim_return_pct"] > 0)
    profitable_old = sum(1 for r in usable if r["old_return_pct"] > 0)
    dead_old = sum(1 for r in usable if r["old_dead"])

    reason_counts: dict[str, int] = {}
    for r in rows:
        reason_counts[r["sim_exit_reason"]] = reason_counts.get(r["sim_exit_reason"], 0) + 1

    lines = []
    lines.append("=" * 78)
    lines.append("EXIT-SIMULATION AUF REKONSTRUIERTEN PREISVERLÄUFEN (user_filter-Strategie)")
    lines.append(f"Generiert: {raw.get('generated_at')}")
    lines.append("=" * 78)
    lines.append("")
    lines.append("KORRIGIERTE FRAGESTELLUNG: nicht 'was ist der Coin heute wert', sondern")
    lines.append("'was wäre in den ersten Stunden nach dem historischen Kauf passiert, wenn")
    lines.append("die tatsächlichen Exit-Regeln (Stop-Loss -45%, Trailing-Stop -30% vom Hoch")
    lines.append("ab +30%, Max-Hold 6h) angewendet worden wären'.")
    lines.append("")
    lines.append(f"Parameter (aus USER_FILTER_PARAMS): stop_loss={USER_FILTER_PARAMS['stop_loss_pct_default']:.0%}, "
                  f"trailing_activation={USER_FILTER_PARAMS['peak_activation_multiple']:.2f}x, "
                  f"trailing_stop={USER_FILTER_PARAMS['trailing_stop_pct']:.0%}, "
                  f"max_hold={USER_FILTER_PARAMS['max_hold_hours']:.0f}h.")
    lines.append(f"Mcap-Floor (${USER_FILTER_PARAMS['mcap_exit_floor_usd']:,.0f}) NICHT simuliert (kein "
                  "historischer Supply-Feed). Wallet-Sell-Signal NICHT simuliert (keine vollständige "
                  "historische Sell-Historie aller 722 Wallets). Beides macht diese Simulation tendenziell "
                  "OPTIMISTISCHER als ein echtes Live-System.")
    lines.append("")
    lines.append(f"STICHPROBE: {len(entries)} historische Käufe insgesamt, {len(verified_rows)} davon als "
                  f"<=15min-Kauf verifiziert. {len(rows)} verarbeitet (Preisrekonstruktion versucht), "
                  f"{len(usable)} mit >=1 rekonstruiertem Preispunkt UND simulierter Rendite "
                  f"({len(verified_usable)} davon aus der verifizierten <=15min-Teilmenge).")
    lines.append("")
    lines.append("-" * 78)
    lines.append("VERGLEICH: alte Methode (Kaufpreis vs. HEUTIGER Preis) vs. neue Methode")
    lines.append("(simulierter Exit gemäss Strategie-Regeln in den ersten Stunden)")
    lines.append("-" * 78)
    lines.append(f"  Alle mit Preisdaten (n={len(usable)}):")
    lines.append(f"    Alte Methode  - Median: {_pct(old_all_med)}  Mittelwert: {_pct(old_all_mean)}")
    lines.append(f"    Neue Methode  - Median: {_pct(sim_all_med)}  Mittelwert: {_pct(sim_all_mean)}")
    lines.append(f"    profitabel (>0%): alt={profitable_old}/{len(usable)}  neu={profitable_sim}/{len(usable)}")
    lines.append(f"    'tot' laut alter Methode (kein Dexscreener-Pool mehr): {dead_old}/{len(usable)}")
    lines.append("")
    lines.append(f"  Nur verifizierte <=15min-Käufe (n={len(verified_usable)}):")
    lines.append(f"    Alte Methode  - Median: {_pct(old_ver_med)}  Mittelwert: {_pct(old_ver_mean)}")
    lines.append(f"    Neue Methode  - Median: {_pct(sim_ver_med)}  Mittelwert: {_pct(sim_ver_mean)}")
    lines.append("")
    lines.append("-" * 78)
    lines.append("VERTEILUNG DER SIMULIERTEN EXIT-GRÜNDE (alle prozessierten Mints)")
    lines.append("-" * 78)
    for reason, count in sorted(reason_counts.items(), key=lambda kv: -kv[1]):
        lines.append(f"    {reason}: {count}")
    lines.append("")
    lines.append("-" * 78)
    lines.append("PRO-MINT-DETAIL")
    lines.append("-" * 78)
    lines.append(f"{'mint':<12} {'wallet':<12} {'<=15min':<8} {'#preis':<7} {'exit_grund':<28} {'sim_%':>9} {'alt_%':>9}")
    for r in sorted(rows, key=lambda r: r["block_time"]):
        sim_pct = f"{r['sim_return_pct']*100:+.1f}%" if r["sim_return_pct"] is not None else "n/a"
        lines.append(
            f"{r['mint'][:10]:<12} {str(r['wallet_label'])[:10]:<12} {('ja' if r['verified_15min_buy'] else 'nein'):<8} "
            f"{r['n_price_points']:<7} {r['sim_exit_reason']:<28} {sim_pct:>9} {r['old_return_pct']*100:+.1f}%"
        )
    lines.append("")
    lines.append("-" * 78)
    lines.append("EHRLICHE EINSCHRÄNKUNGEN")
    lines.append("-" * 78)
    lines.append("1. Kein Wallet-Sell-Signal-Exit simulierbar - Simulation ist optimistischer als live.")
    lines.append("2. Kein Mcap-Floor-Exit simulierbar (kein historischer Supply-Feed) - ebenfalls optimistisch.")
    lines.append("3. Preis aus einzelnen Trade-Tx ist volatiler als ein OHLC-Chart - einzelne Ausreisser")
    lines.append("   können Trailing-Stop/Peak verfälschen, besonders bei wenigen Preispunkten.")
    lines.append("4. Aus RPC-Budgetgründen max. {} tatsächlich geladene Tx pro Mint (von bis zu 1500".format(MAX_TX_FETCHED_PER_MINT))
    lines.append("   gelisteten Signaturen im Fenster), gleichmässig verteilt - keine Tick-für-Tick-Auflösung.")
    lines.append("5. Kleine Stichprobe (<=23 Mints, meist weniger mit genug Preispunkten).")
    lines.append("")

    with open(REPORT_OUT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def _pct(v):
    return f"{v*100:+.1f}%" if v is not None else "n/a"


if __name__ == "__main__":
    main()
