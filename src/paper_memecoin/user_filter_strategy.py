"""VIERTE, SEPARATE Strategie-Variante: "user_filter".

ZIEL: so nah wie technisch möglich am TATSÄCHLICHEN manuellen Einstiegsprozess
eines erfahrenen Memecoin-Traders nachbilden, wie er ihn per Axiom-Filter-
Screenshots und mündlicher Beschreibung an den Coordinator gegeben hat. Läuft
komplett parallel zu "baseline", "mcap_hypothesis" und "wallet_signal" (eigener
Strategy-Tag "user_filter" in trades.csv/state.json, siehe run_loop.py) - kein
bestehender Code wird dafür verändert.

QUELLE DER SCHWELLEN: Axiom-"Trending"-Filtereinstellungen laut Nutzer-
Screenshots + O-Ton "wenn ich sehe so paar kaufen und immer mehr, dann rein".
Alle Zahlen unten sind NUTZER-VORGABEN, nicht neu geraten - Ausnahme ist der
Take-Profit-Ansatz (siehe TAKE_PROFIT_RESEARCH_NOTE), der explizit als
recherchiert/geschätzt gekennzeichnet ist, weil der Nutzer dazu keine Zahl
genannt hat.

--------------------------------------------------------------------------
TEIL 1 - DATENVERFÜGBARKEITS-CHECK (2026-09-18, aus dieser Umgebung heraus
tatsächlich geprüft, siehe rugcheck_client.py für die Basis-Erreichbarkeit):

RugCheck.xyz /v1/tokens/{mint}/report (live gegen einen echten pump.fun-Mint
getestet) liefert u.a.: mint, creator, creatorBalance, mintAuthority,
freezeAuthority, totalHolders, topHolders[], totalMarketLiquidity,
totalStableLiquidity, totalLPProviders, lockers/lockerOwners, launchpad,
markets[] (Pool-/AMM-Details), score, score_normalised, risks[],
graphInsidersDetected (Zahl erkannter Accounts in Insider-Transfer-Clustern),
insiderNetworks[] (Cluster-Grössen, keine %-Zahl).

Ergebnis pro vom Nutzer genanntem, "leer gelassenem" Feld:
  - Top-10-Holder-% : JA verfügbar (topHolders[].pct, wird bereits in
    filter.py:apply_rugcheck() für einen eigenen, weicheren Schwellwert
    genutzt) -> HIER als eigener, strengerer Wert (0.25 statt 0.30)
    nachgerechnet, siehe _rugcheck_top10_and_dev_pct().
  - Dev-Holding-% (Creator-Wallet-Anteil) : JA verfügbar (creatorBalance /
    token.supply) - informativ mitgeloggt.
  - Liquidity ($) : JA verfügbar, aber über Dexscreener (Candidate.liquidity_usd),
    nicht über RugCheck - informativ mitgeloggt.
  - Bundle-% : NEIN. Kein Feld in der RugCheck-Response entspricht Axioms
    "Bundle %" (gemeinsam finanzierte/"gebündelte" Wallet-Cluster beim Launch).
    Auch die getestete Bitquery-GraphQL-API (streaming.bitquery.io/graphql)
    verlangt einen Auth-Token (HTTP 401 ohne Key) - keine freie Alternative
    gefunden. -> bleibt NICHT VERFÜGBAR, kein Ersatzwert erfunden.
  - Snipers-% : NEIN. Kein entsprechendes Feld gefunden (weder RugCheck noch
    Dexscreener). -> NICHT VERFÜGBAR.
  - B.curve-% (Bonding-Curve-Fortschritt) : NEIN. Weder RugCheck noch
    Dexscreener liefern den pump.fun-Bonding-Curve-Fortschritt in %. Für
    bereits zu einem AMM migrierte Pools (dexId != "pumpfun") ist die Kurve
    ohnehin faktisch bei 100%, aber das ist keine echte Prozentzahl aus der
    Quelle. -> NICHT VERFÜGBAR.
  - Insiders-% : TEILWEISE. RugCheck liefert `graphInsidersDetected` (rohe
    Anzahl Accounts in erkannten Insider-Transfer-Clustern) und
    `insiderNetworks` (Cluster-Grössen), aber KEINE Prozentzahl der Supply
    analog zu Axioms "Insiders %"-Spalte. Wird informativ als Rohzahl
    mitgeloggt, NICHT als Ersatz für die Axiom-Metrik behauptet.
  - Global Fees Paid : NEIN. Weder RugCheck noch Dexscreener liefern
    plattformweite Gebühren-Summen. -> NICHT VERFÜGBAR.

=> Ergebnis: Bundle-%, Snipers-%, B.curve-% und Global-Fees bleiben im
Trade-Log als "nicht verfügbar" markiert (siehe INFO_FIELD_UNAVAILABLE).
Sie bilden gemäss Auftrag KEIN Gate - nur Top-10-Holder-% (Nutzer-Vorgabe,
verfügbar) ist ein hartes Gate.
--------------------------------------------------------------------------
"""
from __future__ import annotations

import dataclasses
import datetime as dt

from . import rugcheck_client
from .models import Candidate
from .wallet_signal import (
    CONFLUENCE_SIGNALS_CSV_PATH,
    WALLET_TRADES_CSV_PATH,
    _parse_dt,
    _read_csv_rows,
)

INFO_FIELD_UNAVAILABLE = "nicht verfügbar (keine freie Quelle gefunden, siehe Modul-Docstring)"

# --------------------------------------------------------------------------
# Benannte Parameter - Nutzer-Vorgaben aus den Axiom-Filter-Screenshots.
# --------------------------------------------------------------------------
USER_FILTER_PARAMS = {
    # --- harte, vom Nutzer genannte Zahlen-Gates ---
    "min_volume_usd": 7_000.0,
    "min_market_cap_usd": 7_000.0,
    "max_age_minutes": 20.0,  # Nutzer nennt manchmal 15 statt 20 - 20 ist der
    # Default, absichtlich als eigener, leicht änderbarer Parameter (siehe
    # ALT_MAX_AGE_MINUTES_STRICT) statt hart im Code verstreut.
    "max_top10_holder_pct": 0.25,
    "max_bundle_pct": 0.55,  # Nutzer-Vorgabe, aktuell NICHT durchsetzbar (siehe
    # Docstring TEIL 1) - bleibt hier dokumentiert für den Tag, an dem eine
    # Bundle-%-Quelle verfügbar wird, bildet bis dahin KEIN Gate.

    # --- weiche Nutzer-Präferenz, kein hartes Gate ---
    # Nutzer: "eher unter 50k" für den Einstieg - explizit weicher formuliert
    # als die harten Min-Werte oben, deshalb NICHT hart blockierend, nur als
    # Warn-Reason geloggt, wenn überschritten.
    "soft_max_market_cap_usd": 50_000.0,

    # --- Exit ---
    # Nutzer nennt eine Stop-Loss-BANDBREITE von -40% bis -50%, keinen
    # Einzelwert. -45% (Mittelwert) ist hier der Default, min/max bleiben als
    # eigene Parameter dokumentiert, falls später an einem Bandrand getestet
    # werden soll.
    "stop_loss_pct_default": 0.45,
    "stop_loss_pct_min": 0.40,
    "stop_loss_pct_max": 0.50,
    "mcap_exit_floor_usd": 10_000.0,  # Exit, wenn Mcap wieder darunter fällt.
    "max_hold_hours": 6.0,  # kein Nutzer-Limit genannt; gleiches Sicherheitsnetz
    # wie die anderen drei Strategien (siehe strategy.MAX_HOLD_HOURS), damit
    # keine Position technisch für immer offen bleibt, falls weder SL noch
    # Mcap-Floor je auslösen.

    # --- Echtzeit-Wallet-Konfluenz mit ZUNEHMENDEM Trend ---
    # Nutzer: "wenn ich sehe so paar kaufen und immer mehr, dann rein" - das
    # ist explizit ein TREND (Ableitung), nicht nur "irgendeine Wallet kauft"
    # (das deckt bereits wallet_signal.py mit min_wallets_required=1 ab).
    # Umsetzung: Anzahl distinct kaufender Tracked-Wallets im letzten
    # 5-Minuten-Fenster muss (a) höher sein als im 5-Minuten-Fenster DAVOR
    # UND (b) mindestens wallet_trend_min_current_wallets erreichen (sonst
    # wäre z.B. "1 -> 2" schon ein technischer Anstieg, aber genau das meint
    # der Nutzer mit "so paar" nicht - er will mind. 2 aktuell aktive
    # Wallets sehen).
    "wallet_trend_window_minutes": 5.0,
    "wallet_trend_min_current_wallets": 2,
}

ALT_MAX_AGE_MINUTES_STRICT = 15.0  # vom Nutzer manchmal genannter strengerer Wert; nicht Default, aber hier als Referenz benannt.

# --------------------------------------------------------------------------
# TAKE-PROFIT: vom Nutzer NICHT vorgegeben. Auf ausdrücklichen Auftrag hin
# recherchiert/geschätzt (Stand Wissen bis 2026-01, keine Live-WebSearch in
# dieser Session verfügbar) - in der Memecoin-Trading-Community verbreiteter
# Ansatz bei "früher Sub-50k-Mcap-Einstieg, Ziel ist ein Vielfaches": GESTAFFELTE
# Teilverkäufe statt einem einzelnen TP-Preis, z.B. ~30-40% der Position bei
# 2x raus (Einsatz sichern/"Runner" fahren), weitere ~30% bei 3-5x, Rest mit
# nachgezogenem Stop laufen lassen. Grund: die Verteilung von Memecoin-Erfolgen
# ist extrem rechtsschief (wenige Ausreisser tragen den Erwartungswert), ein
# einzelnes festes TP-Ziel kappt genau diese Ausreisser systematisch.
#
# NICHT implementiert als Zwangs-Exit: SlippageAwarePaperBroker/state.json
# unterstützen in diesem Prototyp (wie bei baseline/mcap_hypothesis/
# wallet_signal auch) nur volle Buy/Sell-Positionen, keine Teilverkäufe -
# das für alle vier Strategien konsistent zu ändern hätte paper_broker.py
# angefasst, was ausserhalb des erlaubten Datei-Scopes dieses Auftrags liegt.
# Die Stages sind deshalb hier NUR als dokumentierter, begründeter Vorschlag
# hinterlegt (informativ im opened_reason geloggt), tatsächlicher Exit läuft
# ausschliesslich über Stop-Loss-Band bzw. Mcap-Floor unten.
TAKE_PROFIT_RESEARCH_NOTE = (
    "Kein Nutzer-TP-Ziel vorgegeben. Recherchiert/geschätzt, NICHT vom Nutzer "
    "bestätigt: gestaffelte Teilverkäufe ~35% bei 2x, ~30% bei 3-5x, Rest mit "
    "nachgezogenem Stop laufen lassen - hier nur dokumentiert, nicht als "
    "Zwangs-Exit implementiert (Paper-Broker unterstützt keine Teilverkäufe, "
    "siehe Kommentar oben)."
)


@dataclasses.dataclass
class WalletTrendResult:
    token_mint: str
    matched: bool
    wallets_recent_window: int
    wallets_prior_window: int
    increasing: bool
    window_minutes: float
    min_required_current: int


@dataclasses.dataclass
class UserFilterEntryResult:
    candidate_token: str
    eligible: bool
    reasons: list[str]
    top10_holder_pct: float | None
    dev_holder_pct: float | None
    liquidity_usd: float | None
    insiders_raw_count: float | None  # RugCheck graphInsidersDetected, KEINE %-Zahl
    bundle_pct: str | float | None  # bleibt Platzhalter-String, solange keine Quelle existiert
    snipers_pct: str | float | None
    bcurve_pct: str | float | None
    global_fees_paid: str | float | None
    wallet_trend: WalletTrendResult


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _distinct_buying_wallets(token_mint: str, window_start: dt.datetime, window_end: dt.datetime) -> set[str]:
    """Eigenständige (in dieser Datei implementierte) Hilfsfunktion, NICHT in
    wallet_signal.py geändert: liefert die Menge distinct Tracked-Wallet-
    Adressen mit einem Buy für `token_mint` in [window_start, window_end).
    Kombiniert wallet_trades.csv + confluence_signals.csv wie
    wallet_signal.check_wallet_signal(), aber fensterweise statt kumulativ,
    weil hier der TREND zwischen zwei Fenstern interessiert, nicht nur "gab es
    irgendwann einen Kauf"."""
    wallets: set[str] = set()
    for row in _read_csv_rows(WALLET_TRADES_CSV_PATH):
        if row.get("token_mint") != token_mint or row.get("action") != "buy":
            continue
        block_time = _parse_dt(row.get("block_time"))
        if block_time is None or block_time < window_start or block_time >= window_end:
            continue
        address = row.get("wallet_address")
        if address:
            wallets.add(address)
    for row in _read_csv_rows(CONFLUENCE_SIGNALS_CSV_PATH):
        if row.get("token_mint") != token_mint or row.get("action") != "buy":
            continue
        window_end_ts = _parse_dt(row.get("window_end"))
        if window_end_ts is None or window_end_ts < window_start or window_end_ts >= window_end:
            continue
        for address in (row.get("wallet_addresses") or "").split("|"):
            address = address.strip()
            if address:
                wallets.add(address)
    return wallets


def check_wallet_trend(token_mint: str, now: dt.datetime | None = None) -> WalletTrendResult:
    """Prüft die Nutzer-Bedingung "so paar kaufen und immer mehr": Anzahl
    distinct kaufender Tracked-Wallets im letzten Fenster (wallet_trend_window_minutes)
    muss grösser sein als im Fenster davor UND mind. wallet_trend_min_current_wallets
    erreichen. Robust gegen fehlende/leere CSVs (liefert dann 0/0, matched=False,
    NIE eine Exception - analog wallet_signal.check_wallet_signal())."""
    now = now or _now()
    window = USER_FILTER_PARAMS["wallet_trend_window_minutes"]
    recent_start = now - dt.timedelta(minutes=window)
    prior_start = now - dt.timedelta(minutes=2 * window)

    recent = _distinct_buying_wallets(token_mint, recent_start, now)
    prior = _distinct_buying_wallets(token_mint, prior_start, recent_start)

    increasing = len(recent) > len(prior)
    min_required = USER_FILTER_PARAMS["wallet_trend_min_current_wallets"]
    matched = increasing and len(recent) >= min_required

    return WalletTrendResult(
        token_mint=token_mint,
        matched=matched,
        wallets_recent_window=len(recent),
        wallets_prior_window=len(prior),
        increasing=increasing,
        window_minutes=window,
        min_required_current=min_required,
    )


def _rugcheck_top10_and_dev_pct(token_address: str) -> tuple[float | None, float | None, float | None, bool]:
    """Fragt (falls RugCheck-Tagesbudget vorhanden) den RugCheck-Report ab und
    liefert (top10_holder_pct, dev_holder_pct, graphInsidersDetected, checked).
    checked=False bedeutet "konnte nicht geprüft werden" (kein Budget, Fetch
    fehlgeschlagen, Token nicht indiziert) - wird vom Aufrufer konservativ als
    'Gate nicht bestätigbar' behandelt, NICHT als 'unauffällig' (gleiche
    Philosophie wie filter.py: unbekannt != neutral)."""
    if not rugcheck_client.has_budget():
        return None, None, None, False
    report = rugcheck_client.fetch_report(token_address)
    if report is None:
        return None, None, None, False

    top10_pct = None
    top_holders = report.get("topHolders") or []
    if top_holders:
        pct_values = [h.get("pct") for h in top_holders[:10] if isinstance(h.get("pct"), (int, float))]
        if pct_values:
            total = sum(pct_values)
            top10_pct = total / 100.0 if total > 1 else total

    dev_pct = None
    supply = (report.get("token") or {}).get("supply")
    creator_balance = report.get("creatorBalance")
    if supply and creator_balance is not None and supply > 0:
        dev_pct = creator_balance / supply

    insiders_raw = report.get("graphInsidersDetected")

    return top10_pct, dev_pct, insiders_raw, True


def check_user_filter_entry(c: Candidate, now: dt.datetime | None = None) -> UserFilterEntryResult:
    """Volle Einstiegsprüfung der "user_filter"-Strategie: Zahlen-Gates (Volume,
    Mcap-Min, Alter, Top-10-Holder) UND der zunehmende Wallet-Konfluenz-Trend.
    Liefert IMMER ein UserFilterEntryResult (nie None/Exception)."""
    now = now or _now()
    params = USER_FILTER_PARAMS
    reasons: list[str] = []
    hard_fail = False

    # --- Alter ---
    if c.age_seconds is None:
        hard_fail = True
        reasons.append("Alter unbekannt (kein pairCreatedAt) -> ausgeschlossen.")
    else:
        age_minutes = c.age_seconds / 60.0
        if age_minutes > params["max_age_minutes"]:
            hard_fail = True
            reasons.append(f"Alter {age_minutes:.1f}min > Maximum {params['max_age_minutes']:.0f}min.")
        else:
            reasons.append(f"Alter {age_minutes:.1f}min OK (Maximum {params['max_age_minutes']:.0f}min).")

    # --- Volume ($) Min ---
    # Näherung: volume_h1 aus Dexscreener als Stellvertreter für Axioms
    # "Volume ($)"-Spalte - für Coins <20min entspricht h1-Volumen i.d.R.
    # praktisch dem gesamten bisherigen Handelsvolumen.
    if c.volume_h1 is None:
        hard_fail = True
        reasons.append("Volumen (h1) nicht verfügbar -> ausgeschlossen (konservativ, wie unbekannte Liquidität in filter.py).")
    elif c.volume_h1 < params["min_volume_usd"]:
        hard_fail = True
        reasons.append(f"Volumen ${c.volume_h1:,.0f} < Minimum ${params['min_volume_usd']:,.0f}.")
    else:
        reasons.append(f"Volumen ${c.volume_h1:,.0f} OK.")

    # --- Market Cap ($) Min + weiche Obergrenze ---
    if c.market_cap is None:
        hard_fail = True
        reasons.append("Marketcap unbekannt -> ausgeschlossen.")
    elif c.market_cap < params["min_market_cap_usd"]:
        hard_fail = True
        reasons.append(f"Marketcap ${c.market_cap:,.0f} < Minimum ${params['min_market_cap_usd']:,.0f}.")
    else:
        reasons.append(f"Marketcap ${c.market_cap:,.0f} >= Minimum OK.")
        if c.market_cap > params["soft_max_market_cap_usd"]:
            reasons.append(
                f"Hinweis: Marketcap ${c.market_cap:,.0f} > weicher Nutzer-Präferenz "
                f"${params['soft_max_market_cap_usd']:,.0f} ('eher unter 50k') - KEIN hartes Gate, nur Warnhinweis."
            )

    # --- Top-10-Holder-% (hartes Gate, RugCheck) ---
    top10_pct, dev_pct, insiders_raw, checked = _rugcheck_top10_and_dev_pct(c.token_address)
    if not checked or top10_pct is None:
        hard_fail = True
        reasons.append(
            "Top-10-Holder-% nicht verifizierbar (RugCheck-Budget erschöpft, Fetch fehlgeschlagen oder "
            "Token nicht indiziert) -> konservativ als nicht bestätigtes Gate behandelt, kein Einstieg."
        )
    elif top10_pct > params["max_top10_holder_pct"]:
        hard_fail = True
        reasons.append(f"Top-10-Holder {top10_pct:.0%} > Maximum {params['max_top10_holder_pct']:.0%}.")
    else:
        reasons.append(f"Top-10-Holder {top10_pct:.0%} OK (Maximum {params['max_top10_holder_pct']:.0%}).")

    # --- informative, NICHT gate-bildende Felder ---
    reasons.append(f"Bundle-%: {INFO_FIELD_UNAVAILABLE}.")
    reasons.append(f"Snipers-%: {INFO_FIELD_UNAVAILABLE}.")
    reasons.append(f"B.curve-%: {INFO_FIELD_UNAVAILABLE}.")
    reasons.append(f"Global Fees Paid: {INFO_FIELD_UNAVAILABLE}.")
    if dev_pct is not None:
        reasons.append(f"Dev-Holding-% (informativ, kein Gate): {dev_pct:.1%}.")
    else:
        reasons.append(f"Dev-Holding-% (informativ): {INFO_FIELD_UNAVAILABLE}.")
    if c.liquidity_usd is not None:
        reasons.append(f"Liquidity ${c.liquidity_usd:,.0f} (informativ, kein Gate).")
    else:
        reasons.append(f"Liquidity (informativ): {INFO_FIELD_UNAVAILABLE}.")
    if insiders_raw is not None:
        reasons.append(
            f"Insiders (informativ, RugCheck graphInsidersDetected={insiders_raw:.0f} Accounts, "
            "KEINE %-Zahl analog Axiom): kein Gate."
        )
    else:
        reasons.append(f"Insiders (informativ): {INFO_FIELD_UNAVAILABLE}.")

    # --- Wallet-Konfluenz mit zunehmendem Trend ---
    wallet_trend = check_wallet_trend(c.token_address, now=now)
    if not wallet_trend.matched:
        hard_fail = True
        reasons.append(
            f"Wallet-Trend NICHT erfüllt: {wallet_trend.wallets_recent_window} kaufende Wallets in den "
            f"letzten {wallet_trend.window_minutes:.0f}min vs. {wallet_trend.wallets_prior_window} im Fenster "
            f"davor (benötigt: zunehmend UND >= {wallet_trend.min_required_current})."
        )
    else:
        reasons.append(
            f"Wallet-Trend erfüllt: {wallet_trend.wallets_recent_window} kaufende Wallets in den letzten "
            f"{wallet_trend.window_minutes:.0f}min, zunehmend gegenüber {wallet_trend.wallets_prior_window} davor."
        )

    return UserFilterEntryResult(
        candidate_token=c.token_address,
        eligible=not hard_fail,
        reasons=reasons,
        top10_holder_pct=top10_pct,
        dev_holder_pct=dev_pct,
        liquidity_usd=c.liquidity_usd,
        insiders_raw_count=insiders_raw,
        bundle_pct=INFO_FIELD_UNAVAILABLE,
        snipers_pct=INFO_FIELD_UNAVAILABLE,
        bcurve_pct=INFO_FIELD_UNAVAILABLE,
        global_fees_paid=INFO_FIELD_UNAVAILABLE,
        wallet_trend=wallet_trend,
    )


def should_exit_user_filter(
    entry_price: float,
    current_price: float,
    entry_mcap: float | None,
    current_mcap: float | None,
    entry_time: dt.datetime,
    now: dt.datetime,
) -> tuple[bool, str] | tuple[bool, None]:
    """Exit-Regeln der user_filter-Strategie: Stop-Loss-Band (-45% Default,
    Nutzer-Bandbreite -40% bis -50%) ODER Mcap faellt unter mcap_exit_floor_usd
    ($10.000) - was ZUERST eintritt. Beide werden pro Poll geprüft (dieser
    Prototyp ist kein Tick-Stream); wenn innerhalb desselben Polls beide
    Bedingungen bereits erfüllt sind, wird das als "beide gleichzeitig
    ausgelöst" behandelt und der Stop-Loss zuerst gemeldet - eine exakte
    Reihenfolge zwischen zwei Polls ist mit diesem Cron-Modell nicht
    feststellbar. Kein Take-Profit-Zwangsexit (siehe TAKE_PROFIT_RESEARCH_NOTE)
    und MAX_HOLD_HOURS als zusätzliches Sicherheitsnetz, falls weder SL noch
    Mcap-Floor je auslösen."""
    params = USER_FILTER_PARAMS
    if entry_price <= 0:
        return False, None

    change = (current_price - entry_price) / entry_price
    if change <= -params["stop_loss_pct_default"]:
        return True, f"user_filter_stop_loss ({change:+.1%}, Default -{params['stop_loss_pct_default']:.0%} aus Band -{params['stop_loss_pct_min']:.0%}/-{params['stop_loss_pct_max']:.0%})"

    if current_mcap is not None and current_mcap < params["mcap_exit_floor_usd"]:
        return True, f"user_filter_mcap_floor (Mcap ${current_mcap:,.0f} < ${params['mcap_exit_floor_usd']:,.0f})"

    held_hours = (now - entry_time).total_seconds() / 3600.0
    if held_hours >= params["max_hold_hours"]:
        return True, f"user_filter_max_hold_time ({held_hours:.1f}h, Sicherheitsnetz - kein Nutzer-Limit vorgegeben)"

    return False, None
