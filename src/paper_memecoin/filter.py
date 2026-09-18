"""Rug-Pull-Heuristik-Score.

EINORDNUNG (Quelle: arXiv 2608.20271, "Catching the Rug: Early Prediction of
Fraudulent Memecoins on Solana via Machine Learning"): ca. 99% der pump.fun-
Launches zeigen Rug-Pull-/Pump-and-Dump-Muster, und die meisten Rug-Merkmale
werden bereits in der ersten Handelsstunde sichtbar. Das ist keine Schätzung
des Nutzers, sondern eine zitierte Studie - dieser Filter kann diese Quote
NICHT auf ein "sicheres" Niveau drücken, sondern bestenfalls die offensichtlich
schlechtesten Kandidaten aussortieren. Erfolgsgeschichten von Sniper-Bot-
Anbietern ("56% Winrate", "$10k/Monat") sind unverifizierte Marketing-
Testimonials mit starkem Survivorship-Bias und fliessen NICHT in die
Kalibrierung dieser Schwellen ein.

ZWEISTUFIGER FILTER:
  Stufe 1 (score_candidate, immer, kostenlos): Dexscreener-Signale.
  Stufe 2 (apply_rugcheck, nur für Stufe-1-Survivors, Tagesbudget begrenzt):
           RugCheck.xyz-Signale (Mint-/Freeze-Authority, LP-Lock, Holder-
           Konzentration, Dev-Wallet-Anteil, RugCheck-eigener Risiko-Score).

EHRLICHE BESTANDSAUFNAHME - was die Datenquelle (Dexscreener, keyless) liefert
und was NICHT:

VERFÜGBAR und hier genutzt:
- liquidity.usd            -> NUR vorhanden, sobald ein Token von der reinen
                               pump.fun-Bonding-Curve zu einem echten AMM-Pool
                               migriert ist (dexId "pumpswap", "raydium",
                               "meteora", ...). Für Tokens, die NOCH auf der
                               pump.fun-Bonding-Curve sind (dexId "pumpfun"),
                               liefert die API liquidity=None. Das trifft
                               gerade auf die ALLERFRISCHESTEN Launches (<20min)
                               überdurchschnittlich oft zu - genau die Coins,
                               die wir eigentlich bewerten wollen, haben also
                               oft NOCH KEINE Liquiditätszahl.
- marketCap / fdv          -> vorhanden, auch für reine Bonding-Curve-Tokens.
- pairCreatedAt (-> Alter)  -> vorhanden.
- txns.h1.buys / .sells     -> vorhanden, als grobes Aktivitäts-/Dump-Signal.
- Vorhandensein von Website/Social-Links im Profil -> vorhanden, aber sehr
  schwaches Signal (leicht fälschbar, kostet nichts).

NICHT VERFÜGBAR über Dexscreener (bewusst NICHT im Stufe-1-Score, keine
erfundenen Felder) - diese vier Signale liefert stattdessen RugCheck.xyz in
Stufe 2, siehe apply_rugcheck() weiter unten, ABER nur für die wenigen
Kandidaten, die Stufe 1 bestehen (Tagesbudget-Limit von RugCheck ohne Key):
- Holder-Anzahl / Holder-Konzentration (Top-10-Wallet-Anteil)
- LP-Token-Lock-Status / Lock-Dauer
- Dev-Wallet-Anteil bzw. ob der Dev bereits verkauft hat
- Contract-Verifizierung / Mint-/Freeze-Authority-Status
Für den grossen Rest der pro Poll gesehenen Kandidaten (die an Stufe 1
scheitern) bleiben diese vier Signale schlicht ungenutzt - das ist die
grösste strukturelle Einschränkung dieses Prototyps. Der Score ist deshalb
explizit eine GROBE Heuristik, kein verlässlicher Rug-Detektor.
"""
from __future__ import annotations

from . import rugcheck_client
from .models import Candidate, FilterResult

# --- Stufe-2-Schwellen (RugCheck.xyz) -----------------------------------
# Von der Produktseite/Coordinator als "reale Schwellen erfahrener Sniper/
# Trader laut mehreren unabhängigen Quellen" vorgegeben - hier als benannte,
# dokumentierte Defaults übernommen, nicht neu geraten:
RUGCHECK_MINT_AUTHORITY_MUST_BE_REVOKED = True
RUGCHECK_FREEZE_AUTHORITY_MUST_BE_REVOKED = True
RUGCHECK_MAX_DEV_WALLET_PCT = 0.25  # Dev-/Creator-Wallet < 20-25% der Supply -> oberes Ende genutzt
RUGCHECK_MAX_TOP10_HOLDER_PCT = 0.30  # Top-10-Holder zusammen < 30%
RUGCHECK_MIN_HOLDERS = 30  # nur durchsetzbar, wenn RugCheck totalHolders>0 liefert (siehe Docstring oben)
# score_normalised: in unseren Live-Tests hatten unauffällige Tokens Werte
# nahe 0-10, ein Token mit einer aktiven Warnung ("High market cap per
# holder") lag bei 50. Offizielle RugCheck-Dokumentation zur exakten Skala
# war aus dieser Umgebung nicht einsehbar - dieser Schwellwert ist deshalb
# EMPIRISCH aus den eigenen Tests kalibriert, nicht aus RugCheck-Docs zitiert.
RUGCHECK_MAX_SCORE_NORMALISED = 50

# Schwellwerte - bewusst konservativ, da ~99% Rug-/Pump-and-Dump-Quote bei
# pump.fun-Launches per Studie belegt ist (siehe Docstring oben). Ziel ist
# NICHT "möglichst viele Trades", sondern nur die (wenigen) am wenigsten
# offensichtlich verdächtigen Kandidaten durchzulassen.
# MIN_LIQUIDITY_USD = $5k ist der von erfahrenen Tradern genannte ABSOLUTE
# Boden ("darunter reicht ein einzelner Verkauf zum Crash"); $50k+ gilt als
# robuster, ist aber für <20-Minuten-alte Launches praktisch unerreichbar
# (würde fast alle Kandidaten ausschliessen) - siehe stattdessen den Bonus
# für Liquidität >= ROBUST_LIQUIDITY_USD weiter unten im Score.
MIN_LIQUIDITY_USD = 5_000.0
ROBUST_LIQUIDITY_USD = 50_000.0
MIN_LIQ_TO_MCAP = 0.04
MIN_MARKET_CAP = 6_000.0  # auf Nutzerwunsch angehoben von 2.000 - unter ~6k
# ist eine Marktkapitalisierung fuer ernsthafte Bewertung ohnehin zu duenn
MAX_MARKET_CAP = 500_000.0  # deutlich höher -> vermutlich schon durchgelaufen
MIN_AGE_SECONDS = 60.0  # unter 1 Minute: noch keine verwertbaren Handelsdaten
MIN_BUY_SELL_RATIO = 0.8  # mehr Verkäufer als Käufer im h1-Fenster -> Warnsignal

SCORE_THRESHOLD = 60.0  # von max. 100 Punkten


def score_candidate(c: Candidate) -> FilterResult:
    reasons: list[str] = []
    unavailable: list[str] = [
        "holder_count", "holder_concentration_top10", "lp_lock_status", "dev_wallet_share",
    ]
    score = 100.0
    hard_fail = False

    # --- Alter ---
    if c.age_seconds is None:
        hard_fail = True
        reasons.append("Kein pairCreatedAt -> Alter unbekannt, ausgeschlossen.")
    elif c.age_seconds < MIN_AGE_SECONDS:
        score -= 15
        reasons.append(f"Sehr jung ({c.age_seconds:.0f}s) - kaum Handelsdaten.")

    # --- Liquidität ---
    if c.liquidity_usd is None:
        # Kein Bluff: wir können hier schlicht nichts über Liquidität sagen.
        # Konservativ: starker Punktabzug statt Ignorieren.
        score -= 40
        reasons.append(
            f"liquidity_usd nicht verfügbar (dexId={c.dex_id}, vermutlich noch auf "
            "pump.fun-Bonding-Curve, noch nicht zu AMM migriert) - Risiko unbekannt."
        )
    elif c.liquidity_usd < MIN_LIQUIDITY_USD:
        hard_fail = True
        reasons.append(f"Liquidität ${c.liquidity_usd:,.0f} < Minimum ${MIN_LIQUIDITY_USD:,.0f}.")
    elif c.liquidity_usd >= ROBUST_LIQUIDITY_USD:
        score += 5
        reasons.append(f"Liquidität ${c.liquidity_usd:,.0f} >= robustem Richtwert ${ROBUST_LIQUIDITY_USD:,.0f}.")
    else:
        reasons.append(f"Liquidität ${c.liquidity_usd:,.0f} OK (über Minimum, unter robustem Richtwert).")

    # --- Liquidität / Marketcap ---
    ratio = c.liquidity_to_mcap
    if ratio is not None:
        if ratio < MIN_LIQ_TO_MCAP:
            score -= 20
            reasons.append(f"Liq/MCap-Ratio {ratio:.2%} < {MIN_LIQ_TO_MCAP:.0%} - dünn gegen Marketcap.")
        else:
            reasons.append(f"Liq/MCap-Ratio {ratio:.2%} OK.")
    else:
        reasons.append("Liq/MCap-Ratio nicht berechenbar (Liquidität oder MCap fehlt).")

    # --- Marketcap-Range ---
    if c.market_cap is None:
        score -= 10
        reasons.append("Marketcap unbekannt.")
    elif c.market_cap < MIN_MARKET_CAP:
        hard_fail = True
        reasons.append(f"Marketcap ${c.market_cap:,.0f} < Minimum ${MIN_MARKET_CAP:,.0f} (Dust).")
    elif c.market_cap > MAX_MARKET_CAP:
        score -= 15
        reasons.append(f"Marketcap ${c.market_cap:,.0f} > ${MAX_MARKET_CAP:,.0f} - früher Vorteil evtl. schon weg.")

    # --- Buy/Sell-Verhältnis (Dump-Signal) ---
    bs = c.buy_sell_ratio_h1
    if bs is not None and bs != float("inf"):
        if bs < MIN_BUY_SELL_RATIO:
            score -= 20
            reasons.append(f"Buy/Sell-Ratio {bs:.2f} < {MIN_BUY_SELL_RATIO} - mehr Verkäufe als Käufe.")
        else:
            reasons.append(f"Buy/Sell-Ratio {bs:.2f} OK.")
    else:
        reasons.append("Buy/Sell-Ratio nicht verfügbar/nicht berechenbar.")

    # --- Website/Social (sehr schwaches Signal) ---
    if not c.has_website and not c.has_social:
        score -= 5
        reasons.append("Kein Website-/Social-Link im Profil (schwaches Signal).")

    score = max(0.0, min(100.0, score))
    passed = (not hard_fail) and score >= SCORE_THRESHOLD

    return FilterResult(
        candidate=c,
        score=score,
        passed=passed,
        reasons=reasons,
        unavailable_signals=unavailable,
    )


def apply_rugcheck(result: FilterResult) -> FilterResult:
    """Stufe 2: nur für Kandidaten, die Stufe 1 (score_candidate) bereits
    bestanden haben. Kann einen bestandenen Stufe-1-Kandidaten NACHTRÄGLICH
    durchfallen lassen (mintAuthority aktiv, freezeAuthority aktiv, rugged,
    zu hohe Dev-/Top10-Konzentration, zu hoher RugCheck-Risiko-Score).
    Fällt bei fehlendem Tagesbudget oder Netzwerkfehler sauber auf das
    Stufe-1-Ergebnis zurück (dokumentiert in reasons, kein stiller Fehler)."""
    if not result.passed:
        return result

    if not rugcheck_client.has_budget():
        result.reasons.append("RugCheck übersprungen: Tagesbudget erschöpft, Stufe-1-Ergebnis bleibt massgeblich.")
        result.unavailable_signals.append("rugcheck_all_signals (quota_exhausted)")
        return result

    report = rugcheck_client.fetch_report(result.candidate.token_address)
    if report is None:
        result.reasons.append("RugCheck-Abfrage fehlgeschlagen/kein Report - Stufe-1-Ergebnis bleibt massgeblich.")
        result.unavailable_signals.append("rugcheck_all_signals (fetch_failed)")
        return result

    reasons = list(result.reasons)
    score = result.score
    hard_fail = False

    if report.get("rugged") is True:
        hard_fail = True
        reasons.append("RugCheck: Token bereits als 'rugged' markiert.")

    mint_authority = report.get("mintAuthority")
    if RUGCHECK_MINT_AUTHORITY_MUST_BE_REVOKED and mint_authority is not None:
        hard_fail = True
        reasons.append(f"RugCheck: mintAuthority noch aktiv ({mint_authority}) - unbegrenztes Dilutions-Risiko.")
    else:
        reasons.append("RugCheck: mintAuthority revoked/nicht gesetzt.")

    freeze_authority = report.get("freezeAuthority")
    if RUGCHECK_FREEZE_AUTHORITY_MUST_BE_REVOKED and freeze_authority is not None:
        hard_fail = True
        reasons.append(f"RugCheck: freezeAuthority noch aktiv ({freeze_authority}).")
    else:
        reasons.append("RugCheck: freezeAuthority revoked/nicht gesetzt.")

    total_holders = report.get("totalHolders")
    if total_holders:
        if total_holders < RUGCHECK_MIN_HOLDERS:
            score -= 20
            reasons.append(f"RugCheck: nur {total_holders} Holder < Minimum {RUGCHECK_MIN_HOLDERS}.")
        else:
            reasons.append(f"RugCheck: {total_holders} Holder OK.")
    else:
        # Unbekannt darf nicht wie "unauffällig" behandelt werden: ein noch
        # nicht indizierter Token ist bei <20-Minuten-Launches der Normalfall
        # und genau das Fenster, in dem ein Rug ohne jede Vorwarnung passieren
        # kann. Analog zur Behandlung von liquidity_usd=None in Stufe 1
        # (dort -40): fehlendes Sicherheitssignal kostet Punkte, statt
        # neutral durchzurutschen.
        score -= 20
        reasons.append(
            "RugCheck: totalHolders=0/nicht indiziert (bei <20-Minuten-Launches häufig) - "
            "unbekannt, als Risikosignal gewertet, nicht als neutral."
        )

    top_holders = report.get("topHolders") or []
    top10_pct = None
    if top_holders:
        pct_values = [h.get("pct") for h in top_holders[:10] if isinstance(h.get("pct"), (int, float))]
        if pct_values:
            top10_pct = sum(pct_values) / 100.0 if sum(pct_values) > 1 else sum(pct_values)
    if top10_pct is not None:
        if top10_pct > RUGCHECK_MAX_TOP10_HOLDER_PCT:
            hard_fail = True
            reasons.append(f"RugCheck: Top-10-Holder {top10_pct:.0%} > Maximum {RUGCHECK_MAX_TOP10_HOLDER_PCT:.0%}.")
        else:
            reasons.append(f"RugCheck: Top-10-Holder {top10_pct:.0%} OK.")
    else:
        # Selbe Begruendung wie bei totalHolders oben: unbekannt != unauffaellig.
        score -= 15
        reasons.append(
            "RugCheck: Top-10-Holder-Konzentration nicht verfügbar/nicht indiziert - "
            "als Risikosignal gewertet, nicht als neutral."
        )

    supply = (report.get("token") or {}).get("supply")
    creator_balance = report.get("creatorBalance")
    if supply and creator_balance is not None and supply > 0:
        dev_pct = creator_balance / supply
        if dev_pct > RUGCHECK_MAX_DEV_WALLET_PCT:
            hard_fail = True
            reasons.append(f"RugCheck: Dev-/Creator-Wallet hält {dev_pct:.0%} > Maximum {RUGCHECK_MAX_DEV_WALLET_PCT:.0%}.")
        else:
            reasons.append(f"RugCheck: Dev-/Creator-Wallet-Anteil {dev_pct:.0%} OK.")
    else:
        reasons.append("RugCheck: Dev-Wallet-Anteil nicht berechenbar (supply/creatorBalance fehlt).")

    score_normalised = report.get("score_normalised")
    if score_normalised is not None:
        if score_normalised > RUGCHECK_MAX_SCORE_NORMALISED:
            score -= 25
            reasons.append(f"RugCheck score_normalised={score_normalised} > {RUGCHECK_MAX_SCORE_NORMALISED} (empirischer Schwellwert).")
        else:
            reasons.append(f"RugCheck score_normalised={score_normalised} OK.")
    risks = report.get("risks") or []
    if risks:
        risk_names = ", ".join(r.get("name", "?") for r in risks)
        reasons.append(f"RugCheck risks: {risk_names}")

    score = max(0.0, min(100.0, score))
    passed = (not hard_fail) and score >= SCORE_THRESHOLD

    return FilterResult(
        candidate=result.candidate,
        score=score,
        passed=passed,
        reasons=reasons,
        unavailable_signals=result.unavailable_signals,
    )
