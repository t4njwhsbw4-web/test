"""Rug-Pull-Heuristik-Score.

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

NICHT VERFÜGBAR (bewusst NICHT im Score, keine erfundenen Felder):
- Holder-Anzahl / Holder-Konzentration (Top-10-Wallet-Anteil)
- LP-Token-Lock-Status / Lock-Dauer
- Dev-Wallet-Anteil bzw. ob der Dev bereits verkauft hat
- Contract-Verifizierung / Mint-Authority-Status
Diese vier Signale sind bei seriösen Rug-Pull-Filtern eigentlich zentral -
ihr Fehlen ist die grösste Einschränkung dieses Prototyps. Der Score ist
deshalb explizit eine GROBE Heuristik, kein verlässlicher Rug-Detektor.
"""
from __future__ import annotations

from .models import Candidate, FilterResult

# Schwellwerte - bewusst konservativ, da 95-99% Totalausfallrate bei ganz
# frischen Sub-100k-MC-Launches empirisch belegt ist (siehe Research-Notiz
# des Nutzers). Ziel ist NICHT "möglichst viele Trades", sondern nur die
# (wenigen) am wenigsten offensichtlich verdächtigen Kandidaten durchzulassen.
MIN_LIQUIDITY_USD = 3_000.0
MIN_LIQ_TO_MCAP = 0.04
MIN_MARKET_CAP = 2_000.0
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
    else:
        reasons.append(f"Liquidität ${c.liquidity_usd:,.0f} OK.")

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
