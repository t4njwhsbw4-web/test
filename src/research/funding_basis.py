"""Cash-and-Carry (Basis-Trade): ehrliche Renditerechnung inkl. aller Kosten.

Mechanik: Spot long (Groesse N) + Perpetual short (Groesse N) auf dasselbe Asset.
Die Kursbewegung neutralisiert sich zwischen den Beinen; vereinnahmt wird die
Funding-Rate, die Perp-Longs an Perp-Shorts zahlen.

Die drei Stellen, an denen diese Strategie ueblicherweise schoengerechnet wird
und die hier explizit behandelt werden:
  1. Rendite auf NOMINAL statt auf GEBUNDENES KAPITAL. Gebunden ist Spot (1x)
     PLUS Margin fuer den Short. Kapitalmultiplikator C = 1 + Margin-Quote.
  2. Gebuehren nur fuer ein Bein / nur fuer den Einstieg. Es sind VIER Fills:
     Spot kaufen, Perp shorten, Perp schliessen, Spot verkaufen.
  3. Negative Funding-Phasen weggelassen oder nur als Mittelwert gezeigt.

Aufruf:  PYTHONPATH=. python src/research/funding_basis.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.research.funding_data import SYMBOLS, fetch_funding, fetch_spot_daily

# --- Kostenannahmen -------------------------------------------------------
# Pro Seite und Fill. Es gibt 4 Fills ueber den Lebenszyklus des Trades.
FEE_TAKER_BPS = 10.0   # 0.10 % pro Fill  -> 40 bps Round-Trip beide Beine
FEE_MAKER_BPS = 2.0    # 0.02 % pro Fill  ->  8 bps Round-Trip beide Beine
N_FILLS = 4

# Kapitalmultiplikatoren: C = 1 (Spot) + Margin-Quote des Short-Beins
CAPITAL_MULT = {"5x Short (C=1.20)": 1.20, "3x Short (C=1.33)": 1.333, "2x Short (C=1.50)": 1.50}

# Binance Maintenance-Margin-Rate fuer BTC/ETH in kleinen Groessen
MMR = 0.005


def load(symbol: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    return fetch_funding(symbol), fetch_spot_daily(symbol)


def daily_funding(fund: pd.DataFrame) -> pd.Series:
    """Summe der Funding-Raten pro UTC-Tag (Rate auf Nominal, Short erhaelt sie)."""
    s = fund.set_index("ts")["last_funding_rate"]
    return s.groupby(s.index.floor("D")).sum()


def regime(spot: pd.DataFrame) -> pd.Series:
    """Bull/Bear ueber 200-Tage-SMA des Spotpreises (klassische Regime-Definition)."""
    px = spot.set_index("ts")["close"]
    sma = px.rolling(200, min_periods=200).mean()
    return pd.Series(np.where(px > sma, "Bull", "Bear"), index=px.index).where(sma.notna())


def annualise(daily: pd.Series) -> float:
    """Einfache Annualisierung der Summe der Tagesraten (kein Compounding)."""
    if len(daily) == 0:
        return np.nan
    return daily.sum() / len(daily) * 365.0


def net_on_capital(gross_ann: float, hold_days: float, fee_bps: float, cmult: float) -> float:
    """Netto-Rendite p.a. auf das tatsaechlich gebundene Kapital.

    Gebuehren fallen EINMAL pro Trade-Lebenszyklus an und werden ueber die
    Haltedauer annualisiert: kurze Haltedauer -> Gebuehren fressen alles.
    """
    fee_total = N_FILLS * fee_bps / 10_000.0
    fee_ann = fee_total * 365.0 / hold_days
    return (gross_ann - fee_ann) / cmult


def negative_streaks(daily: pd.Series) -> pd.DataFrame:
    """Zusammenhaengende Phasen mit kumulativ negativem Funding."""
    neg = daily < 0
    grp = (neg != neg.shift()).cumsum()
    rows = []
    for _, blk in daily.groupby(grp):
        if blk.iloc[0] < 0:
            rows.append({"start": blk.index[0].date(), "end": blk.index[-1].date(),
                         "tage": len(blk), "kumuliert_pct": blk.sum() * 100})
    return pd.DataFrame(rows).sort_values("kumuliert_pct") if rows else pd.DataFrame()


def worst_drawdown_window(daily: pd.Series) -> dict:
    """Schlimmste zusammenhaengende Negativ-Phase im Sinne des Funding-Drawdowns.

    Nicht nur Tage mit rate<0, sondern der tiefste Ruecksetzer der kumulierten
    Funding-Kurve - das ist die Phase, die der Trader real als Verlust erlebt.
    """
    cum = daily.cumsum()
    peak = cum.cummax()
    dd = cum - peak
    if dd.min() >= 0:
        return {}
    end = dd.idxmin()
    start = peak.loc[:end].idxmax()
    return {"von": start.date(), "bis": end.date(),
            "dauer_tage": (end - start).days,
            "tiefe_pct": dd.min() * 100}


def max_runup(spot: pd.DataFrame, windows=(1, 7, 30, 90, 365)) -> dict:
    """Maximaler Kursanstieg ueber rollierende Fenster (echte High-Spitzen).

    Relevant fuer das Short-Bein: es verliert genau diesen Prozentsatz.
    Ohne Nachschuss wird bei Margin-Quote m etwa bei +(m - MMR) liquidiert.
    """
    df = spot.set_index("ts")
    out = {}
    for w in windows:
        fwd_high = df["high"].rolling(w, min_periods=1).max().shift(-(w - 1))
        ru = (fwd_high / df["close"].shift(1) - 1).dropna()
        out[w] = {"max_runup_pct": ru.max() * 100, "datum": ru.idxmax().date(),
                  "p99_pct": ru.quantile(0.99) * 100, "p999_pct": ru.quantile(0.999) * 100}
    return out


def required_margin(runup_pct: float) -> float:
    """Margin-Quote, die einen Kursanstieg von runup_pct ohne Liquidation ueberlebt."""
    return runup_pct / 100.0 + MMR


# --------------------------------------------------------------------------
def report() -> None:
    pd.set_option("display.width", 200)
    print("=" * 100)
    print("TEIL 1 - HISTORISCHE RENDITE DES BASIS-TRADES (Binance USD-M Perpetuals)")
    print("=" * 100)

    store: dict[str, pd.Series] = {}
    summary = []

    for sym in SYMBOLS:
        fund, spot = load(sym)
        d = daily_funding(fund)
        store[sym] = d
        gross = annualise(d)
        pos_share = (d > 0).mean() * 100
        summary.append({
            "symbol": sym, "von": d.index[0].date(), "bis": d.index[-1].date(),
            "tage": len(d),
            "brutto_ann_%": gross * 100,
            "median_tag_bps": d.median() * 10_000,
            "mittel_tag_bps": d.mean() * 10_000,
            "tage_positiv_%": pos_share,
        })

    sm = pd.DataFrame(summary)
    print("\n-- Brutto-Funding auf NOMINAL (noch ohne Gebuehren, ohne Kapitalbindung) --")
    print(sm.to_string(index=False, float_format=lambda x: f"{x:8.2f}"))

    # --- Rendite pro Kalenderjahr -----------------------------------------
    print("\n-- Brutto-Funding p.a. auf Nominal, pro Kalenderjahr (%) --")
    rows = {}
    for sym, d in store.items():
        rows[sym] = d.groupby(d.index.year).apply(annualise) * 100
    yr = pd.DataFrame(rows)
    print(yr.to_string(float_format=lambda x: f"{x:7.2f}"))

    # --- Netto auf gebundenes Kapital -------------------------------------
    print("\n-- NETTO p.a. auf GEBUNDENES KAPITAL, Haltedauer 1 Jahr (%) --")
    print("   (Gebuehren: 4 Fills; Taker 10 bps/Fill = 40 bps, Maker 2 bps/Fill = 8 bps)")
    for label, cm in CAPITAL_MULT.items():
        for fee_label, fee in (("Taker", FEE_TAKER_BPS), ("Maker", FEE_MAKER_BPS)):
            vals = {s: net_on_capital(annualise(d), 365, fee, cm) * 100 for s, d in store.items()}
            print(f"  {label:20s} {fee_label:6s} " +
                  "  ".join(f"{s.replace('USDT',''):>5s}={v:6.2f}" for s, v in vals.items()))

    # --- Gebuehrendruck bei kurzer Haltedauer -----------------------------
    print("\n-- Einfluss der Haltedauer (BTCUSDT, C=1.20, netto auf Kapital, %) --")
    g = annualise(store["BTCUSDT"])
    for hold in (7, 30, 90, 365, 1095):
        t = net_on_capital(g, hold, FEE_TAKER_BPS, 1.20) * 100
        m = net_on_capital(g, hold, FEE_MAKER_BPS, 1.20) * 100
        print(f"  Haltedauer {hold:5d} Tage: Taker {t:7.2f}   Maker {m:7.2f}")

    # --- Regime ------------------------------------------------------------
    print("\n-- Brutto-Funding p.a. auf Nominal nach Regime (Spot > / < SMA200) (%) --")
    reg_rows = []
    for sym in SYMBOLS:
        _, spot = load(sym)
        r = regime(spot)
        d = store[sym]
        idx = d.index.intersection(r.dropna().index)
        rr = r.reindex(idx)
        dd = d.reindex(idx)
        reg_rows.append({
            "symbol": sym,
            "Bull_ann_%": annualise(dd[rr == "Bull"]) * 100,
            "Bull_tage": int((rr == "Bull").sum()),
            "Bear_ann_%": annualise(dd[rr == "Bear"]) * 100,
            "Bear_tage": int((rr == "Bear").sum()),
        })
    print(pd.DataFrame(reg_rows).to_string(index=False, float_format=lambda x: f"{x:8.2f}"))

    # --- Verteilung rollierender 60-Tage-Renditen -------------------------
    print("\n-- Verteilung rollierender 60-Tage-Brutto-Renditen auf Nominal (%) --")
    dist_rows = []
    for sym, d in store.items():
        r60 = d.rolling(60).sum().dropna() * 100
        dist_rows.append({"symbol": sym, "min": r60.min(), "p10": r60.quantile(.10),
                          "median": r60.median(), "p90": r60.quantile(.90),
                          "p99": r60.quantile(.99), "max": r60.max(),
                          "anteil_negativ_%": (r60 < 0).mean() * 100})
    print(pd.DataFrame(dist_rows).to_string(index=False, float_format=lambda x: f"{x:8.2f}"))

    # --- Negative Phasen ---------------------------------------------------
    print("\n" + "=" * 100)
    print("NEGATIVE FUNDING-PHASEN")
    print("=" * 100)
    for sym, d in store.items():
        st = negative_streaks(d)
        wd = worst_drawdown_window(d)
        n_days_neg = (d < 0).mean() * 100
        print(f"\n{sym}: {n_days_neg:.1f}% aller Tage negativ; "
              f"laengste Negativ-Serie: "
              f"{int(st['tage'].max()) if len(st) else 0} Tage")
        if len(st):
            print("  5 tiefste zusammenhaengende Negativ-Serien:")
            print(st.head(5).to_string(index=False, float_format=lambda x: f"{x:8.3f}"))
        if wd:
            print(f"  Tiefster Funding-Drawdown: {wd['tiefe_pct']:.2f}% des Nominals, "
                  f"{wd['von']} bis {wd['bis']} ({wd['dauer_tage']} Tage)")

    # --- Teil 2: Liquidation ----------------------------------------------
    print("\n" + "=" * 100)
    print("TEIL 2 - LIQUIDATIONSRISIKO DES SHORT-BEINS")
    print("=" * 100)
    print("Ohne Nachschuss wird ein Short mit Margin-Quote m bei ca. +(m - MMR) liquidiert.")
    print(f"MMR-Annahme: {MMR*100:.1f}%  |  Zahlen sind echte High-basierte Kursanstiege.\n")
    for sym in ("BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT"):
        _, spot = load(sym)
        ru = max_runup(spot)
        print(f"{sym}:")
        for w, v in ru.items():
            print(f"  Fenster {w:4d} Tage: max Anstieg {v['max_runup_pct']:8.1f}% "
                  f"({v['datum']}) -> noetige Margin {required_margin(v['max_runup_pct'])*100:7.1f}% "
                  f"=> C={1+required_margin(v['max_runup_pct']):5.2f} "
                  f"| p99 {v['p99_pct']:6.1f}%")
        print()

    print("-- Renditeverwaesserung durch Margin-Reserve (BTCUSDT, Taker, 1J Haltedauer) --")
    gbtc = annualise(store["BTCUSDT"])
    _, btc_spot = load("BTCUSDT")
    ru_btc = max_runup(btc_spot)
    scenarios = [("m=0.20 (5x), taegliches Nachschiessen noetig", 0.20),
                 ("m=Puffer schlimmste 7-Tage-Spitze", required_margin(ru_btc[7]["max_runup_pct"])),
                 ("m=Puffer schlimmste 30-Tage-Spitze", required_margin(ru_btc[30]["max_runup_pct"])),
                 ("m=Puffer schlimmste 90-Tage-Spitze", required_margin(ru_btc[90]["max_runup_pct"])),
                 ("m=1.00 (Margin = Nominal, ueberlebt ca. +100%)", 1.00)]
    for label, m in scenarios:
        c = 1 + m
        print(f"  {label:42s} C={c:5.2f}  netto p.a. = "
              f"{net_on_capital(gbtc, 365, FEE_TAKER_BPS, c)*100:6.2f}%")

    # --- Liquidationshaeufigkeit bei realistischem Management -------------
    print("\n-- Wie oft waere ein Short mit Margin-Quote m liquidiert worden? --")
    print("   Annahme: Alle T Tage werden Spot-Gewinne realisiert und Margin "
          "auf m zurueckgesetzt.\n   Liquidation, wenn der Kurs innerhalb eines "
          "T-Tage-Fensters um mehr als (m - MMR) steigt.")
    for sym in ("BTCUSDT", "ETHUSDT", "SOLUSDT"):
        _, sp = load(sym)
        df = sp.set_index("ts")
        print(f"\n  {sym}:")
        for m in (0.10, 0.20, 0.35, 0.50):
            thr = m - MMR
            line = []
            for T in (1, 7, 30):
                fwd = df["high"].rolling(T, min_periods=1).max().shift(-(T - 1))
                ru = (fwd / df["close"].shift(1) - 1).dropna()
                # Zahl der disjunkten T-Tage-Perioden, in denen es gerissen haette
                hits = (ru >= thr)
                n_per = max(1, len(ru) // T)
                line.append(f"T={T:2d}d: {hits.mean()*100:5.1f}% der Fenster")
            print(f"    m={m*100:4.0f}% (Liq. bei +{thr*100:4.1f}%): " + " | ".join(line))

    # --- Teil 3: Verdoppelung ---------------------------------------------
    print("\n" + "=" * 100)
    print("TEIL 3 - VERDOPPELUNG IN 2 WOCHEN BIS 2 MONATEN")
    print("=" * 100)
    for horizon in (14, 30, 60):
        for cm in (1.20, 1.50):
            need_nom = 1.0 * cm  # +100% auf Kapital => need_nom auf Nominal
            per_day = need_nom / horizon
            per_8h = per_day / 3
            ann_simple = per_day * 365
            print(f"  Verdoppelung in {horizon:3d} Tagen bei C={cm:.2f}: "
                  f"{need_nom*100:5.0f}% auf Nominal = {per_day*100:5.2f}%/Tag "
                  f"= {per_8h*100:5.3f}% je 8h-Zahlung = {ann_simple*100:6.0f}% p.a. (einfach)")
    print("\n  Binance-Hardcap der 8h-Funding-Rate fuer BTCUSDT: +/-0.75% "
          "(fuer viele Alts +/-2.00%).")

    print("\n-- Bestes rollierendes Fenster der Historie (brutto auf Nominal) --")
    for horizon in (14, 30, 60):
        rows = []
        for sym, d in store.items():
            r = d.rolling(horizon).sum().dropna()
            best = r.max()
            rows.append({"symbol": sym, f"best_{horizon}d_%": best * 100,
                         "ende": r.idxmax().date(),
                         "auf_Kapital_C1.20_%": best / 1.20 * 100,
                         "tage_ueber_100%_Kapital": int((r / 1.20 >= 1.0).sum())})
        print(f"\n  Horizont {horizon} Tage:")
        print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:8.2f}"))

    print("\n-- Hoechste jemals erreichte annualisierte Funding-Raten (rollierend 7 Tage) --")
    for sym, d in store.items():
        r7 = d.rolling(7).sum().dropna() / 7 * 365
        top = r7.nlargest(1)
        share_above = {lvl: (r7 >= lvl).mean() * 100 for lvl in (0.5, 1.0, 3.0, 6.0)}
        print(f"  {sym:9s} max {top.iloc[0]*100:8.0f}% p.a. ({top.index[0].date()}) | "
              f"Anteil Tage >50%: {share_above[0.5]:5.2f}%  >100%: {share_above[1.0]:5.2f}%  "
              f">300%: {share_above[3.0]:5.2f}%  >600%: {share_above[6.0]:5.2f}%")

    # Laengste Phase, in der 60-Tage-Verdoppelung tatsaechlich gelaufen waere
    print("\n-- Gab es 60-Tage-Fenster mit >=100% Rendite auf Kapital (C=1.20)? --")
    for sym, d in store.items():
        r60 = d.rolling(60).sum().dropna() / 1.20
        hit = r60 >= 1.0
        print(f"  {sym:9s} {int(hit.sum()):4d} von {len(r60)} Fenstern "
              f"({hit.mean()*100:.2f}%)  | bestes Fenster: {r60.max()*100:.1f}% auf Kapital")


if __name__ == "__main__":
    report()
