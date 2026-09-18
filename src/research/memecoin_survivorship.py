"""TEIL 1 - Survivorship-Bias bei Memecoins quantifizieren.

Fragen:
  a) Wie viele USDT-Paare auf Binance sind heute nicht mehr handelbar?
  b) Ab wann ist jeder Memecoin handelbar und was ist seit dem ERSTEN
     verfuegbaren Bar passiert (kein selbstgewaehlter Startpunkt)?
  c) Naive Strategie "kaufe jeden neu gelisteten Memecoin beim Listing und
     halte": MEDIAN vs. MITTELWERT.
  d) Anteil der Coins mit >80% / >95% Verlust vom Allzeithoch.

Einschraenkung (explizit): Coins, die es nie auf Binance geschafft haben oder
dort delistet wurden, fehlen komplett. Die echten Zahlen sind schlechter.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.research.memecoin_data import (  # noqa: E402
    MEMECOIN_CANDIDATES,
    fetch_daily,
    fetch_exchange_info,
)

OUT_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "memecoin"


def exchange_status_table() -> pd.DataFrame:
    info = fetch_exchange_info()
    rows = [
        {
            "symbol": s["symbol"],
            "base": s["baseAsset"],
            "quote": s["quoteAsset"],
            "status": s["status"],
        }
        for s in info["symbols"]
    ]
    return pd.DataFrame(rows)


def coin_life_stats(symbols: list[str]) -> pd.DataFrame:
    rows = []
    for sym in symbols:
        df = fetch_daily(sym)
        if df is None or len(df) < 5:
            continue
        df = df.set_index("timestamp")
        first_close = float(df["close"].iloc[0])
        last_close = float(df["close"].iloc[-1])
        # Einstieg zum SCHLUSSKURS des ersten verfuegbaren Tages. Listing-Tage
        # laufen typisch nach oben, ein Kauf zum Close ist also die
        # pessimistischere und realistischere Annahme gegenueber dem Open.
        ath = float(df["close"].cummax().iloc[-1])
        ath_intraday = float(df["high"].max())
        ath_date = df["close"].idxmax()
        rows.append(
            {
                "symbol": sym,
                "first_bar": df.index[0].date(),
                "last_bar": df.index[-1].date(),
                "days": len(df),
                "first_close": first_close,
                "last_close": last_close,
                "ret_since_listing": last_close / first_close - 1.0,
                "ath_close": ath,
                "ath_date": ath_date.date(),
                "dd_from_ath_close": last_close / ath - 1.0,
                "dd_from_ath_high": last_close / ath_intraday - 1.0,
                "median_quote_vol_30d": float(df["quote_volume"].tail(30).median()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tbl = exchange_status_table()
    usdt = tbl[tbl["quote"] == "USDT"]

    print("=" * 78)
    print("TEIL 1a - EXCHANGE-STATUS (Binance Spot, exchangeInfo)")
    print("=" * 78)
    print(f"Symbole gesamt:            {len(tbl):,}")
    print(f"  Status-Verteilung:       {dict(Counter(tbl['status']))}")
    print(f"USDT-Paare gesamt:         {len(usdt):,}")
    cnt = Counter(usdt["status"])
    print(f"  TRADING:                 {cnt.get('TRADING', 0):,}")
    not_trading = len(usdt) - cnt.get("TRADING", 0)
    print(f"  nicht TRADING (BREAK/HALT): {not_trading:,} "
          f"= {not_trading / len(usdt):.1%} aller je gelisteten USDT-Paare")
    print("  Beispiele nicht handelbar:",
          ", ".join(usdt[usdt["status"] != "TRADING"]["symbol"].head(15)))

    # Leveraged Tokens (UP/DOWN/BULL/BEAR) sind ein Sonderfall - separat zeigen
    lev = usdt["base"].str.contains("UP$|DOWN$|BULL$|BEAR$", regex=True)
    print(f"  davon Leveraged Tokens:  {int((lev & (usdt['status'] != 'TRADING')).sum())}")
    core = usdt[~lev]
    core_nt = int((core["status"] != "TRADING").sum())
    print(f"  ohne Leveraged Tokens:   {core_nt}/{len(core)} = "
          f"{core_nt / len(core):.1%} nicht mehr handelbar")

    print()
    print("=" * 78)
    print("TEIL 1b - MEMECOIN-LEBENSLAEUFE (ab erstem verfuegbaren Bar)")
    print("=" * 78)
    listed = set(usdt["symbol"])
    cand_syms = [f"{c}USDT" for c in MEMECOIN_CANDIDATES]
    on_binance = [s for s in cand_syms if s in listed]
    missing = [s for s in cand_syms if s not in listed]
    print(f"Kandidaten im Universum:   {len(cand_syms)}")
    print(f"  auf Binance USDT:        {len(on_binance)}")
    print(f"  NICHT auf Binance:       {len(missing)}  -> {', '.join(m[:-4] for m in missing)}")

    stats = coin_life_stats(on_binance)
    stats = stats.sort_values("first_bar").reset_index(drop=True)
    with pd.option_context("display.width", 200, "display.max_rows", 100):
        print()
        print(stats[[
            "symbol", "first_bar", "days", "ret_since_listing",
            "dd_from_ath_close", "dd_from_ath_high", "ath_date",
        ]].to_string(index=False, float_format=lambda x: f"{x:9.3f}"))

    r = stats["ret_since_listing"]
    print()
    print("=" * 78)
    print("TEIL 1c - 'KAUFE BEI LISTING, HALTE' (n = %d Coins)" % len(r))
    print("=" * 78)
    print(f"MITTELWERT Rendite:        {r.mean():+.1%}   (Endkapital x{1 + r.mean():.2f})")
    print(f"MEDIAN Rendite:            {r.median():+.1%}   (Endkapital x{1 + r.median():.2f})")
    print(f"Mittelwert / Median-Luecke: Faktor {(1 + r.mean()) / (1 + r.median()):.1f}")
    for q in (5, 10, 25, 50, 75, 90, 95):
        print(f"  {q:>2}. Perzentil:           {np.percentile(r, q):+.1%}")
    print(f"Anteil mit Verlust:        {(r < 0).mean():.1%}  ({int((r < 0).sum())}/{len(r)})")
    print(f"Anteil verdoppelt (>=+100%): {(r >= 1.0).mean():.1%}  ({int((r >= 1.0).sum())}/{len(r)})")
    print(f"Anteil >-50%:              {(r < -0.5).mean():.1%}")
    print(f"Anteil >-90%:              {(r < -0.9).mean():.1%}")
    print(f"Bester:  {stats.loc[r.idxmax(), 'symbol']} {r.max():+.1%}")
    print(f"Schlechtester: {stats.loc[r.idxmin(), 'symbol']} {r.min():+.1%}")
    # Gleichgewichtetes Portfolio aller Listings (= Mittelwert) vs. typischer Coin
    print(f"\nGleichgewichtetes Portfolio ueber alle {len(r)} Coins: {r.mean():+.1%}")
    print("  -> das ist der Mittelwert; er entsteht aus wenigen Ausreissern.")
    top = r.sort_values(ascending=False)
    print(f"  ohne den besten Coin:    {top.iloc[1:].mean():+.1%}")
    print(f"  ohne die besten drei:    {top.iloc[3:].mean():+.1%}")

    print()
    print("=" * 78)
    print("TEIL 1d - DRAWDOWN VOM ALLZEITHOCH (Tages-Close bzw. Tages-High)")
    print("=" * 78)
    for col, label in (("dd_from_ath_close", "Close-ATH"), ("dd_from_ath_high", "High-ATH")):
        d = stats[col]
        print(f"[{label}]  Median-Drawdown: {d.median():.1%}")
        print(f"   >80% unter dem Hoch:  {(d <= -0.8).mean():.1%}  ({int((d <= -0.8).sum())}/{len(d)})")
        print(f"   >90% unter dem Hoch:  {(d <= -0.9).mean():.1%}  ({int((d <= -0.9).sum())}/{len(d)})")
        print(f"   >95% unter dem Hoch:  {(d <= -0.95).mean():.1%}  ({int((d <= -0.95).sum())}/{len(d)})")

    # Wie viele Coins sind heute noch "lebendig" im Sinne von Handelbarkeit + Volumen
    dead_vol = stats["median_quote_vol_30d"] < 100_000
    print(f"\nCoins mit <100k USDT Median-Tagesumsatz (letzte 30 Tage): "
          f"{int(dead_vol.sum())}/{len(stats)}")

    stats.to_csv(OUT_DIR / "survivorship_coins.csv", index=False)
    usdt.to_csv(OUT_DIR / "usdt_symbol_status.csv", index=False)
    print(f"\nGeschrieben: {OUT_DIR}/survivorship_coins.csv")


if __name__ == "__main__":
    main()
