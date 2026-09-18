"""TEIL 1 (Erweiterung) - Der Survivorship-Bias direkt gemessen.

Entscheidende Beobachtung: data-api.binance.vision liefert Klines AUCH fuer
Symbole mit Status BREAK, also fuer bereits delistete Paare. Damit laesst sich
die Verzerrung nicht nur schaetzen, sondern beziffern:

  * Population 1 (was man heute sieht): alle USDT-Paare mit Status TRADING.
  * Population 2 (was verschwunden ist): alle USDT-Paare mit Status BREAK.
  * Vollstaendige Population: beide zusammen.

Fuer jedes Symbol: Rendite vom ersten bis zum LETZTEN verfuegbaren Bar
("kaufen beim Listing, halten bis Delisting bzw. heute").

Caveats, die die Zahlen verzerren und die man kennen muss:
  * Rebrands/Migrationen (MATIC->POL, FTM->S, VEN->VET) sehen wie ein
    Totalverlust aus, obwohl Halter getauscht wurden.
  * Redenominierungen (z.B. COCOS 1:100) sehen wie ein Mega-Gewinn aus.
  * Stablecoins und Leveraged Tokens (UP/DOWN/BULL/BEAR) sind keine
    Investments im gemeinten Sinn und werden separat ausgewiesen.
  Der MEDIAN ist gegen alle drei Effekte weitgehend robust, der Mittelwert
  nicht - was genau der Punkt dieser Untersuchung ist.
"""
from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.research.memecoin_data import fetch_daily, fetch_exchange_info  # noqa: E402

OUT_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "memecoin"

STABLES = {
    "USDC", "TUSD", "BUSD", "PAX", "USDP", "DAI", "FDUSD", "USDS", "USDSB",
    "EUR", "GBP", "AEUR", "USD1", "XUSD", "SUSD", "USDE", "USDT", "USD1",
    "PYUSD", "RLUSD", "USDSOLD",
}
LEV_SUFFIX = ("UP", "DOWN", "BULL", "BEAR")


def classify(base: str) -> str:
    if base in STABLES:
        return "stablecoin"
    if any(base.endswith(s) for s in LEV_SUFFIX) and len(base) > 4:
        return "leveraged"
    return "coin"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    info = fetch_exchange_info()
    usdt = [s for s in info["symbols"] if s["quoteAsset"] == "USDT"]
    print(f"USDT-Paare in exchangeInfo: {len(usdt)}  (Klines werden fuer alle abgerufen)")

    def one(s: dict) -> dict:
        sym, base, status = s["symbol"], s["baseAsset"], s["status"]
        meta = {"symbol": sym, "base": base, "status": status, "kind": classify(base)}
        df = fetch_daily(sym)
        if df is None or len(df) < 5:
            return {**meta, "bars": 0}
        c = df["close"].astype(float)
        h = df["high"].astype(float)
        return {
            **meta,
            "bars": len(df),
            "first_bar": df["timestamp"].iloc[0].date(),
            "last_bar": df["timestamp"].iloc[-1].date(),
            "ret_listing_to_end": float(c.iloc[-1] / c.iloc[0] - 1.0),
            "dd_from_ath_high": float(c.iloc[-1] / h.max() - 1.0),
        }

    rows = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i, row in enumerate(pool.map(one, usdt), 1):
            rows.append(row)
            if i % 100 == 0:
                print(f"  ... {i}/{len(usdt)}", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(OUT_DIR / "all_usdt_pairs.csv", index=False)
    ok = res[(res["bars"] >= 5) & (res["kind"] == "coin")].copy()

    print()
    print("=" * 78)
    print("SURVIVORSHIP DIREKT GEMESSEN - alle USDT-Paare, Coins ohne Stables/Lev")
    print("=" * 78)
    print(f"auswertbare Paare: {len(ok)}   "
          f"(Stablecoins {int((res['kind'] == 'stablecoin').sum())}, "
          f"Leveraged Tokens {int((res['kind'] == 'leveraged').sum())}, "
          f"ohne Daten {int((res['bars'] < 5).sum())})")

    for label, sub in (
        ("HEUTE SICHTBAR (TRADING)", ok[ok["status"] == "TRADING"]),
        ("VERSCHWUNDEN (BREAK)", ok[ok["status"] != "TRADING"]),
        ("VOLLSTAENDIGE POPULATION", ok),
    ):
        r = sub["ret_listing_to_end"]
        d = sub["dd_from_ath_high"]
        print(f"\n[{label}]  n = {len(sub)}")
        print(f"  MITTELWERT Rendite: {r.mean():+.1%}")
        print(f"  MEDIAN Rendite:     {r.median():+.1%}")
        print(f"  Anteil im Minus:    {(r < 0).mean():.1%}")
        print(f"  Anteil <-50%:       {(r < -0.5).mean():.1%}")
        print(f"  Anteil <-90%:       {(r < -0.9).mean():.1%}")
        print(f"  >80% unter ATH:     {(d <= -0.8).mean():.1%}")
        print(f"  >95% unter ATH:     {(d <= -0.95).mean():.1%}")
        print(f"  Perzentile 10/25/50/75/90: " + " / ".join(
            f"{np.percentile(r, q):+.0%}" for q in (10, 25, 50, 75, 90)))

    surv = ok[ok["status"] == "TRADING"]["ret_listing_to_end"]
    allp = ok["ret_listing_to_end"]
    print()
    print("=" * 78)
    print("GROESSE DES BIAS")
    print("=" * 78)
    print(f"Median, nur Survivors:          {surv.median():+.1%}")
    print(f"Median, volle Population:       {allp.median():+.1%}")
    print(f"Differenz (Bias im Median):     {(surv.median() - allp.median()) * 100:+.1f} Prozentpunkte")
    print(f"Mittelwert, nur Survivors:      {surv.mean():+.1%}")
    print(f"Mittelwert, volle Population:   {allp.mean():+.1%}")
    print(f"Differenz (Bias im Mittelwert): {(surv.mean() - allp.mean()) * 100:+.1f} Prozentpunkte")
    print("\nUnd das ist NUR die Binance-Lucke. Coins, die nie auf Binance")
    print("gelistet wurden, fehlen auch hier komplett.")


if __name__ == "__main__":
    main()
