"""TEIL 3 - Liquiditaet und Ausfuehrbarkeit: was kostet eine Order wirklich?

Der Backtest nimmt Fills zum Schlusskurs ohne Slippage an. Hier wird mit dem
echten Orderbuch (depth, limit=100) gemessen, wie weit eine Marktorder von
500 / 5.000 / 50.000 USD den Preis gegen sich bewegt - fuer kleine Memecoins,
liquide Memecoins und BTC als Referenz.

Wichtig: limit=100 deckt nur die obersten 100 Levels ab. Reicht die Tiefe
nicht, ist die wahre Slippage NOCH groesser; das wird als "Buch erschoepft"
ausgewiesen und nicht stillschweigend abgeschnitten.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.research.memecoin_data import fetch_daily, fetch_depth  # noqa: E402

SIZES_USD = (500.0, 5_000.0, 50_000.0)
OUT_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "memecoin"

# Klein / mittel / liquide / Referenz - vor dem Messen festgelegt
SYMBOLS = (
    "BANANAUSDT", "ANIMEUSDT", "1000CATUSDT",   # kleine Memecoins
    "ACTUSDT", "TURBOUSDT",                      # mittlere
    "PEPEUSDT", "DOGEUSDT",                      # liquide Memecoins
    "BTCUSDT",                                   # Referenz
)


def walk_book(levels: list, notional: float, side: str) -> tuple[float, bool]:
    """Volumengewichteter Fill-Preis fuer `notional` USD; (preis, buch_erschoepft)."""
    spent = 0.0
    qty = 0.0
    for px_s, qty_s in levels:
        px, avail = float(px_s), float(qty_s)
        room = notional - spent
        take_notional = min(room, px * avail)
        qty += take_notional / px
        spent += take_notional
        if spent >= notional - 1e-9:
            return spent / qty, False
    return (spent / qty if qty else float("nan")), True


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for sym in SYMBOLS:
        book = fetch_depth(sym, limit=100)
        bids, asks = book["bids"], book["asks"]
        best_bid, best_ask = float(bids[0][0]), float(asks[0][0])
        mid = (best_bid + best_ask) / 2.0
        spread_bps = (best_ask - best_bid) / mid * 1e4
        ask_depth = sum(float(p) * float(q) for p, q in asks)
        bid_depth = sum(float(p) * float(q) for p, q in bids)
        df = fetch_daily(sym)
        adv = float(df.set_index("timestamp")["quote_volume"].tail(30).median()) if df is not None else float("nan")

        for size in SIZES_USD:
            buy_px, buy_out = walk_book(asks, size, "buy")
            sell_px, sell_out = walk_book(bids, size, "sell")
            buy_slip = (buy_px / mid - 1.0) * 1e4
            sell_slip = (1.0 - sell_px / mid) * 1e4
            rows.append({
                "symbol": sym,
                "size_usd": size,
                "spread_bps": spread_bps,
                "buy_slip_bps": buy_slip,
                "sell_slip_bps": sell_slip,
                "roundtrip_bps": buy_slip + sell_slip,
                "book_exhausted": buy_out or sell_out,
                "ask_depth_usd_100lvl": ask_depth,
                "bid_depth_usd_100lvl": bid_depth,
                "median_adv_usd_30d": adv,
                "size_pct_of_adv": size / adv * 100 if adv and adv == adv else float("nan"),
            })

    res = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print("=" * 100)
    print("TEIL 3 - SLIPPAGE AUS DEM ECHTEN ORDERBUCH (depth limit=100, Momentaufnahme)")
    print("=" * 100)
    print(res[[
        "symbol", "size_usd", "spread_bps", "buy_slip_bps", "sell_slip_bps",
        "roundtrip_bps", "book_exhausted", "size_pct_of_adv",
    ]].to_string(index=False, float_format=lambda x: f"{x:10.2f}"))

    print("\nBuch-Tiefe in den obersten 100 Levels (USD) und Median-Tagesumsatz:")
    d = res.drop_duplicates("symbol")[[
        "symbol", "ask_depth_usd_100lvl", "bid_depth_usd_100lvl", "median_adv_usd_30d",
    ]]
    print(d.to_string(index=False, float_format=lambda x: f"{x:15,.0f}"))

    print("\nVergleich mit der Backtest-Annahme (10 bps pro Seite = 20 bps Roundtrip):")
    for size in SIZES_USD:
        sub = res[res["size_usd"] == size]
        print(f"  Order {size:>9,.0f} USD: Roundtrip-Kosten inkl. Spread "
              f"Median {sub['roundtrip_bps'].median():7.1f} bps, "
              f"Max {sub['roundtrip_bps'].max():8.1f} bps "
              f"({sub['roundtrip_bps'].max() / 20:.1f}x der Annahme)")
    small = res[res["symbol"].isin(("BANANAUSDT", "ANIMEUSDT", "1000CATUSDT"))]
    for size in SIZES_USD:
        s = small[small["size_usd"] == size]
        print(f"  nur kleine Memecoins, {size:>9,.0f} USD: Roundtrip Median "
              f"{s['roundtrip_bps'].median():7.1f} bps = "
              f"{s['roundtrip_bps'].median() / 20:.1f}x der Annahme")

    res.to_csv(OUT_DIR / "slippage.csv", index=False)
    print(f"\nGeschrieben: {OUT_DIR}/slippage.csv")


if __name__ == "__main__":
    main()
