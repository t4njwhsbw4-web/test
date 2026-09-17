"""TEIL 2 - Cross-sektionale Momentum-/Rotations-Strategie auf Memecoins.

These des Nutzers: "in den Pump reinkommen". Umsetzung: jeden Rebalance-Tag
die k Memecoins mit der hoechsten Rendite der letzten L Tage gleichgewichtet
halten.

Sauberkeit:
  * Signal fuer Bar t nutzt ausschliesslich Daten bis Close t.
  * Ausfuehrung um einen Bar verzoegert: Position wird zum Close t+1 gehalten
    und verdient die Rendite Close t+1 -> Close t+2.
  * Universum zu jedem Zeitpunkt nur Coins, die DAMALS schon gelistet waren
    (kein Vorgriff auf spaetere Listings).
  * Kosten 10 bps pro Seite auf den Turnover.
  * Chronologischer Split 70/30, Hold-out wird GENAU EINMAL bewertet.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.research.memecoin_data import MEMECOIN_CANDIDATES, fetch_daily  # noqa: E402

COST_BPS = 10.0
ANN = 365.0
OUT_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "memecoin"


def load_panels() -> tuple[pd.DataFrame, pd.DataFrame]:
    closes, qvols = {}, {}
    for coin in MEMECOIN_CANDIDATES:
        sym = f"{coin}USDT"
        df = fetch_daily(sym)
        if df is None or len(df) < 120:
            continue
        s = df.set_index("timestamp")
        closes[coin] = s["close"]
        qvols[coin] = s["quote_volume"]
    close = pd.DataFrame(closes).sort_index()
    qvol = pd.DataFrame(qvols).sort_index()
    close.index = close.index.tz_convert("UTC").normalize()
    qvol.index = qvol.index.tz_convert("UTC").normalize()
    return close, qvol


def sharpe(r: pd.Series) -> float:
    r = r.dropna()
    if len(r) < 20 or r.std() == 0:
        return float("nan")
    return float(r.mean() / r.std() * np.sqrt(ANN))


def max_dd(equity: pd.Series) -> float:
    return float((equity / equity.cummax() - 1.0).min())


def backtest(
    close: pd.DataFrame,
    qvol: pd.DataFrame,
    lookback: int,
    top_k: int,
    rebal: int,
    min_qvol: float = 1e6,
) -> pd.Series:
    """Tagesrenditen der Strategie (nach Kosten), indexiert auf den Ertragstag."""
    rets = close.pct_change()
    mom = close / close.shift(lookback) - 1.0
    liq = qvol.rolling(30, min_periods=10).median()

    dates = close.index
    weights = pd.DataFrame(0.0, index=dates, columns=close.columns)
    current = pd.Series(0.0, index=close.columns)

    for i, t in enumerate(dates):
        if i % rebal == 0:
            m = mom.loc[t]
            # nur Coins mit ausreichend Historie, echtem Preis und Liquiditaet
            valid = m.notna() & close.loc[t].notna() & (liq.loc[t] >= min_qvol)
            cand = m[valid].sort_values(ascending=False)
            picks = list(cand.index[:top_k])
            new = pd.Series(0.0, index=close.columns)
            if picks:
                new[picks] = 1.0 / len(picks)
            current = new
        weights.loc[t] = current

    # Signal an t -> Gewicht ab Close t+1 -> Ertrag von t+1 nach t+2
    held = weights.shift(2)
    gross = (held * rets).sum(axis=1, min_count=1)
    turnover = held.diff().abs().sum(axis=1)
    cost = turnover * COST_BPS / 1e4
    net = gross - cost
    # Tage ohne Position (kein Coin qualifiziert) = 0 Rendite, nicht NaN
    return net.fillna(0.0).loc[held.dropna(how="all").index]


def rolling_window_stats(daily_ret: pd.Series, window: int = 60) -> dict:
    g = (1.0 + daily_ret.fillna(0.0)).rolling(window).apply(np.prod, raw=True)
    g = g.dropna()
    if len(g) == 0:
        return {}
    return {
        "n_windows": len(g),
        "p_double": float((g >= 2.0).mean()),
        "p_halve": float((g <= 0.5).mean()),
        "p_up": float((g > 1.0).mean()),
        "median": float(g.median()),
        "mean": float(g.mean()),
        "p05": float(np.percentile(g, 5)),
        "p95": float(np.percentile(g, 95)),
        "worst": float(g.min()),
        "best": float(g.max()),
    }


def print_window_stats(label: str, st: dict) -> None:
    if not st:
        print(f"{label:<34} (keine Fenster)")
        return
    print(
        f"{label:<34} n={st['n_windows']:>5}  verdoppelt {st['p_double']:6.1%}  "
        f"halbiert {st['p_halve']:6.1%}  >1x {st['p_up']:6.1%}  "
        f"Median x{st['median']:.2f}  5%-Q x{st['p05']:.2f}  95%-Q x{st['p95']:.2f}  "
        f"schlimmstes x{st['worst']:.2f}"
    )


def run_universe(close: pd.DataFrame, qvol: pd.DataFrame, label: str, btc: pd.Series) -> None:
    print()
    print("=" * 78)
    print(f"TEIL 2 - MOMENTUM-ROTATION: {label}")
    print("=" * 78)
    print(f"Zeitraum: {close.index[0].date()} .. {close.index[-1].date()}  "
          f"({len(close)} Tage, {close.shape[1]} Coins im Universum)")

    n = len(close)
    split = int(n * 0.70)
    dev_end = close.index[split - 1]
    print(f"Dev (70%):     {close.index[0].date()} .. {dev_end.date()}  ({split} Tage)")
    print(f"Hold-out (30%):{close.index[split].date()} .. {close.index[-1].date()}  "
          f"({n - split} Tage)")

    grid = [
        (lb, k, rb)
        for lb in (7, 14, 30, 60, 90)
        for k in (1, 2, 3, 5)
        for rb in (1, 3, 7, 14)
    ]
    rows = []
    cache: dict[tuple, pd.Series] = {}
    for lb, k, rb in grid:
        r = backtest(close, qvol, lb, k, rb)
        cache[(lb, k, rb)] = r
        dev = r.loc[:dev_end]
        rows.append({
            "lookback": lb, "top_k": k, "rebal": rb,
            "dev_sharpe": sharpe(dev),
            "dev_cagr": float((1 + dev).prod() ** (ANN / max(len(dev), 1)) - 1),
            "dev_maxdd": max_dd((1 + dev).cumprod()),
        })
    res = pd.DataFrame(rows).sort_values("dev_sharpe", ascending=False)
    print(f"\nGrid: {len(grid)} Kombinationen. Top 8 nach Dev-Sharpe:")
    print(res.head(8).to_string(index=False, float_format=lambda x: f"{x:8.3f}"))
    print("\nSchlechteste 3 (zur Einordnung der Streuung):")
    print(res.tail(3).to_string(index=False, float_format=lambda x: f"{x:8.3f}"))

    best = res.iloc[0]
    key = (int(best["lookback"]), int(best["top_k"]), int(best["rebal"]))
    r = cache[key]
    dev, hold = r.loc[:dev_end], r.loc[close.index[split]:]

    print(f"\nGewaehlt auf Dev (einmalige Hold-out-Bewertung): "
          f"lookback={key[0]}, top_k={key[1]}, rebal={key[2]}")
    print(f"  Dev      Sharpe {sharpe(dev):+.2f}  CAGR {(1+dev).prod()**(ANN/len(dev))-1:+.1%}  "
          f"MaxDD {max_dd((1+dev).cumprod()):.1%}  Gesamt x{(1+dev).prod():.2f}")
    print(f"  HOLD-OUT Sharpe {sharpe(hold):+.2f}  CAGR {(1+hold).prod()**(ANN/len(hold))-1:+.1%}  "
          f"MaxDD {max_dd((1+hold).cumprod()):.1%}  Gesamt x{(1+hold).prod():.2f}")

    # Robustheit: wie sieht das ganze Grid out-of-sample aus?
    hs = pd.Series({k: sharpe(v.loc[close.index[split]:]) for k, v in cache.items()})
    ds = pd.Series({k: sharpe(v.loc[:dev_end]) for k, v in cache.items()})
    print(f"\nAlle {len(grid)} Kombis out-of-sample: Median-Sharpe {hs.median():+.2f}, "
          f"Anteil Sharpe>0 {(hs > 0).mean():.1%}, bester {hs.max():+.2f}, schlechtester {hs.min():+.2f}")
    print(f"Rangkorrelation Dev-Sharpe vs. Hold-out-Sharpe (Spearman): "
          f"{ds.corr(hs, method='spearman'):+.2f}  "
          f"(0 = Dev-Ergebnis sagt nichts ueber die Zukunft)")

    # Benchmarks auf demselben Hold-out
    print("\nBenchmarks auf demselben Hold-out-Zeitraum:")
    ew = close.pct_change().mean(axis=1).loc[close.index[split]:]
    print(f"  Memecoin Equal-Weight B&H   Sharpe {sharpe(ew):+.2f}  Gesamt x{(1+ew).prod():.2f}")
    b = btc.pct_change().reindex(close.index).loc[close.index[split]:]
    print(f"  BTC Buy & Hold              Sharpe {sharpe(b):+.2f}  Gesamt x{(1+b).prod():.2f}")
    hold_slice = close.loc[close.index[split]:]
    per_coin = (hold_slice.iloc[-1] / hold_slice.iloc[0] - 1).dropna().sort_values()
    print(f"  Einzelne Coins B&H: Median {per_coin.median():+.1%}, "
          f"bester {per_coin.index[-1]} {per_coin.iloc[-1]:+.1%}, "
          f"schlechtester {per_coin.index[0]} {per_coin.iloc[0]:+.1%}")
    print(f"  Anteil Coins mit Gewinn im Hold-out: {(per_coin > 0).mean():.1%}")

    print("\nROLLIERENDE 60-TAGE-FENSTER (Verteilung, nicht Durchschnitt):")
    print_window_stats("Strategie (gesamter Zeitraum)", rolling_window_stats(r))
    print_window_stats("Strategie (nur Dev)", rolling_window_stats(dev))
    print_window_stats("Strategie (nur Hold-out)", rolling_window_stats(hold))
    print_window_stats("Memecoin Equal-Weight B&H", rolling_window_stats(close.pct_change().mean(axis=1)))
    print_window_stats("BTC Buy & Hold", rolling_window_stats(btc.pct_change().reindex(close.index)))
    # Einzelcoin-Lotterie: alle Coins gepoolt
    pooled = []
    for c in close.columns:
        s = close[c].dropna().pct_change()
        g = (1 + s.fillna(0)).rolling(60).apply(np.prod, raw=True).dropna()
        pooled.append(g)
        if len(g) >= 60:
            st = rolling_window_stats(s)
            print_window_stats(f"  B&H {c}", st)
    allg = pd.concat(pooled)
    print(f"\nAlle Einzelcoin-60-Tage-Fenster gepoolt (n={len(allg)}): "
          f"verdoppelt {(allg >= 2).mean():.1%}, halbiert {(allg <= 0.5).mean():.1%}, "
          f"Median x{allg.median():.2f}, Mittelwert x{allg.mean():.2f}")

    res.to_csv(OUT_DIR / f"momentum_grid_{label.split()[0].lower()}.csv", index=False)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    close, qvol = load_panels()
    btc = fetch_daily("BTCUSDT").set_index("timestamp")["close"]
    btc.index = btc.index.tz_convert("UTC").normalize()

    print(f"Memecoins mit >=120 Tagesbars: {close.shape[1]} -> {list(close.columns)}")

    # Universum A: ab dem Tag, an dem >=5 Coins 90 Tage Historie haben.
    # Vorab festgelegt, nicht nach Sichtung der Ergebnisse gewaehlt.
    have_hist = close.notna().rolling(90, min_periods=90).count().ge(90).sum(axis=1)
    startA = have_hist[have_hist >= 5].index[0]
    run_universe(close.loc[startA:], qvol.loc[startA:], "A ab >=5 Coins mit 90d Historie", btc)

    # Universum B: volle Historie (dominiert von DOGE/SHIB, wenige Coins frueh)
    run_universe(close, qvol, "B volle Historie", btc)


if __name__ == "__main__":
    main()
