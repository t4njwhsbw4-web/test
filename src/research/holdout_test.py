"""Einmaliger Hold-out-Test der vorregistrierten Kandidaten.

DISZIPLIN: Die Kandidatenliste unten wurde festgelegt, BEVOR irgendwelche
Hold-out-Daten berechnet wurden - sie stammt direkt aus den Empfehlungen der
Dev-Phase. Dieses Skript wird EINMAL ausgeführt. Wer nach Sicht der
Ergebnisse Parameter nachjustiert und erneut laufen lässt, hat keinen
Hold-out mehr, sondern einen zweiten Trainingsdatensatz - und das Ergebnis
ist dann nichts mehr wert.

Warmup: Signale werden auf der VOLLEN Historie berechnet, aber nur die
Renditen ab dem Hold-out-Cutoff ausgewertet. Das ist kein Leakage - ein
Signal zu Zeitpunkt t nutzt ausschliesslich Preise vor t, genau wie im
Live-Betrieb, wo die Vergangenheit ja auch bekannt ist.
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import pandas as pd

from src.research.harness import RESEARCH_SYMBOLS, evaluate, holdout_cutoff, load_data
from src.strategies import cross_sectional_momentum as xsec
from src.strategies import trend_vol_target as tvt

# Vorregistrierte Kandidaten aus der Dev-Phase --------------------------------

XSEC_CANDIDATES = [
    # Die einzige risikoadjustiert verteidigbare Variante (Dev-MDD -58%).
    {"lookback": 30, "top_k": 3, "rebalance_days": 30, "absolute_filter": True, "vol_normalize": False},
    # Höchster Dev-Sharpe, aber Dev-MDD -93% - praktisch nicht handelbar.
    # Mitgetestet, um zu sehen, ob der Sharpe überhaupt out-of-sample hält.
    {"lookback": 90, "top_k": 1, "rebalance_days": 3, "absolute_filter": False, "vol_normalize": False},
    {"lookback": 90, "top_k": 1, "rebalance_days": 3, "absolute_filter": True, "vol_normalize": False},
]

TVT_CANDIDATE = {
    "trend_mode": "donchian", "donchian_entry": 55, "donchian_exit": 20,
    "vol_mode": "target", "target_vol": 0.30, "vol_window": 60,
}


def full_panel() -> pd.DataFrame:
    return pd.DataFrame({s: load_data(s, period="full")["close"] for s in RESEARCH_SYMBOLS}).sort_index()


def main() -> None:
    cutoff = holdout_cutoff()
    panel = full_panel()
    holdout_bars = (panel.index >= cutoff).sum()
    print(f"Hold-out: {cutoff.date()} .. {panel.index.max().date()} ({holdout_bars} Bars)")
    print(f"Signal-Warmup auf voller Historie ab {panel.index.min().date()}\n")

    print("=== Benchmarks im Hold-out ===")
    for label, bench in (
        ("BTC Buy-and-Hold", xsec.benchmark_buy_and_hold(panel, "BTC/USD", warmup_from=cutoff)),
        ("Equal-Weight (alle 6)", xsec.benchmark_equal_weight(panel, warmup_from=cutoff)),
    ):
        s = bench.summary()
        print(f"  {label:24s} Sharpe {s['sharpe']:>6.3f}  CAGR {s['cagr']:>7.2%}  MDD {s['max_drawdown']:>7.2%}")

    print("\n=== Cross-Sectional Momentum ===")
    for params in XSEC_CANDIDATES:
        weights = xsec.generate_weights(panel, **params)
        res = xsec.backtest_portfolio(panel, weights, label="xsec", params=params, warmup_from=cutoff)
        s = res.summary()
        tag = f"lb{params['lookback']}/k{params['top_k']}/rb{params['rebalance_days']}" \
              f"{'/abs' if params['absolute_filter'] else ''}"
        print(f"  {tag:24s} Sharpe {s['sharpe']:>6.3f}  CAGR {s['cagr']:>7.2%}  "
              f"MDD {s['max_drawdown']:>7.2%}  Rebal {s['n_rebalances']:>3d}  top3 {s['top3_day_conc']:.2f}")

    print("\n=== Trendfolge + Vol-Targeting (pro Symbol) ===")
    sharpes, mdds = [], []
    for symbol in RESEARCH_SYMBOLS:
        df = load_data(symbol, period="full")
        signals = tvt.generate_signals(df, **TVT_CANDIDATE)
        mask = df.index >= cutoff
        res = evaluate(signals[mask], df["close"][mask], symbol, TVT_CANDIDATE)
        bh = evaluate(pd.Series(1, index=df.index[mask]), df["close"][mask], symbol, {"s": "bh"})
        sharpes.append(res.sharpe)
        mdds.append(res.max_drawdown)
        print(f"  {symbol:9s} Sharpe {res.sharpe:>6.3f} (B&H {bh.sharpe:>6.3f})  "
              f"MDD {res.max_drawdown:>7.2%} (B&H {bh.max_drawdown:>7.2%})  Trades {res.n_trades:>3d}")
    print(f"  {'MITTEL':9s} Sharpe {sum(sharpes)/len(sharpes):>6.3f}  MDD {sum(mdds)/len(mdds):>7.2%}")


if __name__ == "__main__":
    main()
