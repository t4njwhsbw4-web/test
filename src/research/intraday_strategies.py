"""Intraday-Tests (15m-Bars, Haltedauer ~2 Bars = 30 Minuten) auf BTCUSDT/ETHUSDT.

Drei oekonomisch begruendete Thesen:
  (a) momentum  - Fortsetzung kurzfristiger Bewegungen
  (b) reversion - Rueckkehr nach Ueberdehnung (z-Score)
  (c) breakout  - Ausbruch aus Volatilitaets-Kompression

WICHTIG - Annualisierung: src/backtest/engine.py nutzt 252 Handelstage, was
fuer Intraday-Bars in 24/7-Krypto falsch ist. Hier wird der Sharpe selbst mit
dem korrekten Faktor gerechnet: 15m -> 4*24*365 = 35040 Bars pro Jahr.

Kosten: 10 bps pro Seite (Binance Taker ~0.1%) -> 20 bps pro Round-Trip.
Long/flat, kein Hebel. Signal fuer Bar t nutzt nur Daten bis t und wird zu
t+1 ausgefuehrt (engine.run_backtest macht die shift(1)-Verzoegerung).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.backtest.engine import run_backtest  # noqa: E402
from src.data.intraday_fetch import load_panel  # noqa: E402

BARS_PER_YEAR = {"15m": 4 * 24 * 365, "30m": 2 * 24 * 365}
COST_BPS_PER_SIDE = 10.0
HOLD_BARS = 2  # 2 x 15m = 30 Minuten, die Idee des Nutzers
DEV_FRACTION = 0.70


# --------------------------------------------------------------------------
# Signale: alle Fenster enden bei t (kein Lookahead)
# --------------------------------------------------------------------------
def _hold(entries: pd.Series, hold_bars: int) -> pd.Series:
    """Position bleibt hold_bars Bars long, nachdem ein Entry ausgeloest wurde."""
    return entries.rolling(hold_bars, min_periods=1).max().fillna(0).astype(float)


def sig_momentum(close: pd.Series, lookback: int, hold_bars: int = HOLD_BARS) -> pd.Series:
    mom = close.pct_change(lookback)
    return _hold((mom > 0).astype(float), hold_bars)


def sig_reversion(
    close: pd.Series, window: int, z_thresh: float, hold_bars: int = HOLD_BARS
) -> pd.Series:
    ret = close.pct_change()
    z = (ret - ret.rolling(window).mean()) / ret.rolling(window).std()
    cum_z = z.rolling(3).sum()  # kurze Ueberdehnung ueber ~45 Minuten
    return _hold((cum_z < -z_thresh).astype(float), hold_bars)


def sig_breakout(
    close: pd.Series, vol_window: int, brk_window: int, q: float, hold_bars: int = HOLD_BARS
) -> pd.Series:
    ret = close.pct_change()
    vol = ret.rolling(vol_window).std()
    # Kompression: aktuelle Vol im unteren Quantil der eigenen Historie (expanding,
    # damit kein zukuenftiges Wissen einflieszt)
    vol_rank = vol.rolling(vol_window * 10, min_periods=vol_window * 2).rank(pct=True)
    compressed = vol_rank < q
    breakout = close > close.rolling(brk_window).max().shift(1)
    return _hold((compressed & breakout).astype(float), hold_bars)


# --------------------------------------------------------------------------
# Metriken mit korrekter Intraday-Annualisierung
# --------------------------------------------------------------------------
@dataclass
class Stats:
    sharpe_gross: float
    sharpe_net: float
    n_trades: int
    trades_per_year: float
    gross_bps_per_trade: float
    net_bps_per_trade: float
    time_in_market: float
    total_return_net: float
    max_dd_net: float


def _sharpe(returns: pd.Series, bars_per_year: int) -> float:
    sd = returns.std()
    if not np.isfinite(sd) or sd <= 0:
        return 0.0
    return float(returns.mean() / sd * np.sqrt(bars_per_year))


def _trade_bps(position: pd.Series, asset_ret: pd.Series) -> tuple[int, float]:
    """Mittlerer Brutto-Gewinn pro Trade in bps (zusammenhaengende Long-Phasen)."""
    in_pos = position > 0
    prev = in_pos.shift(1, fill_value=False)
    trade_id = (in_pos & ~prev).cumsum().where(in_pos)
    pnl = (position * asset_ret).groupby(trade_id).apply(lambda r: (1 + r).prod() - 1)
    if pnl.empty:
        return 0, 0.0
    return len(pnl), float(pnl.mean() * 10_000)


def evaluate(prices: pd.Series, signals: pd.Series, interval: str = "15m") -> Stats:
    bpy = BARS_PER_YEAR[interval]
    gross = run_backtest(prices, signals, transaction_cost_bps=0.0)
    net = run_backtest(prices, signals, transaction_cost_bps=COST_BPS_PER_SIDE)

    position = signals.reindex(prices.index).fillna(0).clip(0, 1).shift(1).fillna(0)
    asset_ret = prices.pct_change().fillna(0)
    n_trades, gross_bps = _trade_bps(position, asset_ret)

    years = len(prices) / bpy
    return Stats(
        sharpe_gross=_sharpe(gross.returns, bpy),
        sharpe_net=_sharpe(net.returns, bpy),
        n_trades=n_trades,
        trades_per_year=n_trades / years if years > 0 else 0.0,
        gross_bps_per_trade=gross_bps,
        net_bps_per_trade=gross_bps - 2 * COST_BPS_PER_SIDE,
        time_in_market=float(position.mean()),
        total_return_net=net.total_return,
        max_dd_net=net.max_drawdown,
    )


# --------------------------------------------------------------------------
# Kandidaten-Grid
# --------------------------------------------------------------------------
def candidates() -> list[tuple[str, dict]]:
    out: list[tuple[str, dict]] = []
    for lb in (1, 2, 4, 8, 16, 32):
        out.append(("momentum", {"lookback": lb}))
    for w in (20, 50, 100):
        for z in (1.5, 2.0, 2.5, 3.0):
            out.append(("reversion", {"window": w, "z_thresh": z}))
    for vw in (20, 50):
        for bw in (8, 16, 32):
            for q in (0.2, 0.35):
                out.append(("breakout", {"vol_window": vw, "brk_window": bw, "q": q}))
    return out


def build(kind: str, close: pd.Series, params: dict) -> pd.Series:
    if kind == "momentum":
        return sig_momentum(close, **params)
    if kind == "reversion":
        return sig_reversion(close, **params)
    if kind == "breakout":
        return sig_breakout(close, **params)
    raise ValueError(kind)


def split(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cut = int(len(panel) * DEV_FRACTION)
    return panel.iloc[:cut], panel.iloc[cut:]


def _row(label: str, s: Stats) -> str:
    return (
        f"{label:<46} gross_SR={s.sharpe_gross:6.2f}  net_SR={s.sharpe_net:7.2f}  "
        f"trades={s.n_trades:6d} ({s.trades_per_year:8.0f}/J)  "
        f"gross_bps/Trade={s.gross_bps_per_trade:7.2f}  net_bps/Trade={s.net_bps_per_trade:7.2f}  "
        f"exposure={s.time_in_market:.2f}"
    )


def main() -> None:
    interval = "15m"
    panel = load_panel(("BTCUSDT", "ETHUSDT"), interval=interval)
    panel = panel.dropna()
    print(f"Panel: {len(panel):,} Bars  {panel.index.min()} .. {panel.index.max()}")
    print(f"= {len(panel) / BARS_PER_YEAR[interval]:.2f} Jahre 24/7\n")

    dev, hold = split(panel)
    print(f"DEV:     {len(dev):,} Bars  {dev.index.min()} .. {dev.index.max()}")
    print(f"HOLDOUT: {len(hold):,} Bars  {hold.index.min()} .. {hold.index.max()}\n")

    # Buy&Hold-Referenz auf DEV
    print("--- Buy & Hold (Referenz, DEV) ---")
    for sym in panel.columns:
        ones = pd.Series(1.0, index=dev.index)
        print(_row(f"buyhold {sym}", evaluate(dev[sym], ones, interval)))
    print()

    print("--- DEV: alle Kandidaten ---")
    results: list[tuple[float, str, str, dict, Stats]] = []
    for sym in panel.columns:
        close = dev[sym]
        for kind, params in candidates():
            sig = build(kind, close, params)
            st = evaluate(close, sig, interval)
            label = f"{sym} {kind} {params}"
            results.append((st.sharpe_net, sym, kind, params, st))
            print(_row(label, st))

    print("\n--- DEV: Top 5 nach Netto-Sharpe ---")
    results.sort(key=lambda r: r[0], reverse=True)
    for _, sym, kind, params, st in results[:5]:
        print(_row(f"{sym} {kind} {params}", st))

    print("\n--- DEV: Top 5 nach BRUTTO-Sharpe (wo ist ueberhaupt Edge?) ---")
    by_gross = sorted(results, key=lambda r: r[4].sharpe_gross, reverse=True)
    for _, sym, kind, params, st in by_gross[:5]:
        print(_row(f"{sym} {kind} {params}", st))

    print("\n--- DEV: bester Brutto-bps/Trade ---")
    by_bps = sorted(results, key=lambda r: r[4].gross_bps_per_trade, reverse=True)
    for _, sym, kind, params, st in by_bps[:5]:
        print(_row(f"{sym} {kind} {params}", st))

    # Holdout: EINMAL, mit den auf DEV besten Varianten je These und Symbol
    print("\n" + "=" * 100)
    print("HOLDOUT (einmalige Bewertung)")
    print("=" * 100)
    for sym in panel.columns:
        ones = pd.Series(1.0, index=hold.index)
        print(_row(f"buyhold {sym}", evaluate(hold[sym], ones, interval)))
    for sym in panel.columns:
        for kind in ("momentum", "reversion", "breakout"):
            best = max(
                (r for r in results if r[1] == sym and r[2] == kind),
                key=lambda r: r[0],
            )
            params = best[3]
            sig = build(kind, hold[sym], params)
            st = evaluate(hold[sym], sig, interval)
            print(_row(f"{sym} {kind} {params}", st))


if __name__ == "__main__":
    main()
