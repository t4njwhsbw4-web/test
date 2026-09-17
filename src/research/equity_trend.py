"""Trendfolge + Vol-Targeting auf US-Aktienindizes: der Out-of-Universe-Test.

Fragestellung
-------------
Bei Krypto hat von fünf Thesen nur eine den versiegelten Hold-out überlebt:
Trendfolge mit Volatilitäts-Targeting - kein Rendite-Edge, aber der maximale
Drawdown sank von ~-70% auf ~-20%, und dieser Effekt übertrug sich
out-of-sample. Schwäche des Tests: 5 Jahre Historie, ein einziges Regime
(Bärenmarkt) im Hold-out.

Dieses Skript testet dieselbe Strategie-Familie (identischer Code, importiert
aus src.strategies.trend_vol_target) auf US-Aktienindizes mit langer Historie.
Trendfolge auf Indizes ist der am besten dokumentierte systematische Ansatz
(Managed Futures / CTA) - wenn die These irgendwo halten muss, dann hier.

Zeitraum-Disziplin
------------------
- DEV:     Beginn der Historie bis 2018-12-31. Nur hier wird gesucht.
- HOLDOUT: 2019-01-01 bis heute. Enthält COVID-Crash, 2021-Bullenmarkt,
           2022-Bärenmarkt, 2023+ KI-Rally. Wird GENAU EINMAL bewertet,
           mit den vorher per Regel fixierten Parametern.
Die Auswahlregel (`select_finalists`) steht als Code fest, bevor der Hold-out
angefasst wird - sie ist nicht nachträglich justierbar.

Datenwahl
---------
- ^GSPC ab 1962-01-02: echte Intraday-Hochs/Tiefs erst ab diesem Datum
  (davor high == low == close, was Donchian-Kanäle verfälschen würde).
  57 Jahre Dev-Historie: Stagflation 1966-82, 1973/74 (-48%), Crash 1987,
  1990, Dotcom, Finanzkrise. Preisindex OHNE Dividenden.
- SPY ab 1993, QQQ ab 1999: tatsächlich handelbare ETFs, yfinance
  auto_adjust=True, also inklusive Dividenden.
Index-ETFs statt Einzelaktien: kein Survivorship-Bias in der Titelauswahl.

Kosten
------
3 bps pro Seite. SPY/QQQ haben Spreads von ~0.2-1 bp, Kommission bei
Retail-Brokern 0, Market-Impact bei Retail-Größen ~0. 10 bps (Krypto-Taker)
wären hier um eine Größenordnung zu hoch. 3 bps ist eher konservativ. Eine
Sensitivität mit 10 bps wird für die Finalisten mitberichtet.

Bekannte Verzerrung
-------------------
Die Engine verzinst Cash nicht: ist die Strategie flat, verdient sie 0%.
Historisch gab es auf Cash aber 2-8% p.a. Das benachteiligt die Trendfolge
systematisch. Deshalb wird zusätzlich eine klar gekennzeichnete
Cash-Zins-Sensitivität ausgewiesen (Aufschlag nur auf den nicht investierten
Anteil, Buy-and-Hold erhält korrekterweise keinen).
"""
from __future__ import annotations

import itertools
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from src.data.fetch import fetch_ohlcv
from src.research.harness import evaluate
from src.strategies.trend_vol_target import generate_signals

CACHE_DIR = "data_cache"
COST_BPS = 3.0
COST_BPS_STRESS = 10.0
HISTORY_DAYS = 27500          # ~75 Jahre; yfinance liefert, was es hat

DEV_END = pd.Timestamp("2018-12-31")
HOLDOUT_START = pd.Timestamp("2019-01-01")
WARMUP_BARS = 300             # alle Indikatoren (max. 252-Tage-Lookback) gültig

SYMBOLS = {
    "^GSPC": pd.Timestamp("1962-01-02"),   # echte High/Low-Daten erst ab hier
    "SPY": None,
    "QQQ": None,
}

# Cash-Zins-Annahmen für die Sensitivität (grobe Durchschnitte 3M-T-Bill).
RF_DEV = 0.045
RF_HOLDOUT = 0.025

# ---------------------------------------------------------------------------
# Parameterraum (vollständig auf Dev-Daten gesucht)
# ---------------------------------------------------------------------------
# Aktienindizes haben eine realisierte Vol von ~12-18% - deutlich unter
# Krypto (~60-80%). Deshalb werden hier auch niedrige target_vol-Werte
# getestet: bei target_vol=0.30 wäre die Skalierung fast immer auf 1.0
# gecappt und das Vol-Targeting praktisch inaktiv.
TREND_SPECS = [
    {"trend_mode": "none"},
    {"trend_mode": "price_ma", "slow_ma": 100},
    {"trend_mode": "price_ma", "slow_ma": 200},
    {"trend_mode": "ma_cross", "fast_ma": 20, "slow_ma": 100},
    {"trend_mode": "ma_cross", "fast_ma": 50, "slow_ma": 100},
    {"trend_mode": "ma_cross", "fast_ma": 50, "slow_ma": 200},
    {"trend_mode": "tsmom"},
    {"trend_mode": "donchian", "donchian_entry": 20, "donchian_exit": 10},
    {"trend_mode": "donchian", "donchian_entry": 55, "donchian_exit": 20},
    {"trend_mode": "donchian", "donchian_entry": 100, "donchian_exit": 50},
]

VOL_SPECS = [{"vol_mode": "off"}] + [
    {"vol_mode": "target", "target_vol": tv, "vol_window": vw}
    for tv, vw in itertools.product([0.08, 0.10, 0.12, 0.15, 0.20, 0.30], [20, 60])
]

BANDS = [0.05, 0.10]


def param_combos() -> list[dict]:
    out = []
    for trend, vol, band in itertools.product(TREND_SPECS, VOL_SPECS, BANDS):
        combo = {**trend, **vol, "rebalance_band": band}
        out.append(combo)
    return out


def combo_key(combo: dict) -> str:
    return "|".join(f"{k}={combo[k]}" for k in sorted(combo))


# ---------------------------------------------------------------------------
# Daten
# ---------------------------------------------------------------------------
def load(symbol: str) -> pd.DataFrame:
    df = fetch_ohlcv(symbol, history_days=HISTORY_DAYS, cache_dir=CACHE_DIR).sort_index()
    start = SYMBOLS.get(symbol)
    if start is not None:
        df = df.loc[df.index >= start]
    return df


def dev_window(df: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Dev-Fenster nach Warmup. Für alle Kombinationen und B&H identisch."""
    return df.index[WARMUP_BARS], DEV_END


# ---------------------------------------------------------------------------
# Auswertung
# ---------------------------------------------------------------------------
def _slice(s: pd.Series, start, end) -> pd.Series:
    return s.loc[(s.index >= start) & (s.index <= end)]


def eval_window(
    df: pd.DataFrame,
    signals: pd.Series,
    start,
    end,
    symbol: str,
    params: dict,
    cost_bps: float = COST_BPS,
) -> dict | None:
    """Bewertet ein Zeitfenster. Signale werden auf der GESAMTEN Historie
    berechnet (rein trailing, kein Look-Ahead) und dann geschnitten - so ist
    der Warmup sauber, ohne dass Zukunftsdaten einfließen."""
    prices = _slice(df["close"], start, end)
    if len(prices) < 10:   # kurze Krisenfenster (z.B. COVID: 23 Bars) zulassen
        return None
    sig = signals.reindex(prices.index).fillna(0.0)
    res = evaluate(sig, prices, symbol, params, cost_bps=cost_bps)
    # StrategyResult liefert nur die Equity-Kurve; die Tagesrenditen daraus
    # rekonstruieren (equity = 10_000 * (1 + r).cumprod()).
    eq = res.equity_curve
    rets = eq.pct_change()
    rets.iloc[0] = eq.iloc[0] / 10_000.0 - 1.0
    years = len(prices) / 252
    dd = abs(res.max_drawdown)
    return {
        "sharpe": res.sharpe,
        "cagr": res.cagr,
        "max_dd": res.max_drawdown,
        "calmar": res.cagr / dd if dd > 1e-9 else np.nan,
        "total_return": res.total_return,
        "n_trades": res.n_trades,
        "trades_per_year": res.n_trades / years if years > 0 else np.nan,
        "exposure": float(sig.shift(1).fillna(0.0).mean()),
        "win_rate": res.win_rate,
        "top3_day": res.top3_day_concentration,
        "years": years,
        "returns": rets,
        "position": sig.shift(1).fillna(0.0),
    }


def eval_bh(df: pd.DataFrame, start, end, symbol: str) -> dict | None:
    prices = _slice(df["close"], start, end)
    if len(prices) < 30:
        return None
    ones = pd.Series(1.0, index=prices.index)
    return eval_window(df, ones, start, end, symbol, {"strategy": "buy_and_hold"}, cost_bps=0.0)


def with_cash_yield(stats: dict, rf: float) -> dict:
    """Sensitivität: nicht investierter Anteil verzinst sich mit rf p.a."""
    ret = stats["returns"] + (1.0 - stats["position"]) * (rf / 252)
    eq = (1 + ret).cumprod()
    total = eq.iloc[-1] - 1
    years = stats["years"]
    cagr = (1 + total) ** (1 / years) - 1
    std = ret.std()
    sharpe = (ret.mean() / std) * np.sqrt(252) if std > 0 else 0.0
    dd = abs((eq / eq.cummax() - 1).min())
    return {"sharpe": sharpe, "cagr": cagr, "max_dd": -dd,
            "calmar": cagr / dd if dd > 1e-9 else np.nan}


# ---------------------------------------------------------------------------
# Auswahlregel - FIXIERT, bevor der Hold-out angefasst wird
# ---------------------------------------------------------------------------
def select_finalists(dev: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    """Vorab festgelegte Regel:

    1. Die Kombination muss auf ALLEN Dev-Symbolen einen höheren Sharpe als
       Buy-and-Hold liefern (risikoadjustiert schlagen, nicht nur bei einem
       Symbol - das wäre Symbol-Cherry-Picking).
    2. Sie muss auf ALLEN Dev-Symbolen einen kleineren maximalen Drawdown als
       Buy-and-Hold haben (die These ist Drawdown-Schutz).
    3. Ranking nach dem MINIMUM des Sharpe über die Symbole (worst case),
       nicht nach dem Mittel - das bestraft Ausreißer-Glück.
    Fällt Filter 1+2 auf 0 Kombinationen, wird ohne Filter nach Regel 3
       gerankt und das im Bericht offengelegt.
    """
    piv = dev.pivot_table(index="key", columns="symbol",
                          values=["sharpe", "max_dd", "bh_sharpe", "bh_max_dd"])
    beats_sharpe = (piv["sharpe"] > piv["bh_sharpe"]).all(axis=1)
    beats_dd = (piv["max_dd"] > piv["bh_max_dd"]).all(axis=1)
    worst_sharpe = piv["sharpe"].min(axis=1)

    ok = beats_sharpe & beats_dd
    ranked = worst_sharpe[ok].sort_values(ascending=False)
    filtered = True
    if ranked.empty:
        ranked = worst_sharpe.sort_values(ascending=False)
        filtered = False

    res = dev[dev["key"].isin(ranked.index[:n])].copy()
    res["worst_sharpe"] = res["key"].map(worst_sharpe)
    res.attrs["filtered"] = filtered
    res.attrs["n_passing"] = int(ok.sum())
    res.attrs["order"] = list(ranked.index[:n])
    res.attrs["passing_keys"] = list(piv.index[ok])
    return res


# ---------------------------------------------------------------------------
# Regime-Fenster
# ---------------------------------------------------------------------------
DEV_REGIMES = [
    ("1962-1969", "1963-03-01", "1969-12-31"),
    ("1970er (Stagflation)", "1970-01-01", "1979-12-31"),
    ("1980er", "1980-01-01", "1989-12-31"),
    ("1990er", "1990-01-01", "1999-12-31"),
    ("2000er", "2000-01-01", "2009-12-31"),
    ("2010-2018", "2010-01-01", "2018-12-31"),
    ("BEAR 1973/74", "1973-01-11", "1974-10-03"),
    ("CRASH 1987", "1987-08-25", "1987-12-04"),
    ("BEAR Dotcom 2000-02", "2000-03-24", "2002-10-09"),
    ("BEAR Finanzkrise 07-09", "2007-10-09", "2009-03-09"),
]

HOLDOUT_REGIMES = [
    ("COVID-Crash 2020", "2020-02-19", "2020-03-23"),
    ("Bull 2020-04..2021", "2020-04-01", "2021-12-31"),
    ("BEAR 2022", "2022-01-03", "2022-10-12"),
    ("2023-heute", "2023-01-01", "2030-01-01"),
]


def fmt(v, pct=False):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "  n/a"
    return f"{v * 100:6.1f}%" if pct else f"{v:6.2f}"


def line(label, s, width=26):
    return (f"{label:<{width}} sharpe {fmt(s['sharpe'])}  cagr {fmt(s['cagr'], True)}"
            f"  maxDD {fmt(s['max_dd'], True)}  calmar {fmt(s['calmar'])}"
            f"  expo {fmt(s['exposure'], True)}  tr/J {fmt(s['trades_per_year'])}")


# ---------------------------------------------------------------------------
def main() -> None:
    data = {s: load(s) for s in SYMBOLS}
    combos = param_combos()

    print("=" * 108)
    print("DATEN")
    print("=" * 108)
    for s, df in data.items():
        d0, d1 = dev_window(df)
        print(f"{s:<7} {len(df):>6} Bars  {df.index.min().date()} .. {df.index.max().date()}"
              f"   Dev-Fenster {d0.date()} .. {d1.date()}")
    print(f"\n{len(combos)} Parameter-Kombinationen x {len(data)} Symbole = "
          f"{len(combos) * len(data)} Dev-Backtests, Kosten {COST_BPS} bps/Seite")

    # ---------------- DEV: Grid ----------------
    signal_cache: dict[tuple[str, str], pd.Series] = {}
    bh_dev = {}
    for sym, df in data.items():
        d0, d1 = dev_window(df)
        bh_dev[sym] = eval_bh(df, d0, d1, sym)

    rows = []
    for sym, df in data.items():
        d0, d1 = dev_window(df)
        dev_df = df.loc[df.index <= DEV_END]          # Entwicklung sieht NUR Dev-Daten
        for combo in combos:
            key = combo_key(combo)
            sig = generate_signals(dev_df, **combo)
            signal_cache[(sym, key)] = sig
            st = eval_window(dev_df, sig, d0, d1, sym, combo)
            if st is None:
                continue
            rows.append({
                "symbol": sym, "key": key, **{k: v for k, v in combo.items()},
                **{k: st[k] for k in ("sharpe", "cagr", "max_dd", "calmar",
                                      "trades_per_year", "exposure", "win_rate",
                                      "n_trades", "top3_day")},
                "bh_sharpe": bh_dev[sym]["sharpe"],
                "bh_max_dd": bh_dev[sym]["max_dd"],
            })
    dev = pd.DataFrame(rows)

    print("\n" + "=" * 108)
    print("DEV: BUY-AND-HOLD BENCHMARK (Messlatte)")
    print("=" * 108)
    for sym in data:
        print(line(f"B&H {sym}", bh_dev[sym]))

    # Ablations-Übersicht: welche Komponente trägt?
    print("\n" + "=" * 108)
    print("DEV: ABLATION - Mittelwerte über Symbole & Parameter je Komponente")
    print("=" * 108)
    dev["familie"] = np.where(
        dev["trend_mode"] == "none",
        np.where(dev["vol_mode"] == "off", "gar nichts (=B&H)", "nur Vol-Targeting"),
        np.where(dev["vol_mode"] == "off", "nur Trend", "Trend + Vol-Targeting"))
    agg = dev.groupby("familie")[["sharpe", "cagr", "max_dd", "calmar",
                                  "trades_per_year", "exposure"]].mean()
    print(agg.round(3).to_string())

    print("\nDEV: nach trend_mode (nur Kombinationen MIT Vol-Targeting)")
    sub = dev[(dev["vol_mode"] == "target")]
    print(sub.groupby("trend_mode")[["sharpe", "cagr", "max_dd", "calmar"]]
          .mean().round(3).to_string())
    print("\nDEV: nach target_vol (nur Kombinationen MIT Trend + Vol-Targeting)")
    sub2 = sub[sub["trend_mode"] != "none"]
    print(sub2.groupby("target_vol")[["sharpe", "cagr", "max_dd", "calmar", "exposure"]]
          .mean().round(3).to_string())

    # ---------------- Finalisten wählen (Regel steht oben fest) ----------------
    fin = select_finalists(dev, n=3)
    order = fin.attrs["order"]
    print("\n" + "=" * 108)
    print("DEV: FINALISTEN (Auswahlregel vorab fixiert: schlägt B&H in Sharpe UND maxDD "
          "auf allen 3 Symbolen; Ranking nach schlechtestem Symbol-Sharpe)")
    print("=" * 108)
    print(f"Kombinationen, die den Filter passieren: {fin.attrs['n_passing']} von {len(combos)}"
          f"{'' if fin.attrs['filtered'] else '  -> FILTER LEER, ungefiltertes Ranking!'}")
    for i, key in enumerate(order, 1):
        block = fin[fin["key"] == key]
        print(f"\n[{i}] {key}")
        print(f"    worst-Symbol-Sharpe {block['worst_sharpe'].iloc[0]:.3f}")
        for sym in data:
            r = block[block["symbol"] == sym]
            if r.empty:
                continue
            r = r.iloc[0]
            print(f"    {sym:<6} sharpe {r.sharpe:6.2f} (B&H {r.bh_sharpe:5.2f})"
                  f"  cagr {r.cagr * 100:6.1f}%  maxDD {r.max_dd * 100:6.1f}%"
                  f" (B&H {r.bh_max_dd * 100:6.1f}%)  calmar {r.calmar:5.2f}"
                  f"  tr/J {r.trades_per_year:5.1f}  expo {r.exposure * 100:4.0f}%")

    # ---------------- DEV: Regime-Analyse ----------------
    print("\n" + "=" * 108)
    print("DEV: REGIME-ANALYSE ^GSPC (hält die Strategie regimeübergreifend?)")
    print("=" * 108)
    gspc = data["^GSPC"]
    gspc_dev = gspc.loc[gspc.index <= DEV_END]
    for name, a, b in DEV_REGIMES:
        a, b = pd.Timestamp(a), min(pd.Timestamp(b), DEV_END)
        bh = eval_bh(gspc_dev, a, b, "^GSPC")
        if bh is None:
            continue
        print(f"\n{name}   ({a.date()} .. {b.date()})")
        print("   " + line("B&H", bh, 22))
        for i, key in enumerate(order, 1):
            st = eval_window(gspc_dev, signal_cache[("^GSPC", key)], a, b, "^GSPC", {})
            if st:
                print("   " + line(f"Finalist {i}", st, 22))

    # ---------------- HOLD-OUT: genau ein Blick ----------------
    print("\n" + "=" * 108)
    print("HOLD-OUT 2019-01-01 .. heute - EINMALIGE BEWERTUNG, Parameter fixiert")
    print("=" * 108)
    ho_stats: dict[tuple[str, str], dict] = {}
    for sym, df in data.items():
        ho_end = df.index.max()
        bh = eval_bh(df, HOLDOUT_START, ho_end, sym)
        print(f"\n--- {sym} ({HOLDOUT_START.date()} .. {ho_end.date()}) ---")
        print("   " + line("B&H", bh, 22))
        ho_stats[(sym, "bh")] = bh
        for i, key in enumerate(order, 1):
            combo = dict(x.split("=", 1) for x in key.split("|"))
            # Signale auf der vollen Historie, rein trailing -> Warmup sauber
            sig = generate_signals(df, **_typed(combo))
            st = eval_window(df, sig, HOLDOUT_START, ho_end, sym, {})
            ho_stats[(sym, key)] = st
            print("   " + line(f"Finalist {i}", st, 22))
            stress = eval_window(df, sig, HOLDOUT_START, ho_end, sym, {},
                                 cost_bps=COST_BPS_STRESS)
            cash = with_cash_yield(st, RF_HOLDOUT)
            print(f"       @{COST_BPS_STRESS:.0f}bps: sharpe {fmt(stress['sharpe'])}"
                  f"  cagr {fmt(stress['cagr'], True)}"
                  f"   | +{RF_HOLDOUT * 100:.1f}% Cash-Zins: sharpe {fmt(cash['sharpe'])}"
                  f"  cagr {fmt(cash['cagr'], True)}  maxDD {fmt(cash['max_dd'], True)}")

    # Robustheit: hängt das Hold-out-Ergebnis an der konkreten Top-3-Auswahl?
    # Verteilung über ALLE Kombinationen, die den Dev-Filter passiert haben.
    # Kein Nachjustieren - dieselben, vorher fixierten Parameter, nur breiter
    # ausgewertet.
    print("\n" + "=" * 108)
    print("HOLD-OUT: ROBUSTHEIT - Verteilung über ALLE "
          f"{fin.attrs['n_passing']} Dev-qualifizierten Kombinationen")
    print("=" * 108)
    for sym, df in data.items():
        ho_end = df.index.max()
        bh = ho_stats[(sym, "bh")]
        sh, dd = [], []
        for key in fin.attrs["passing_keys"]:
            sig = generate_signals(df, **_typed(dict(x.split("=", 1) for x in key.split("|"))))
            st = eval_window(df, sig, HOLDOUT_START, ho_end, sym, {})
            if st:
                sh.append(st["sharpe"])
                dd.append(st["max_dd"])
        sh, dd = np.array(sh), np.array(dd)
        print(f"{sym:<6} Sharpe  p10 {np.percentile(sh, 10):5.2f}  median "
              f"{np.median(sh):5.2f}  p90 {np.percentile(sh, 90):5.2f}   (B&H {bh['sharpe']:5.2f})"
              f"  Anteil > B&H: {(sh > bh['sharpe']).mean() * 100:4.0f}%")
        print(f"{'':<6} maxDD   p10 {np.percentile(dd, 10) * 100:5.1f}% median "
              f"{np.median(dd) * 100:5.1f}% p90 {np.percentile(dd, 90) * 100:5.1f}%"
              f"  (B&H {bh['max_dd'] * 100:5.1f}%)"
              f"  Anteil besser: {(dd > bh['max_dd']).mean() * 100:4.0f}%")

    print("\n" + "=" * 108)
    print("HOLD-OUT: REGIME-AUFSCHLÜSSELUNG ^GSPC")
    print("=" * 108)
    for name, a, b in HOLDOUT_REGIMES:
        a = pd.Timestamp(a)
        b = min(pd.Timestamp(b), gspc.index.max())
        bh = eval_bh(gspc, a, b, "^GSPC")
        if bh is None:
            continue
        print(f"\n{name}   ({a.date()} .. {b.date()})")
        print(f"   {'B&H':<22} total {fmt(bh['total_return'], True)}  maxDD {fmt(bh['max_dd'], True)}")
        for i, key in enumerate(order, 1):
            sig = generate_signals(gspc, **_typed(dict(x.split("=", 1) for x in key.split("|"))))
            st = eval_window(gspc, sig, a, b, "^GSPC", {})
            if st:
                print(f"   {'Finalist ' + str(i):<22} total {fmt(st['total_return'], True)}"
                      f"  maxDD {fmt(st['max_dd'], True)}  expo {fmt(st['exposure'], True)}")

    # ---------------- Dev vs Hold-out Gegenüberstellung ----------------
    print("\n" + "=" * 108)
    print("DEV vs HOLD-OUT (Übertragbarkeit)")
    print("=" * 108)
    print(f"{'Symbol/Variante':<34}{'Dev Sharpe':>12}{'HO Sharpe':>12}"
          f"{'Dev maxDD':>12}{'HO maxDD':>12}{'HO CAGR':>10}")
    for sym in data:
        b = bh_dev[sym]
        h = ho_stats[(sym, "bh")]
        print(f"{sym + ' B&H':<34}{b['sharpe']:>12.2f}{h['sharpe']:>12.2f}"
              f"{b['max_dd'] * 100:>11.1f}%{h['max_dd'] * 100:>11.1f}%{h['cagr'] * 100:>9.1f}%")
        for i, key in enumerate(order, 1):
            d = dev[(dev.symbol == sym) & (dev.key == key)]
            if d.empty:
                continue
            d = d.iloc[0]
            h = ho_stats[(sym, key)]
            print(f"{sym + f' Finalist {i}':<34}{d.sharpe:>12.2f}{h['sharpe']:>12.2f}"
                  f"{d.max_dd * 100:>11.1f}%{h['max_dd'] * 100:>11.1f}%{h['cagr'] * 100:>9.1f}%")


def _typed(combo: dict) -> dict:
    """Rekonstruiert Parametertypen aus dem String-Key."""
    out = {}
    for k, v in combo.items():
        if k in ("trend_mode", "vol_mode"):
            out[k] = v
        elif k in ("target_vol", "rebalance_band"):
            out[k] = float(v)
        else:
            out[k] = int(v)
    return out


if __name__ == "__main__":
    main()
