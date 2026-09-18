"""Kombination der zwei ueberlebenden Mechanismen: Trendfilter je Asset
UND Vol-Targeting auf Portfolioebene.

Frage
-----
Zwei Komponenten haben unabhaengig einen Out-of-Sample-Test ueberlebt, beide
wirken ueber Risiko, nicht ueber Prognose:

1. Trendfolge + Vol-Targeting auf Einzelassets
   (`src/strategies/trend_vol_target.py`) - senkt Drawdowns massiv, ohne
   Renditevorteil.
2. Multi-Asset-Allokation + Vol-Targeting auf Portfolioebene
   (`src/research/multi_asset.py`) - Dev-Sharpe 1.10 -> Hold-out 1.24.

Die bekannte Schwaeche von (2) war 2022: Aktien und Anleihen fielen
gleichzeitig, Diversifikation versagte, weil die Korrelation kippte. Ein
Trendfilter ist korrelations-agnostisch - er nimmt ein fallendes Asset heraus,
egal was die anderen tun. Genau dort muesste die Kombination ihren Mehrwert
zeigen, falls sie einen hat.

Aufbau der Kombination
----------------------
    gate_i[t]   = Trendfilter je Asset (0/1), aus trend_vol_target.generate_signals
                  mit vol_mode="off" - also exakt die validierte Trendlogik.
    base_w[t]   = Equal-Weight ueber die Assets mit gate == 1
                  Variante "cash":     Gewicht eines gefilterten Assets -> Cash
                  Variante "redistr.": Gewicht wird auf die Ueberlebenden verteilt
    w[t]        = base_w[t] * min(target_vol / exante_vol(base_w, t), 1.0)

Long-only, Gesamtexposure <= 1.0, kein Hebel, keine Shorts.

Validierung: Walk-Forward, KEIN Re-Use des 2019er Hold-outs
-----------------------------------------------------------
Der Hold-out 2019-2026 wurde fuer die Multi-Asset-Strategie schon einmal
verbraucht. Ihn erneut zur Variantenauswahl zu nutzen, wuerde ihn zu einem
zweiten Trainingsset machen. Stattdessen: expandierendes Trainingsfenster,
Parameterwahl NUR darauf, Bewertung auf dem darauffolgenden 2-Jahres-Block,
dann Fenster erweitern. Jeder OOS-Block wird einzeln berichtet - nur so ist
sichtbar, ob ein Vorteil konsistent ist oder aus einer Periode stammt.

Kein Look-Ahead
---------------
- Gates und Vol-Schaetzungen sind rollierend und nutzen nur Bars <= t
  (Donchian-Kanaele zusaetzlich mit .shift(1)).
- Ausfuehrung ein Bar verzoegert (`run_portfolio` aus multi_asset).
- Parameter eines OOS-Blocks stammen ausschliesslich aus Daten VOR dem Block.
- Keine zentrierten Fenster, kein shift(-n).

Kosten: 3 bps pro Seite (COST_BPS aus multi_asset), Turnover und Kostendrag
werden pro Block mitberichtet - ein Trendfilter erhoeht den Turnover, und das
muss in den Netto-Zahlen stehen.
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from src.data.fetch import fetch_ohlcv
from src.research.multi_asset import (
    COST_BPS,
    TRADING_DAYS_PER_YEAR,
    apply_vol_target,
    load_closes,
    performance_metrics,
    rebalance_flags,
    run_portfolio,
    weights_buy_and_hold,
    weights_equal,
)
from src.strategies.trend_vol_target import generate_signals

CACHE_DIR = "data_cache"
HISTORY_DAYS = 12000

# Briefing-Universum der validierten Allokation. GLD (ab 2004-11-18) bindet den
# gemeinsamen Zeitraum; IEF/TLT starten 2002, SPY 1993, QQQ 1999.
UNIVERSE = ["SPY", "QQQ", "TLT", "GLD"]
BENCHMARK_SYMBOLS = ["SPY", "QQQ"]

WARMUP_BARS = 300          # alle 252-Tage-Lookbacks gueltig
FIRST_OOS_YEAR = 2008      # erster OOS-Block - die Finanzkrise ist damit OOS
OOS_BLOCK_YEARS = 2

# --- Parameterraum (klein gehalten; jede Wahl faellt nur auf Trainingsdaten) --
TREND_SPECS = {
    "donch55": dict(trend_mode="donchian", donchian_entry=55, donchian_exit=20),
    "sma200": dict(trend_mode="price_ma", slow_ma=200),
    "sma100": dict(trend_mode="price_ma", slow_ma=100),
}
VT_LOOKBACKS = (63, 126, 252)
TARGET_VOLS = (0.06, 0.08, 0.10)
REBALANCE_FREQS = ("monthly", "weekly", "daily")
SINGLE_TARGET_VOLS = (0.10, 0.15, 0.20)

METRIC_COLS = ("sharpe", "cagr", "max_drawdown", "calmar")


# ---------------------------------------------------------------------------
# Daten
# ---------------------------------------------------------------------------
def load_ohlc(symbols, history_days: int = HISTORY_DAYS) -> dict:
    """Volle OHLC-Historie je Symbol (Donchian braucht High/Low)."""
    return {
        s: fetch_ohlcv(s, history_days, cache_dir=CACHE_DIR).sort_index() for s in symbols
    }


# ---------------------------------------------------------------------------
# Trendfilter je Asset
# ---------------------------------------------------------------------------
def trend_gates(ohlc: dict, index: pd.DatetimeIndex, spec: dict) -> pd.DataFrame:
    """0/1-Trendfilter je Asset, berechnet auf der NATIVEN Historie des Assets.

    `generate_signals(..., vol_mode="off")` ist exakt die validierte
    Trendlogik ohne Sizing - kein Nachbau. Die Berechnung auf der nativen
    Historie (nicht auf dem beschnittenen gemeinsamen Kalender) gibt dem
    Donchian-Zustand seinen Warmup, ohne Zukunftsinformation zu nutzen;
    danach wird auf den gemeinsamen Kalender reindexiert.
    """
    cols = {}
    for sym, df in ohlc.items():
        gate = generate_signals(
            df, vol_mode="off", max_weight=1.0, rebalance_band=0.0, **spec
        )
        cols[sym] = gate.reindex(index).ffill().fillna(0.0)
    return pd.DataFrame(cols, index=index)


def weights_trend_equal(
    closes: pd.DataFrame,
    universe,
    gates: pd.DataFrame,
    redistribute: bool,
) -> pd.DataFrame:
    """Equal-Weight nur ueber Assets im Aufwaertstrend.

    redistribute=False: das Gewicht eines ausgefilterten Assets geht in Cash.
    redistribute=True:  es wird gleichmaessig auf die Ueberlebenden verteilt
                        (Summe 1, solange mindestens ein Asset durchkommt).
    """
    syms = list(universe)
    g = gates[syms].reindex(closes.index).fillna(0.0)
    if redistribute:
        active = g.sum(axis=1)
        w = g.div(active.replace(0.0, np.nan), axis=0).fillna(0.0)
    else:
        w = g / float(len(syms))
    out = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)
    out[syms] = w
    return out


def weights_single_trend(
    closes: pd.DataFrame,
    ohlc: dict,
    symbol: str,
    target_vol: float,
) -> pd.DataFrame:
    """Messlatte (c): reine Trendfolge + Vol-Targeting auf EINEM Asset.

    Konfiguration wie validiert: Donchian 55/20, vol_window=60,
    rebalance_band=0.10. Nur das Vol-Ziel wird auf dem Trainingsfenster
    gewaehlt (bei ETFs ist 0.30 wie bei Krypto nie bindend).
    """
    sig = generate_signals(
        ohlc[symbol],
        trend_mode="donchian",
        donchian_entry=55,
        donchian_exit=20,
        vol_mode="target",
        target_vol=target_vol,
        vol_window=60,
        max_weight=1.0,
        rebalance_band=0.10,
    )
    out = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)
    out[symbol] = sig.reindex(closes.index).ffill().fillna(0.0)
    return out


# ---------------------------------------------------------------------------
# Kandidaten (Gewichte einmal auf der VOLLEN Historie, rollierend/kausal)
# ---------------------------------------------------------------------------
def build_candidates(closes: pd.DataFrame, ohlc: dict, universe) -> dict:
    """{familie: {parameter-key: (weights, rebalance_flags)}}.

    Die Gewichte sind kausal (nur Daten <= t), deshalb duerfen sie einmal auf
    der Gesamthistorie berechnet und pro Fenster nur zugeschnitten werden -
    genau das Muster aus multi_asset.evaluate.
    """
    flags = {f: rebalance_flags(closes.index, f) for f in REBALANCE_FREQS}
    daily = rebalance_flags(closes.index, "daily")

    gates = {k: trend_gates(ohlc, closes.index, spec) for k, spec in TREND_SPECS.items()}
    eq = weights_equal(closes, universe)

    families: dict[str, dict] = {
        "alloc_vt": {},        # (b) reine Allokation + Vol-Target, kein Trendfilter
        "combo_cash": {},      # Kombination, gefiltertes Gewicht -> Cash
        "combo_redistr": {},   # Kombination, gefiltertes Gewicht umverteilt
        "single_trend": {},    # (c) Trendfolge ohne Diversifikation
    }

    for lb in VT_LOOKBACKS:
        for tv in TARGET_VOLS:
            w = apply_vol_target(closes, eq, lb, tv)
            for f in REBALANCE_FREQS:
                families["alloc_vt"][f"lb{lb}_tv{tv:.2f}_{f}"] = (w, flags[f])

    for gate_name, g in gates.items():
        for redistribute, fam in ((False, "combo_cash"), (True, "combo_redistr")):
            base = weights_trend_equal(closes, universe, g, redistribute)
            for lb in VT_LOOKBACKS:
                for tv in TARGET_VOLS:
                    w = apply_vol_target(closes, base, lb, tv)
                    for f in REBALANCE_FREQS:
                        key = f"{gate_name}_lb{lb}_tv{tv:.2f}_{f}"
                        families[fam][key] = (w, flags[f])

    for tv in SINGLE_TARGET_VOLS:
        w = weights_single_trend(closes, ohlc, "SPY", tv)
        families["single_trend"][f"SPY_tv{tv:.2f}"] = (w, daily)

    # Ablation: Trendfilter OHNE Portfolio-Vol-Target (zeigt, welcher Teil wirkt)
    families["trend_only_redistr"] = {}
    families["trend_only_cash"] = {}
    for gate_name, g in gates.items():
        for redistribute, fam in (
            (False, "trend_only_cash"),
            (True, "trend_only_redistr"),
        ):
            base = weights_trend_equal(closes, universe, g, redistribute)
            for f in REBALANCE_FREQS:
                families[fam][f"{gate_name}_{f}"] = (base, flags[f])

    return families


# ---------------------------------------------------------------------------
# Bewertung eines Fensters
# ---------------------------------------------------------------------------
def _run_window(
    closes: pd.DataFrame,
    weights: pd.DataFrame,
    reb: pd.Series,
    start: pd.Timestamp,
    end: pd.Timestamp,
    name: str = "x",
):
    sub = closes.loc[start:end]
    sub_reb = reb.loc[sub.index].copy()
    sub_reb.iloc[0] = True  # Einstieg am Fensterbeginn (gilt fuer alle gleich)
    return run_portfolio(sub, weights.loc[sub.index], sub_reb, name=name, cost_bps=COST_BPS)


def _select(
    closes: pd.DataFrame,
    candidates: dict,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    metric: str,
) -> tuple[str, dict]:
    """Bestes Parameterset einer Familie auf dem Trainingsfenster."""
    best_key, best_val, best_m = None, -np.inf, None
    for key, (w, reb) in candidates.items():
        m = _run_window(closes, w, reb, train_start, train_end, name=key).metrics
        val = m.get(metric, np.nan)
        if not np.isfinite(val):
            continue
        if val > best_val:
            best_key, best_val, best_m = key, val, m
    return best_key, best_m


def walk_forward(
    closes: pd.DataFrame,
    families: dict,
    static: dict,
    selection_metric: str = "calmar",
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Expandierendes Training, 2-Jahres-OOS-Bloecke.

    Rueckgabe: (Tabelle je Block/Strategie, OOS-Renditeserien, gewaehlte Parameter)
    """
    train_start = closes.index[WARMUP_BARS]
    last_year = int(closes.index[-1].year)

    rows, chosen_rows = [], []
    oos_returns: dict[str, list] = {k: [] for k in list(families) + list(static)}

    year = FIRST_OOS_YEAR
    while year <= last_year:
        block_end_year = min(year + OOS_BLOCK_YEARS - 1, last_year)
        t_end = pd.Timestamp(f"{year - 1}-12-31")
        o_start = pd.Timestamp(f"{year}-01-01")
        o_end = pd.Timestamp(f"{block_end_year}-12-31")
        label = f"{year}-{block_end_year % 100:02d}"
        partial = closes.loc[o_start:o_end].index[-1] < pd.Timestamp(f"{block_end_year}-12-15")

        for fam, cands in families.items():
            key, train_m = _select(closes, cands, train_start, t_end, selection_metric)
            res = _run_window(closes, *cands[key], o_start, o_end, name=fam)
            m = res.metrics
            oos_returns[fam].append(res.returns)
            rows.append(
                {
                    "block": label,
                    "strategy": fam,
                    "partial": partial,
                    **{c: m[c] for c in METRIC_COLS},
                    "turnover_pa": m["turnover_pa"],
                    "cost_drag_pa": m["cost_drag_pa"],
                    "avg_exposure": m["avg_exposure"],
                    "train_metric": train_m[selection_metric] if train_m else np.nan,
                }
            )
            chosen_rows.append(
                {"block": label, "strategy": fam, "params": key,
                 "train_start": str(train_start.date()), "train_end": str(t_end.date())}
            )

        for name, (w, reb) in static.items():
            res = _run_window(closes, w, reb, o_start, o_end, name=name)
            m = res.metrics
            oos_returns[name].append(res.returns)
            rows.append(
                {
                    "block": label,
                    "strategy": name,
                    "partial": partial,
                    **{c: m[c] for c in METRIC_COLS},
                    "turnover_pa": m["turnover_pa"],
                    "cost_drag_pa": m["cost_drag_pa"],
                    "avg_exposure": m["avg_exposure"],
                    "train_metric": np.nan,
                }
            )

        year += OOS_BLOCK_YEARS

    table = pd.DataFrame(rows)
    streams = {k: pd.concat(v).sort_index() for k, v in oos_returns.items() if v}
    return table, streams, pd.DataFrame(chosen_rows)


# ---------------------------------------------------------------------------
# Reporting-Helfer
# ---------------------------------------------------------------------------
ORDER = [
    "combo_cash",
    "combo_redistr",
    "alloc_vt",
    "trend_only_cash",
    "trend_only_redistr",
    "single_trend",
    "B&H SPY",
    "B&H QQQ",
]
SHORT = {
    "combo_cash": "KOMBI cash",
    "combo_redistr": "KOMBI umvert.",
    "alloc_vt": "(b) Alloc+VT",
    "trend_only_cash": "Trend cash o.VT",
    "trend_only_redistr": "Trend umv. o.VT",
    "single_trend": "(c) Trend SPY",
    "B&H SPY": "(a) B&H SPY",
    "B&H QQQ": "(a) B&H QQQ",
}


def pivot_block_table(table: pd.DataFrame, metric: str) -> pd.DataFrame:
    p = table.pivot_table(index="block", columns="strategy", values=metric, sort=False)
    cols = [c for c in ORDER if c in p.columns]
    return p[cols].rename(columns=SHORT)


def annual_table(streams: dict, metric: str) -> pd.DataFrame:
    out = {}
    for name, r in streams.items():
        vals = {}
        for year, chunk in r.groupby(r.index.year):
            vals[year] = performance_metrics(chunk)[metric]
        out[SHORT.get(name, name)] = pd.Series(vals)
    df = pd.DataFrame(out)
    cols = [SHORT.get(c, c) for c in ORDER if SHORT.get(c, c) in df.columns]
    return df[cols]


def _pct(df: pd.DataFrame, cols_pct=True) -> pd.DataFrame:
    return (df * 100).round(1) if cols_pct else df.round(2)


def cash_adjusted(streams: dict, table_exposure: dict, rf_daily: pd.Series) -> dict:
    """Sensitivitaet: nicht investiertes Kapital verzinst mit 3M-T-Bill.

    run_portfolio rechnet Cash mit 0% - das benachteiligt jede Strategie, die
    Exposure reduziert (also genau die hier untersuchten). Hier wird der
    Zinsertrag auf den tatsaechlich gehaltenen Cash-Anteil nachtraeglich
    addiert, ohne die Engine zu veraendern.
    """
    out = {}
    for name, r in streams.items():
        exp = table_exposure.get(name)
        if exp is None:
            out[name] = r
            continue
        rf = rf_daily.reindex(r.index).ffill().fillna(0.0)
        out[name] = r + (1.0 - exp.reindex(r.index).fillna(0.0)) * rf
    return out


def main() -> None:
    global FIRST_OOS_YEAR
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)

    closes = load_closes(UNIVERSE, HISTORY_DAYS, cache_dir=CACHE_DIR)
    ohlc = load_ohlc(UNIVERSE)

    print("=" * 100)
    print("DATENBASIS")
    print("=" * 100)
    for s in UNIVERSE:
        d = ohlc[s]
        print(f"  {s:5s} {len(d):5d} Bars  {d.index.min().date()} .. {d.index.max().date()}")
    print(
        f"\n  Gemeinsamer Zeitraum ({'/'.join(UNIVERSE)}): "
        f"{closes.index.min().date()} .. {closes.index.max().date()} ({len(closes)} Bars)"
    )
    print(f"  Warmup: erste {WARMUP_BARS} Bars nur fuer Indikatoren, Training ab "
          f"{closes.index[WARMUP_BARS].date()}")
    print(f"  Walk-Forward: expandierendes Training, OOS-Bloecke von {OOS_BLOCK_YEARS} Jahren ab {FIRST_OOS_YEAR}")
    print("  Kosten: 3 bps pro Seite und Asset, Turnover voll belastet")

    families = build_candidates(closes, ohlc, UNIVERSE)
    never = rebalance_flags(closes.index, "never")
    static = {
        f"B&H {b}": (weights_buy_and_hold(closes, b), never) for b in BENCHMARK_SYMBOLS
    }

    print("\n  Kandidaten je Familie: "
          + ", ".join(f"{k}={len(v)}" for k, v in families.items()))

    table, streams, chosen = walk_forward(closes, families, static, selection_metric="calmar")

    print("\n" + "=" * 100)
    print("WALK-FORWARD, JEDER OOS-BLOCK EINZELN (Parameter nur aus Daten VOR dem Block)")
    print("=" * 100)
    for metric, label, pct in (
        ("sharpe", "SHARPE", False),
        ("cagr", "CAGR %", True),
        ("max_drawdown", "MAX DRAWDOWN %", True),
        ("calmar", "CALMAR", False),
    ):
        print(f"\n--- {label} ---")
        print(_pct(pivot_block_table(table, metric), pct).to_string())

    print("\n--- TURNOVER p.a. (einseitig, Summe ueber Assets) ---")
    print(pivot_block_table(table, "turnover_pa").round(1).to_string())
    print("\n--- KOSTENDRAG % p.a. ---")
    print(_pct(pivot_block_table(table, "cost_drag_pa")).to_string())
    print("\n--- DURCHSCHNITTLICHES EXPOSURE ---")
    print(pivot_block_table(table, "avg_exposure").round(2).to_string())

    print("\n" + "=" * 100)
    print("AGGREGAT UEBER ALLE OOS-BLOECKE (Renditeserien verkettet, keine Ueberlappung)")
    print("=" * 100)
    agg_rows = {}
    for name, r in streams.items():
        m = performance_metrics(r)
        agg_rows[SHORT.get(name, name)] = {
            "sharpe": m["sharpe"],
            "cagr%": m["cagr"] * 100,
            "vol%": m["vol"] * 100,
            "mdd%": m["max_drawdown"] * 100,
            "calmar": m["calmar"],
        }
    agg = pd.DataFrame(agg_rows).T.round(2)
    agg = agg.reindex([SHORT.get(c, c) for c in ORDER if SHORT.get(c, c) in agg.index])
    print(agg.to_string())
    print(f"\n  OOS-Zeitraum: {min(r.index.min() for r in streams.values()).date()}"
          f" .. {max(r.index.max() for r in streams.values()).date()}")

    print("\n" + "=" * 100)
    print("JAHRESWEISE, NUR OOS-DATEN (Sharpe / CAGR% / MDD%)")
    print("=" * 100)
    print("\n--- Jahres-Sharpe ---")
    print(annual_table(streams, "sharpe").round(2).to_string())
    print("\n--- Jahres-Rendite % ---")
    print((annual_table(streams, "total_return") * 100).round(1).to_string())
    print("\n--- Jahres-Max-Drawdown % ---")
    print((annual_table(streams, "max_drawdown") * 100).round(1).to_string())

    print("\n" + "=" * 100)
    print("FOKUS 2008 UND 2022 (die beiden Stresstests)")
    print("=" * 100)
    for year in (2008, 2022):
        rows = {}
        for name, r in streams.items():
            chunk = r[r.index.year == year]
            if chunk.empty:
                continue
            m = performance_metrics(chunk)
            rows[SHORT.get(name, name)] = {
                "sharpe": round(m["sharpe"], 2),
                "rendite%": round(m["total_return"] * 100, 1),
                "mdd%": round(m["max_drawdown"] * 100, 1),
            }
        df = pd.DataFrame(rows).T
        df = df.reindex([SHORT.get(c, c) for c in ORDER if SHORT.get(c, c) in df.index])
        print(f"\n--- {year} ---")
        print(df.to_string())

    print("\n" + "=" * 100)
    print("GEWAEHLTE PARAMETER JE BLOCK (Auswahl nach Trainings-Calmar)")
    print("=" * 100)
    piv = chosen.pivot_table(index="block", columns="strategy", values="params",
                             aggfunc="first", sort=False)
    keep = [c for c in ORDER if c in piv.columns]
    print(piv[keep].rename(columns=SHORT).to_string())

    print("\n" + "=" * 100)
    print("ROBUSTHEIT: Auswahl nach Trainings-SHARPE statt Trainings-Calmar")
    print("=" * 100)
    table_s, streams_s, _ = walk_forward(closes, families, static, selection_metric="sharpe")
    print("\n--- OOS-Sharpe je Block ---")
    print(pivot_block_table(table_s, "sharpe").round(2).to_string())
    agg_s = {}
    for name, r in streams_s.items():
        m = performance_metrics(r)
        agg_s[SHORT.get(name, name)] = {
            "sharpe": round(m["sharpe"], 2),
            "cagr%": round(m["cagr"] * 100, 2),
            "mdd%": round(m["max_drawdown"] * 100, 1),
            "calmar": round(m["calmar"], 2),
        }
    df_s = pd.DataFrame(agg_s).T
    df_s = df_s.reindex([SHORT.get(c, c) for c in ORDER if SHORT.get(c, c) in df_s.index])
    print("\n--- Aggregat ---")
    print(df_s.to_string())

    print("\n" + "=" * 100)
    print("SENSITIVITAET: Cash verzinst (3M-T-Bill ^IRX) - hilft allen Strategien,")
    print("die Exposure reduzieren, und ist die realistischere Rechnung")
    print("=" * 100)
    try:
        irx = fetch_ohlcv("^IRX", HISTORY_DAYS, cache_dir=CACHE_DIR)["close"]
        rf_daily = (irx / 100.0 / TRADING_DAYS_PER_YEAR).sort_index()
        # Exposure-Serien je Strategie aus den OOS-Bloecken rekonstruieren
        exposures: dict[str, pd.Series] = {}
        year = FIRST_OOS_YEAR
        last_year = int(closes.index[-1].year)
        chosen_map = {(r.block, r.strategy): r.params for r in chosen.itertuples()}
        while year <= last_year:
            b_end = min(year + OOS_BLOCK_YEARS - 1, last_year)
            label = f"{year}-{b_end % 100:02d}"
            o_start, o_end = pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{b_end}-12-31")
            for fam, cands in families.items():
                key = chosen_map[(label, fam)]
                res = _run_window(closes, *cands[key], o_start, o_end)
                exposures.setdefault(fam, []).append(res.gross_exposure)
            for name, (w, reb) in static.items():
                res = _run_window(closes, w, reb, o_start, o_end)
                exposures.setdefault(name, []).append(res.gross_exposure)
            year += OOS_BLOCK_YEARS
        exp_series = {k: pd.concat(v).sort_index() for k, v in exposures.items()}
        adj = cash_adjusted(streams, exp_series, rf_daily)
        rows = {}
        for name, r in adj.items():
            m = performance_metrics(r)
            rows[SHORT.get(name, name)] = {
                "sharpe": round(m["sharpe"], 2),
                "cagr%": round(m["cagr"] * 100, 2),
                "mdd%": round(m["max_drawdown"] * 100, 1),
                "calmar": round(m["calmar"], 2),
            }
        df = pd.DataFrame(rows).T
        df = df.reindex([SHORT.get(c, c) for c in ORDER if SHORT.get(c, c) in df.index])
        print(df.to_string())
        print("\n--- 2022 mit Cash-Zins ---")
        r22 = {}
        for name, r in adj.items():
            chunk = r[r.index.year == 2022]
            if chunk.empty:
                continue
            m = performance_metrics(chunk)
            r22[SHORT.get(name, name)] = {
                "sharpe": round(m["sharpe"], 2),
                "rendite%": round(m["total_return"] * 100, 1),
                "mdd%": round(m["max_drawdown"] * 100, 1),
            }
        d22 = pd.DataFrame(r22).T
        d22 = d22.reindex([SHORT.get(c, c) for c in ORDER if SHORT.get(c, c) in d22.index])
        print(d22.to_string())
    except Exception as exc:  # pragma: no cover - Datenquelle optional
        print(f"  ^IRX nicht verfuegbar, Sensitivitaet uebersprungen: {exc}")

    # --- Ist der Vorteil konsistent oder eine einzelne Krise? --------------
    print("\n" + "=" * 100)
    print("KONSISTENZ: Siege je OOS-Block und Aggregat OHNE die Finanzkrise")
    print("=" * 100)
    sh = table.pivot_table(index="block", columns="strategy", values="sharpe", sort=False)
    ca = table.pivot_table(index="block", columns="strategy", values="calmar", sort=False)
    n = len(sh)
    for a in ("combo_cash", "combo_redistr", "trend_only_cash"):
        for b in ("alloc_vt", "single_trend", "B&H SPY"):
            print(
                f"  {SHORT[a]:16s} vs {SHORT[b]:14s}: "
                f"Sharpe-Siege {int((sh[a] > sh[b]).sum())}/{n}, "
                f"Calmar-Siege {int((ca[a] > ca[b]).sum())}/{n}"
            )
    rows = {}
    for name, r in streams.items():
        m = performance_metrics(r[r.index.year >= 2010])
        rows[SHORT.get(name, name)] = {
            "sharpe": round(m["sharpe"], 2),
            "cagr%": round(m["cagr"] * 100, 2),
            "mdd%": round(m["max_drawdown"] * 100, 1),
            "calmar": round(m["calmar"], 2),
        }
    df = pd.DataFrame(rows).T
    df = df.reindex([SHORT.get(c, c) for c in ORDER if SHORT.get(c, c) in df.index])
    print("\n--- Aggregat ohne 2008/2009 (ab 2010) ---")
    print(df.to_string())

    # --- Andere Universen: haelt das Ergebnis, wenn man das Universum dreht? -
    print("\n" + "=" * 100)
    print("UNIVERSUMS-ROBUSTHEIT (gleiches Walk-Forward-Schema, anderes Universum)")
    print("=" * 100)
    for alt_universe, first_oos in (
        (["SPY", "TLT", "IEF", "GLD"], 2008),   # ohne QQQ
        (["SPY", "TLT", "IEF"], 2005),          # ab 2002 -> 3 Jahre mehr OOS
    ):
        alt_closes = load_closes(alt_universe, HISTORY_DAYS, cache_dir=CACHE_DIR)
        alt_ohlc = load_ohlc(alt_universe)
        alt_fams = build_candidates(alt_closes, alt_ohlc, alt_universe)
        alt_never = rebalance_flags(alt_closes.index, "never")
        alt_static = {"B&H SPY": (weights_buy_and_hold(alt_closes, "SPY"), alt_never)}
        saved, FIRST_OOS_YEAR = FIRST_OOS_YEAR, first_oos
        alt_table, alt_streams, _ = walk_forward(
            alt_closes, alt_fams, alt_static, selection_metric="calmar"
        )
        FIRST_OOS_YEAR = saved
        print(
            f"\n--- {'/'.join(alt_universe)} | {alt_closes.index.min().date()}"
            f" .. {alt_closes.index.max().date()} | OOS ab {first_oos} ---"
        )
        print("OOS-Sharpe je Block:")
        print(pivot_block_table(alt_table, "sharpe").round(2).to_string())
        print("OOS-Calmar je Block:")
        print(pivot_block_table(alt_table, "calmar").round(2).to_string())
        rows = {}
        for name, r in alt_streams.items():
            m = performance_metrics(r)
            m2 = performance_metrics(r[r.index.year >= 2010])
            rows[SHORT.get(name, name)] = {
                "sharpe": round(m["sharpe"], 2),
                "cagr%": round(m["cagr"] * 100, 2),
                "mdd%": round(m["max_drawdown"] * 100, 1),
                "calmar": round(m["calmar"], 2),
                "sharpe_ab2010": round(m2["sharpe"], 2),
                "calmar_ab2010": round(m2["calmar"], 2),
            }
        df = pd.DataFrame(rows).T
        df = df.reindex([SHORT.get(c, c) for c in ORDER if SHORT.get(c, c) in df.index])
        print("Aggregat:")
        print(df.to_string())


if __name__ == "__main__":
    main()
