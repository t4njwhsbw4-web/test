"""Quantitative Risikoanalyse: Was verlangt "Kapital in 60 Tagen verdoppeln"?

Vier Teile:
  1. Analytisch: welcher taegliche Log-Drift / welcher Sharpe ist noetig, damit
     P(Verdoppeln in 60 Tagen) > 50 % - je Volatilitaetsniveau, gemessen an
     echten Daten statt angenommen.
  2. Monte Carlo mit Block-Bootstrap auf echten Tagesrenditen (Blocklaenge 7,
     Sensitivitaet 5/10), je Asset und Positionsgroesse/Hebel.
  3. Dieselbe Simulation mit demeanten Renditen (Drift = 0): wie viel der
     Verdoppelungschance haengt nur an der historischen Aufwaertsdrift?
  4. Kelly: optimale Groesse bei einem hypothetischen kleinen Edge und
     wie schnell Overbetting den Median-Endwert zerstoert.

Alles mit festem Seed. Keine Kosten, keine Slippage, keine Funding-Gebuehren
fuer Hebel - die Zahlen sind also die optimistische Obergrenze.

Aufruf:  python -m src.research.risk_montecarlo
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from src.research.risk_data import GROUPS, describe, load_returns

SEED = 20260917
HORIZON = 60          # Kalendertage (Krypto handelt 365 Tage)
N_PATHS = 20_000
BLOCK = 7
LEVERAGES = (0.25, 0.50, 1.00, 2.00, 3.00)
TRADING_DAYS = 365
LN2 = np.log(2.0)


# ---------------------------------------------------------------- Teil 1
def required_edge(vol_ann: float, horizon: int = HORIZON, p_target: float = 0.50) -> dict:
    """Noetiger Drift/Sharpe, damit P(Endkapital >= 2x) = p_target.

    Lognormal-Modell: ln(W_T) ~ N(mu_log * T, sigma_d^2 * T).
    P(W_T >= 2) = p  <=>  mu_log * T = ln2 + z_p * sigma_d * sqrt(T),
    mit z_p = Phi^-1(p) (fuer p = 0.5 ist z = 0, also Median = 2).
    """
    sigma_d = vol_ann / np.sqrt(TRADING_DAYS)
    z = stats.norm.ppf(p_target)
    mu_log_d = (LN2 + z * sigma_d * np.sqrt(horizon)) / horizon
    mu_arith_d = mu_log_d + 0.5 * sigma_d**2          # Vol-Drag zurueckrechnen
    sharpe_ann = mu_arith_d * TRADING_DAYS / (sigma_d * np.sqrt(TRADING_DAYS))
    # Was passiert bei genau diesem Drift auf der Verlustseite?
    p_half = stats.norm.cdf((-LN2 - mu_log_d * horizon) / (sigma_d * np.sqrt(horizon)))
    p_80 = stats.norm.cdf((np.log(0.2) - mu_log_d * horizon) / (sigma_d * np.sqrt(horizon)))
    return {
        "Vol p.a.": vol_ann,
        "Tagesvol": sigma_d,
        "noetige Tagesrendite (log)": mu_log_d,
        "noetige Tagesrendite (arith)": mu_arith_d,
        "noetige Rendite p.a. (arith)": mu_arith_d * TRADING_DAYS,
        "noetiger Sharpe p.a.": sharpe_ann,
        "dabei P(halbieren)": p_half,
        "dabei P(-80%)": p_80,
    }


def p_double_given_sharpe(sharpe_ann: float, vol_ann: float, horizon: int = HORIZON) -> dict:
    """Umgekehrt: welche Chancen bietet ein real erreichbarer Sharpe?"""
    sigma_d = vol_ann / np.sqrt(TRADING_DAYS)
    mu_arith_d = sharpe_ann * sigma_d / np.sqrt(TRADING_DAYS)
    mu_log_d = mu_arith_d - 0.5 * sigma_d**2
    s = sigma_d * np.sqrt(horizon)
    m = mu_log_d * horizon
    return {
        "P(verdoppeln)": 1 - stats.norm.cdf((LN2 - m) / s),
        "P(halbieren)": stats.norm.cdf((-LN2 - m) / s),
        "Median-Endkapital": float(np.exp(m)),
        "Tage bis Median 2x": LN2 / mu_log_d if mu_log_d > 0 else np.inf,
    }


# ---------------------------------------------------------------- Teil 2/3
def block_bootstrap_paths(
    returns: np.ndarray, n_paths: int, horizon: int, block: int, rng: np.random.Generator
) -> np.ndarray:
    """Zirkulaerer Block-Bootstrap: (n_paths, horizon) gezogene Tagesrenditen.

    Ganze Bloecke aufeinanderfolgender Tage erhalten Vol-Clustering und die
    Abfolge von Crash-Tagen. I.i.d.-Ziehen wuerde beides zerstoeren und das
    Risiko systematisch unterschaetzen.
    """
    n = len(returns)
    n_blocks = int(np.ceil(horizon / block))
    starts = rng.integers(0, n, size=(n_paths, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % n
    return returns[idx.reshape(n_paths, -1)[:, :horizon]]


def simulate(
    returns: np.ndarray,
    leverage: float,
    n_paths: int = N_PATHS,
    horizon: int = HORIZON,
    block: int = BLOCK,
    seed: int = SEED,
) -> dict:
    """Taeglich rebalancierte konstante Positionsgroesse `leverage`.

    Faellt das Eigenkapital an einem Tag auf <= 0 (moeglich ab Hebel > 1),
    bleibt der Pfad bei 0 - das ist die Liquidation/Totalverlust.
    """
    rng = np.random.default_rng(seed)
    sampled = block_bootstrap_paths(returns, n_paths, horizon, block, rng)
    step = np.maximum(1.0 + leverage * sampled, 0.0)
    equity = np.cumprod(step, axis=1)
    final = equity[:, -1]
    min_equity = equity.min(axis=1)
    p_double = float((final >= 2.0).mean())
    p_half = float((final <= 0.5).mean())
    return {
        "P(2x)": p_double,
        "P(0.5x)": p_half,
        "P(-80%)": float((final <= 0.2).mean()),
        "P(Ruin <1%)": float((min_equity <= 0.01).mean()),
        "Median": float(np.median(final)),
        "Mittelwert": float(final.mean()),
        "5%-Quantil": float(np.quantile(final, 0.05)),
        "95%-Quantil": float(np.quantile(final, 0.95)),
        "P(2x)/P(0.5x)": p_double / p_half if p_half > 0 else np.inf,
    }


def run_grid(returns: dict[str, pd.Series], demean: bool, block: int = BLOCK) -> pd.DataFrame:
    rows = []
    for name, ser in returns.items():
        r = ser.to_numpy(dtype=float)
        if demean:
            r = r - r.mean()          # arithmetischer Erwartungswert = 0
        for lev in LEVERAGES:
            res = simulate(r, lev, block=block)
            rows.append({"Asset": name, "Gruppe": GROUPS.get(name, "?"), "Groesse": lev, **res})
    return pd.DataFrame(rows)


def empirical_windows(returns: dict[str, pd.Series], horizon: int = HORIZON) -> pd.DataFrame:
    """Realitaets-Check: wie oft hat sich das Asset historisch in 60 Tagen
    tatsaechlich verdoppelt/halbiert (ueberlappende Fenster, ungehebelt)?

    Wichtig als Gegenprobe zum Bootstrap: 7-Tage-Bloecke zerschneiden
    mehrmonatige Bullenlaeufe, der Bootstrap kann P(2x) also unterschaetzen.
    """
    rows = []
    for name, ser in returns.items():
        g = (1 + ser).to_numpy(dtype=float)
        lg = np.log(g)
        c = np.cumsum(lg)
        w = np.exp(c[horizon:] - c[:-horizon])
        rows.append(
            {
                "Asset": name,
                "Fenster": len(w),
                "P(2x) hist": float((w >= 2).mean()),
                "P(0.5x) hist": float((w <= 0.5).mean()),
                "Median hist": float(np.median(w)),
                "Bestes": float(w.max()),
                "Schlechtestes": float(w.min()),
            }
        )
    return pd.DataFrame(rows).set_index("Asset")


def max_pdouble_sweep(returns: dict[str, pd.Series], demean: bool = False) -> pd.DataFrame:
    """Feiner Hebel-Sweep: wo liegt das Maximum von P(2x) - und was kostet es?"""
    rows = []
    for name in ("BTC", "SOL", "PEPE"):
        if name not in returns:
            continue
        r = returns[name].to_numpy(dtype=float)
        if demean:
            r = r - r.mean()
        for lev in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0):
            res = simulate(r, lev)
            rows.append({"Asset": name, "Groesse": lev, **res})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- Teil 4
def kelly_analysis(returns: np.ndarray, sharpe_ann: float, label: str) -> pd.DataFrame:
    """Hypothetischer Edge: Renditen demeanen und Drift mit Ziel-Sharpe einsetzen.

    Vergleicht die Gauss-Kelly-Formel f* = mu/sigma^2 mit dem empirischen
    Optimum von E[log(1 + f*r)] auf der echten (fat-tailed) Verteilung.
    """
    r0 = returns - returns.mean()
    sigma_d = r0.std(ddof=1)
    mu_d = sharpe_ann * sigma_d / np.sqrt(TRADING_DAYS)
    r = r0 + mu_d
    f_gauss = mu_d / sigma_d**2

    grid = np.concatenate([np.arange(0.25, 8.01, 0.25), [f_gauss]])
    rows = []
    rng = np.random.default_rng(SEED + 1)
    sampled = block_bootstrap_paths(r, N_PATHS, HORIZON, BLOCK, rng)
    for f in np.sort(np.unique(np.round(grid, 4))):
        step = np.maximum(1.0 + f * r, 0.0)
        with np.errstate(divide="ignore"):
            g_emp = np.mean(np.where(step > 0, np.log(np.maximum(step, 1e-300)), -50.0))
        g_gauss = f * mu_d - 0.5 * f**2 * sigma_d**2
        path_step = np.maximum(1.0 + f * sampled, 0.0)
        final = np.cumprod(path_step, axis=1)[:, -1]
        rows.append(
            {
                "Asset": label,
                "f (Anteil/Hebel)": f,
                "g_emp (log/Tag)": g_emp,
                "g_gauss (log/Tag)": g_gauss,
                "Median 60T": float(np.median(final)),
                "P(2x)": float((final >= 2).mean()),
                "P(0.5x)": float((final <= 0.5).mean()),
                "P(Ruin)": float((final <= 0.01).mean()),
            }
        )
    df = pd.DataFrame(rows)
    df.attrs["f_gauss"] = f_gauss
    df.attrs["mu_d"] = mu_d
    df.attrs["sigma_d"] = sigma_d
    return df


# ---------------------------------------------------------------- Report
def _fmt(df: pd.DataFrame, pct_cols=(), digits=3) -> str:
    out = df.copy()
    for c in pct_cols:
        if c in out.columns:
            out[c] = (out[c] * 100).map(lambda v: f"{v:5.1f}%")
    return out.to_string(float_format=lambda x: f"{x:,.{digits}f}")


def main() -> None:
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 200)
    rets = load_returns()
    stats_df = describe(rets)

    print("=" * 110)
    print("DATENBASIS (echte Tagesbars; yfinance + data-api.binance.vision)")
    print("=" * 110)
    print(_fmt(stats_df.drop(columns=["Tagesvol"])))

    print("\n" + "=" * 110)
    print("TEIL 1 - WAS 'VERDOPPELN IN 60 TAGEN MIT P>50%' RECHNERISCH VERLANGT")
    print("=" * 110)
    levels = {
        f"{n} (gemessen {stats_df.loc[n, 'Vol p.a.']:.0%})": stats_df.loc[n, "Vol p.a."]
        for n in ("BTC", "ETH", "SOL", "PEPE", "DOGE")
        if n in stats_df.index
    }
    levels["Aktien-Portfolio (20%)"] = 0.20
    req = pd.DataFrame([required_edge(v) for v in levels.values()], index=list(levels))
    print(_fmt(req[["Vol p.a.", "noetige Tagesrendite (log)", "noetige Rendite p.a. (arith)",
                    "noetiger Sharpe p.a.", "dabei P(halbieren)", "dabei P(-80%)"]],
               pct_cols=("dabei P(halbieren)", "dabei P(-80%)")))

    print("\nReferenz: was Sharpe-Werte real bedeuten (Chancen ueber 60 Tage, BTC-Vol "
          f"{stats_df.loc['BTC', 'Vol p.a.']:.0%}):")
    ref = {
        "Unsere beste Dev-Strategie (1.10)": 1.10,
        "Dieselbe im Hold-out (-1.10)": -1.10,
        "BTC Buy-and-Hold (hist.)": float(stats_df.loc["BTC", "Sharpe (hist)"]),
        "Guter Hedgefonds (1.0)": 1.00,
        "Weltklasse dauerhaft (2.0)": 2.00,
        "Renaissance Medallion-Klasse (3.0)": 3.00,
    }
    ref_df = pd.DataFrame(
        [p_double_given_sharpe(s, float(stats_df.loc["BTC", "Vol p.a."])) for s in ref.values()],
        index=[f"{k}" for k in ref],
    )
    ref_df.insert(0, "Sharpe", list(ref.values()))
    print(_fmt(ref_df, pct_cols=("P(verdoppeln)", "P(halbieren)"), digits=2))

    print("\n" + "=" * 110)
    print(f"TEIL 2 - MONTE CARLO, ECHTE RENDITEN (Block-Bootstrap L={BLOCK}, "
          f"{N_PATHS:,} Pfade, {HORIZON} Tage, Seed {SEED})")
    print("=" * 110)
    real = run_grid(rets, demean=False)
    print(_fmt(real, pct_cols=("P(2x)", "P(0.5x)", "P(-80%)", "P(Ruin <1%)")))

    print("\n" + "=" * 110)
    print("TEIL 3 - DIESELBE SIMULATION MIT DRIFT = 0 (Renditen demeaned)")
    print("=" * 110)
    zero = run_grid(rets, demean=True)
    print(_fmt(zero, pct_cols=("P(2x)", "P(0.5x)", "P(-80%)", "P(Ruin <1%)")))

    print("\nDIREKTVERGLEICH P(2x) mit vs. ohne historische Drift:")
    cmp = real.merge(zero, on=["Asset", "Groesse"], suffixes=("_real", "_drift0"))
    cmp["Anteil aus Drift"] = 1 - cmp["P(2x)_drift0"] / cmp["P(2x)_real"].replace(0, np.nan)
    print(_fmt(cmp[["Asset", "Groesse", "P(2x)_real", "P(2x)_drift0", "Anteil aus Drift",
                    "P(0.5x)_real", "P(0.5x)_drift0", "Median_real", "Median_drift0"]],
               pct_cols=("P(2x)_real", "P(2x)_drift0", "Anteil aus Drift",
                         "P(0.5x)_real", "P(0.5x)_drift0")))

    print("\nREALITAETS-CHECK: historische ueberlappende 60-Tage-Fenster (ungehebelt, 1.0x)")
    print("Der Bootstrap zerschneidet mehrmonatige Trends - diese Zahlen zeigen, "
          "wie stark das P(2x) nach unten verzerrt.")
    print(_fmt(empirical_windows(rets), pct_cols=("P(2x) hist", "P(0.5x) hist")))

    print("\nSENSITIVITAET Blocklaenge (BTC/SOL/PEPE, Groesse 1.0 und 3.0):")
    sens = []
    for b in (1, 5, 7, 10, 20, 30, 60):
        g = run_grid({k: v for k, v in rets.items() if k in ("BTC", "SOL", "PEPE")},
                     demean=False, block=b)
        g = g[g["Groesse"].isin([1.0, 3.0])].copy()
        g["Block"] = b
        sens.append(g)
    sens_df = pd.concat(sens)[["Block", "Asset", "Groesse", "P(2x)", "P(0.5x)", "P(-80%)",
                               "P(Ruin <1%)", "Median"]].sort_values(["Asset", "Groesse", "Block"])
    print(_fmt(sens_df, pct_cols=("P(2x)", "P(0.5x)", "P(-80%)", "P(Ruin <1%)")))

    print("\n" + "=" * 110)
    print("TEIL 4 - KELLY: OPTIMALE GROESSE BEI HYPOTHETISCHEM KLEINEM EDGE (Sharpe 0.5 p.a.)")
    print("=" * 110)
    for asset in ("BTC", "SOL"):
        if asset not in rets:
            continue
        k = kelly_analysis(rets[asset].to_numpy(dtype=float), sharpe_ann=0.5, label=asset)
        f_g = k.attrs["f_gauss"]
        best = k.loc[k["g_emp (log/Tag)"].idxmax()]
        print(f"\n{asset}: Tagesvol {k.attrs['sigma_d']:.4f}, unterstellter Drift "
              f"{k.attrs['mu_d']*100:.4f}%/Tag (Sharpe 0.5)")
        print(f"  Kelly (Gauss-Formel mu/sigma^2) = {f_g:.2f}x   |   "
              f"empirisches Optimum auf echten Fat Tails = {best['f (Anteil/Hebel)']:.2f}x")
        show = k[k["f (Anteil/Hebel)"].isin(
            [0.25, 0.5, 1.0, np.round(f_g, 4), 2.0, 3.0, 4.0, 5.0, 6.0, 8.0])]
        print(_fmt(show[["f (Anteil/Hebel)", "g_emp (log/Tag)", "g_gauss (log/Tag)", "Median 60T",
                         "P(2x)", "P(0.5x)", "P(Ruin)"]],
                   pct_cols=("P(2x)", "P(0.5x)", "P(Ruin)"), digits=4))

    print("\n" + "=" * 110)
    print("MEHR RISIKO = MEHR GEWINN? Feiner Hebel-Sweep, Maximum von P(2x) (echte Daten)")
    print("=" * 110)
    sweep = max_pdouble_sweep(rets)
    print(_fmt(sweep[["Asset", "Groesse", "P(2x)", "P(0.5x)", "P(-80%)", "P(Ruin <1%)",
                      "Median", "Mittelwert"]],
               pct_cols=("P(2x)", "P(0.5x)", "P(-80%)", "P(Ruin <1%)")))
    print("\nMaximum von P(2x) je Asset:")
    print(_fmt(sweep.loc[sweep.groupby("Asset")["P(2x)"].idxmax()]
               [["Asset", "Groesse", "P(2x)", "P(0.5x)", "P(-80%)", "Median"]],
               pct_cols=("P(2x)", "P(0.5x)", "P(-80%)")))
    sweep0 = max_pdouble_sweep(rets, demean=True)
    print("\nDasselbe Maximum mit Drift = 0 (so viel bleibt ohne historische Aufwaertsdrift):")
    print(_fmt(sweep0.loc[sweep0.groupby("Asset")["P(2x)"].idxmax()]
               [["Asset", "Groesse", "P(2x)", "P(0.5x)", "P(-80%)", "P(Ruin <1%)", "Median"]],
               pct_cols=("P(2x)", "P(0.5x)", "P(-80%)", "P(Ruin <1%)")))

    print("\nROBUSTHEIT: DOGE ohne den einzelnen +355%-Tag (2021-01-28, WSB-Squeeze)")
    doge = rets["DOGE"]
    trimmed = doge.drop(doge.idxmax()).to_numpy(dtype=float)
    for lev in (0.5, 1.0, 2.0):
        a = simulate(doge.to_numpy(dtype=float), lev)
        b = simulate(trimmed, lev)
        print(f"  {lev:.2f}x  P(2x): {a['P(2x)']:.1%} -> {b['P(2x)']:.1%} | "
              f"P(0.5x): {a['P(0.5x)']:.1%} -> {b['P(0.5x)']:.1%} | "
              f"Median: {a['Median']:.3f} -> {b['Median']:.3f}")

    print("\n" + "=" * 110)
    print("BESTER KOMPROMISS: Frontier - hoechstes P(2x) unter Verlust-Nebenbedingungen")
    print("(echte Daten; Verhaeltnis nur aussagekraeftig, wenn P(0.5x) nicht ~0 ist)")
    print("=" * 110)
    for cap in (0.05, 0.10, 0.20):
        ok = real[real["P(0.5x)"] <= cap]
        if ok.empty:
            continue
        best = ok.loc[ok["P(2x)"].idxmax()]
        print(f"\n  Nebenbedingung P(halbieren) <= {cap:.0%}  ->  "
              f"{best['Asset']} @ {best['Groesse']:.2f}x : "
              f"P(2x)={best['P(2x)']:.1%}, P(0.5x)={best['P(0.5x)']:.1%}, "
              f"P(-80%)={best['P(-80%)']:.1%}, Median={best['Median']:.3f}")
        print(_fmt(ok.sort_values("P(2x)", ascending=False).head(5)
                   [["Asset", "Groesse", "P(2x)", "P(0.5x)", "P(-80%)", "Median",
                     "P(2x)/P(0.5x)"]],
                   pct_cols=("P(2x)", "P(0.5x)", "P(-80%)")))


if __name__ == "__main__":
    main()
