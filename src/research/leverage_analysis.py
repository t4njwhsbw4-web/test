"""Gehebeltes Vol-Targeting: wie weit traegt der Hebel, und was kostet er?

Ausgangslage
------------
Von ~1700 getesteten Kombinationen hat genau eine Strategie ihren Vorteil
out-of-sample gehalten: `Equal-Weight + VolTarget` aus `src/research/multi_asset.py`
(Dev-Sharpe 1.10 2004-2018 -> Hold-out-Sharpe 1.24 2019-2026, CAGR 8.0%,
MDD -12.2%). Hoher Sharpe bei kleinem Drawdown ist der EINZIGE legitime
Kandidat fuer Hebel. Diese Datei rechnet aus, wie weit man gehen kann.

Was hier NICHT passiert
-----------------------
Der Hebel wird nicht optimiert und dann als "validiert" verkauft. Hebel ist
kein Parameter, den man an Daten fittet, sondern eine Risikoentscheidung. Der
Hold-out 2019-2026 wurde fuer diese Strategie bereits einmal verbraucht; er
wird hier nur noch als Teilperiode BERICHTET, nicht zur Auswahl benutzt.
Alle Kennzahlen kommen primaer ueber den Gesamtzeitraum, zusaetzlich getrennt
pro Teilperiode und fuer die Stressjahre 2008 / 2020 / 2022.

Was explizit modelliert wird
----------------------------
1. Finanzierungskosten auf den geliehenen Teil: variabler Satz = 13-Wochen-
   T-Bill (^IRX) + Broker-Aufschlag. Historische Zinswende ist entscheidend -
   Hebel kostete 2009-2021 fast nichts und kostet seit 2023 6-8%.
2. Cash-Verzinsung des nicht investierten Teils (die Strategie haelt im
   Schnitt ~22-35% Cash), ebenfalls ^IRX.
3. Pfadabhaengigkeit: taegliche Rendite x Hebel, taeglich kumuliert. Der
   Volatilitaets-Drag (-L^2/2 * sigma^2) entsteht dadurch von selbst.
4. Maintenance Margin / Zwangsliquidation: der Pfad ENDET, wenn die
   Eigenkapitalquote unter die Wartungsmarge faellt. Renditen danach sind
   irrelevant und werden nicht mitgezaehlt.
5. Transaktionskosten skalieren mit dem Hebel (L-facher Umschlag).

Kein Look-Ahead: die Gewichte kommen unveraendert aus multi_asset.py (kausal,
ein Bar Ausfuehrungsverzoegerung); Zinsen werden mit ffill nur aus der
Vergangenheit fortgeschrieben.
"""
from __future__ import annotations

import dataclasses
from typing import Sequence

import numpy as np
import pandas as pd

from src.data.fetch import fetch_ohlcv
from src.research.multi_asset import (
    CLASSIC_UNIVERSE,
    COST_BPS,
    DEFAULT_COST_BPS,
    DEV_END,
    HOLDOUT_START,
    TRADING_DAYS_PER_YEAR,
    apply_vol_target,
    load_closes,
    performance_metrics,
    rebalance_flags,
    weights_buy_and_hold,
    weights_equal,
)

RNG_SEED = 42

# --- Fixierte Parameter der ueberlebenden Strategie (aus multi_asset.py) ---
VT_LOOKBACK = 21
TARGET_VOL = 0.06
BASE_NAME = "Equal-Weight + VolTarget"

# --- Hebelstufen ---
LEVERAGE_GRID = (1.0, 1.5, 2.0, 3.0, 4.0, 5.0)

# --- Broker-Annahmen ---
# Wartungsmarge: 25% ist der Reg-T-Standard fuer Aktien-Margin (FINRA-Minimum),
# viele Broker verlangen 30% ("House Requirement"). Eigenkapital/Positionswert
# muss >= m bleiben, sonst Nachschuss bzw. Zwangsliquidation. m=0.25 entspricht
# einer maximal zulaessigen Brutto-Exposure von 1/0.25 = 4.0.
MAINT_MARGIN = 0.25
MAINT_MARGIN_STRICT = 0.30

# Reg-T-Ersteinschuss 50% -> maximal 2x Brutto-Exposure beim Eroeffnen.
# Portfolio Margin (ab 100k USD, risikobasiert) erlaubt real ca. 6.7x.
REG_T_MAX_GROSS = 2.0
PORTFOLIO_MARGIN_MAX_GROSS = 6.67

# Broker-Aufschlag auf den Geldmarktsatz. IBKR liegt bei ca. +1.5% (BM+1.5),
# klassische Broker bei +2 bis +4%. 2.0% ist eine faire Mitte.
BORROW_SPREAD = 0.020
BORROW_SPREAD_LOW = 0.015
BORROW_SPREAD_HIGH = 0.030

RATE_SYMBOL = "^IRX"  # 13-Wochen-T-Bill, in Prozent quotiert


# ---------------------------------------------------------------------------
# Zinsen
# ---------------------------------------------------------------------------
def load_short_rate(index: pd.DatetimeIndex, cache_dir: str = "data_cache") -> pd.Series:
    """Kurzfristiger risikofreier Satz p.a. als Dezimalzahl, auf `index` gelegt.

    ^IRX ist in Prozent quotiert. Nur ffill (Vergangenheit), am Anfang bfill
    als Notnagel - das betrifft hoechstens die ersten Tage.
    """
    raw = fetch_ohlcv(RATE_SYMBOL, 12000, cache_dir=cache_dir)["close"].sort_index()
    rate = (raw / 100.0).clip(lower=0.0)
    out = rate.reindex(rate.index.union(index)).ffill().reindex(index)
    return out.bfill().rename("short_rate")


# ---------------------------------------------------------------------------
# Gehebelter Portfolio-Backtest
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class LeveredResult:
    name: str
    leverage: float
    returns: pd.Series          # Tagesrenditen des Kontos, bis Liquidation
    equity_curve: pd.Series     # Kontowert, nach Liquidation konstant
    gross_exposure: pd.Series   # Positionswert / Eigenkapital (Tagesbeginn)
    financing_cost: pd.Series   # Tageskosten der Finanzierung (Anteil Equity)
    cash_income: pd.Series
    cost_drag: pd.Series
    margin_ratio: pd.Series     # Eigenkapital / Positionswert am Tagesschluss
    cap_binding: pd.Series      # True, wenn der Broker die Zielgroesse beschnitt
    short_rate: pd.Series
    liquidated: bool
    liquidation_date: pd.Timestamp | None
    liquidation_equity: float
    liquidation_cause: str

    @property
    def metrics(self) -> dict:
        r = self.returns.dropna()
        m = performance_metrics(r)
        m["sharpe_excess"] = excess_sharpe(r, self.short_rate)
        m["cap_binding_share"] = float(self.cap_binding.mean())
        # Bei Liquidation ist der Pfad kuerzer -> CAGR auf die REALE Kalenderzeit
        # des Gesamtfensters beziehen, nicht auf die ueberlebte Zeit. Sonst
        # sieht eine Liquidation nach 3 Monaten wie eine Jahresrendite aus.
        m["cagr"] = self.calendar_cagr
        m["final_equity"] = float(self.equity_curve.iloc[-1])
        m["avg_gross_exposure"] = float(self.gross_exposure.mean())
        m["max_gross_exposure"] = float(self.gross_exposure.max())
        m["financing_pa"] = float(self.financing_cost.sum() / self._years)
        m["cash_income_pa"] = float(self.cash_income.sum() / self._years)
        m["cost_pa"] = float(self.cost_drag.sum() / self._years)
        m["min_margin_ratio"] = float(self.margin_ratio.min())
        m["liquidated"] = self.liquidated
        m["liquidation_date"] = self.liquidation_date
        m["doubling_years"] = doubling_time(m["cagr"])
        mdd = m["max_drawdown"]
        m["calmar"] = m["cagr"] / abs(mdd) if mdd < -1e-12 else float("nan")
        return m

    @property
    def _years(self) -> float:
        return max(len(self.equity_curve) / TRADING_DAYS_PER_YEAR, 1e-9)

    @property
    def calendar_cagr(self) -> float:
        final = float(self.equity_curve.iloc[-1])
        if final <= 0:
            return -1.0
        return final ** (1.0 / self._years) - 1.0


def excess_sharpe(returns: pd.Series, short_rate: pd.Series) -> float:
    """Sharpe MIT risikofreiem Zins im Zaehler.

    Das Repo rechnet Sharpe sonst mit rf=0. Bei einer Strategie, die 22-35%
    Cash haelt, schenkt das im Hochzinsumfeld Sharpe: 4.3% Geldmarkt auf ein
    6%-Vol-Portfolio sind rund 0.3-0.5 Sharpe-Punkte aus dem Nichts. Diese
    Kennzahl zieht den Zins ab und ist die ehrlichere Zahl.
    """
    r = returns.dropna()
    if r.empty or float(r.std()) <= 0:
        return float("nan")
    rate = short_rate.reindex(r.index).ffill().bfill()
    excess = r - rate / TRADING_DAYS_PER_YEAR
    return float(excess.mean() / r.std() * np.sqrt(TRADING_DAYS_PER_YEAR))


def doubling_time(cagr: float) -> float:
    """Jahre bis zur Kapitalverdoppelung bei konstanter CAGR."""
    if not np.isfinite(cagr) or cagr <= 0:
        return float("inf")
    return float(np.log(2.0) / np.log(1.0 + cagr))


def run_levered_portfolio(
    closes: pd.DataFrame,
    target_weights: pd.DataFrame,
    rebalance: pd.Series,
    short_rate: pd.Series,
    leverage: float = 1.0,
    name: str = "levered",
    cost_bps: dict | None = None,
    borrow_spread: float = BORROW_SPREAD,
    maint_margin: float = MAINT_MARGIN,
    stop_on_liquidation: bool = True,
    lows: pd.DataFrame | None = None,
    max_gross: float | None = None,
) -> LeveredResult:
    """Wie `multi_asset.run_portfolio`, aber mit Hebel, Zinsen und Margin-Call.

    `target_weights` sind die UNGEHEBELTEN Zielgewichte der Strategie (Summe <=1).
    Sie werden mit `leverage` multipliziert: die Strategie behaelt ihre Signatur
    (Vol-Ziel 6%, Cash-Puffer), das Konto haelt sie nur L-fach. Der geliehene
    Betrag ist L*Exposure - 1, sofern positiv.

    Zwei getrennte Broker-Mechanismen, die oft verwechselt werden:

    1. Ersteinschuss / Groessenlimit (`max_gross`): eine Position, die das Limit
       ueberschreitet, laesst der Broker gar nicht erst eroeffnen. Das Ziel wird
       proportional beschnitten, `cap_binding` zaehlt diese Tage. Standard ist
       Portfolio Margin (6.67x); Reg-T waere 2.0x. Das ist KEINE Liquidation.
    2. Zwangsliquidation: Eigenkapital / Brutto-Positionswert faellt durch
       VERLUSTE unter `maint_margin`. Dann endet der Pfad; alle spaeteren
       Renditen sind irrelevant und werden nicht mehr gezaehlt.

    Ist `lows` gesetzt, wird zusaetzlich mit den Tagestiefs geprueft. Das ist
    eine bewusst konservative SCHRANKE (unterstellt, dass alle Assets
    gleichzeitig ihr Tagestief erreichen), deshalb nicht der Standard.
    """
    cost_bps = cost_bps or COST_BPS
    assets = list(closes.columns)
    asset_returns = closes.pct_change().fillna(0.0)

    w_target = target_weights.reindex(index=closes.index, columns=assets).fillna(0.0)
    if (w_target.values < -1e-12).any():
        raise ValueError("Negative Gewichte - Long-only ist verletzt.")
    w_target = w_target * leverage

    rebalance = rebalance.reindex(closes.index).fillna(False).astype(bool)
    exec_flag = rebalance.shift(1, fill_value=False)
    exec_flag.iloc[0] = False
    w_source = w_target.shift(1).fillna(0.0)

    bps = np.array([cost_bps.get(a, DEFAULT_COST_BPS) for a in assets]) / 10_000.0
    rate = short_rate.reindex(closes.index).ffill().bfill().values
    cash_daily = rate / TRADING_DAYS_PER_YEAR
    borrow_daily = (rate + borrow_spread) / TRADING_DAYS_PER_YEAR

    low_ret = None
    if lows is not None:
        # Tagestief relativ zum Vortagesschluss - konservative Intraday-Schranke.
        low_ret = (lows.reindex(index=closes.index, columns=assets) / closes.shift(1) - 1.0)
        low_ret = low_ret.fillna(0.0).clip(upper=0.0).values

    gross_cap = (
        float(max_gross) if max_gross is not None else PORTFOLIO_MARGIN_MAX_GROSS
    )

    n = len(closes)
    port_ret = np.zeros(n)
    gross_exp = np.zeros(n)
    fin_cost = np.zeros(n)
    cash_inc = np.zeros(n)
    costs = np.zeros(n)
    margin = np.ones(n)
    equity = np.ones(n)
    capped = np.zeros(n, dtype=bool)

    w_prev_end = np.zeros(len(assets))
    eq = 1.0
    r_mat = asset_returns.values
    w_src_mat = w_source.values
    exec_mat = exec_flag.values

    liquidated = False
    liq_date: pd.Timestamp | None = None
    liq_equity = float("nan")
    liq_cause = ""
    last_t = n - 1

    for t in range(n):
        if exec_mat[t]:
            w_start = w_src_mat[t].copy()
            want = float(w_start.sum())
            if want > gross_cap + 1e-12:
                # Der Broker laesst diese Groesse nicht zu -> proportional kuerzen.
                w_start = w_start * (gross_cap / want)
                capped[t] = True
            trade = np.abs(w_start - w_prev_end)
            costs[t] = float((trade * bps).sum())
        else:
            w_start = w_prev_end

        invested = float(w_start.sum())
        gross_exp[t] = invested
        cash_w = 1.0 - invested
        if cash_w >= 0.0:
            cash_leg = cash_w * cash_daily[t]
            cash_inc[t] = cash_leg
        else:
            cash_leg = cash_w * borrow_daily[t]  # negativ = Kosten
            fin_cost[t] = -cash_leg

        gross = float((w_start * r_mat[t]).sum()) + cash_leg
        net = gross - costs[t]
        port_ret[t] = net
        eq_new = eq * (1.0 + net)
        equity[t] = max(eq_new, 0.0)

        # Positionswert am Schluss, relativ zum Eigenkapital am Schluss.
        pos_end = float((w_start * (1.0 + r_mat[t])).sum())
        growth = 1.0 + net
        if growth <= 0.0 or eq_new <= 0.0:
            margin[t] = 0.0
            w_prev_end = np.zeros(len(assets))
        else:
            w_prev_end = w_start * (1.0 + r_mat[t]) / growth
            end_exposure = pos_end / growth
            margin[t] = 1.0 / end_exposure if end_exposure > 1e-12 else np.inf

        eq = max(eq_new, 0.0)

        breach = margin[t] < maint_margin - 1e-12 or eq <= 0.0
        cause = "Verlust (Schlusskurs)"
        if not breach and low_ret is not None:
            worst = float((w_start * low_ret[t]).sum()) + cash_leg - costs[t]
            g_low = 1.0 + worst
            pos_low = float((w_start * (1.0 + low_ret[t])).sum())
            if g_low <= 0.0 or (pos_low / g_low) > 1.0 / maint_margin:
                breach = True
                cause = "Intraday-Tief"

        if breach and stop_on_liquidation:
            liquidated = True
            liq_date = closes.index[t]
            liq_equity = eq
            liq_cause = cause
            last_t = t
            break

    idx = closes.index[: last_t + 1]
    eq_series = pd.Series(equity[: last_t + 1], index=idx, name=name)
    if liquidated:
        # Nach der Zwangsliquidation ist das Konto flach: der Restwert bleibt
        # bis zum Ende des Fensters stehen, weitere Marktrenditen zaehlen nicht.
        tail = closes.index[last_t + 1 :]
        if len(tail):
            eq_series = pd.concat(
                [eq_series, pd.Series(eq_series.iloc[-1], index=tail, name=name)]
            )

    return LeveredResult(
        name=name,
        leverage=leverage,
        returns=pd.Series(port_ret[: last_t + 1], index=idx, name=name),
        equity_curve=eq_series,
        gross_exposure=pd.Series(gross_exp[: last_t + 1], index=idx),
        financing_cost=pd.Series(fin_cost[: last_t + 1], index=idx),
        cash_income=pd.Series(cash_inc[: last_t + 1], index=idx),
        cost_drag=pd.Series(costs[: last_t + 1], index=idx),
        margin_ratio=pd.Series(margin[: last_t + 1], index=idx),
        cap_binding=pd.Series(capped[: last_t + 1], index=idx),
        short_rate=short_rate.reindex(closes.index).ffill().bfill(),
        liquidated=liquidated,
        liquidation_date=liq_date,
        liquidation_equity=liq_equity,
        liquidation_cause=liq_cause,
    )


# ---------------------------------------------------------------------------
# 60-Tage-Fenster: Verdoppeln oder halbieren?
# ---------------------------------------------------------------------------
def rolling_window_distribution(
    returns: pd.Series,
    gross_exposure: pd.Series,
    window: int = 60,
    maint_margin: float = MAINT_MARGIN,
) -> dict:
    """Verteilung der Ergebnisse ueber ALLE rollierenden `window`-Tage-Fenster.

    Jedes Fenster wird als frisches Konto behandelt (Startkapital 1). Innerhalb
    des Fensters wird die Margin-Bedingung gepruef; bricht sie, endet das
    Fenster mit dem Liquidationswert.

    `returns` muss die Renditeserie OHNE globalen Abbruch sein, sonst gibt es
    nach dem ersten Margin-Call keine Fenster mehr.
    """
    r = returns.dropna().values
    n = len(r)
    if n < window + 1:
        return {}
    ends = np.zeros(n - window + 1)
    liq = np.zeros(n - window + 1, dtype=bool)
    for i in range(n - window + 1):
        e = 1.0
        dead = False
        for j in range(window):
            e *= 1.0 + r[i + j]
            if e <= 0.0:  # Totalverlust: Konto ist weg, Fenster endet
                e = 0.0
                dead = True
                break
        ends[i] = e - 1.0
        liq[i] = dead
    out = {
        "windows": int(len(ends)),
        "mean": float(ends.mean()),
        "median": float(np.median(ends)),
        "p05": float(np.percentile(ends, 5)),
        "p95": float(np.percentile(ends, 95)),
        "best": float(ends.max()),
        "worst": float(ends.min()),
        "p_double": float((ends >= 1.0).mean()),
        "p_half": float((ends <= -0.5).mean()),
        "p_ruin": float(liq.mean()),
        "p_positive": float((ends > 0).mean()),
    }
    return out


def window_liquidation_rate(
    result_unstopped: LeveredResult,
    window: int = 60,
    maint_margin: float = MAINT_MARGIN,
) -> float:
    """Anteil der 60-Tage-Fenster, in denen die Margin-Bedingung bricht.

    Die Margin-Quote haengt nur vom Pfad ab, nicht vom Startzeitpunkt des
    Fensters (das Konto wird taeglich auf L*w zurueckgesetzt). Ein Fenster gilt
    als liquidiert, wenn irgendein Tag darin die Wartungsmarge verletzt.
    """
    breach = (result_unstopped.margin_ratio < maint_margin - 1e-12).values
    n = len(breach)
    if n < window:
        return float("nan")
    roll = pd.Series(breach.astype(float)).rolling(window).sum().dropna().values
    return float((roll > 0).mean())


def stationary_bootstrap_double(
    returns: pd.Series,
    horizon: int = 60,
    paths: int = 20_000,
    mean_block: int = 10,
    seed: int = RNG_SEED,
) -> dict:
    """Stationaerer Bootstrap (Politis/Romano) fuer P(Verdoppeln in `horizon`).

    Die rollierenden Fenster ueberlappen stark - der empirische Anteil hat
    effektiv nur eine Handvoll unabhaengiger Beobachtungen. Der Block-Bootstrap
    erhaelt einen Teil der Vol-Cluster und gibt eine ehrlichere Streuung.
    Fixer Seed.
    """
    r = returns.dropna().values
    n = len(r)
    if n < horizon:
        return {}
    rng = np.random.default_rng(seed)
    p = 1.0 / mean_block
    # Totalverlust-Tage (r <= -100%) koennen bei hohem Hebel auftreten; sie
    # werden auf -99.9999% gekappt, damit log endlich bleibt. Ein Pfad, der das
    # trifft, landet ohnehin klar unter -50%.
    logs = np.log1p(np.clip(r, -0.999999, None))
    idx = rng.integers(0, n, size=paths)
    acc = np.zeros(paths)
    for _ in range(horizon):
        acc += logs[idx]
        jump = rng.random(paths) < p
        idx = np.where(jump, rng.integers(0, n, size=paths), (idx + 1) % n)
    total = np.expm1(acc)
    return {
        "p_double": float((total >= 1.0).mean()),
        "p_half": float((total <= -0.5).mean()),
        "median": float(np.median(total)),
        "mean": float(total.mean()),
    }


# ---------------------------------------------------------------------------
# Kelly-artige Grenze: welcher Hebel maximiert das Endkapital?
# ---------------------------------------------------------------------------
def terminal_equity_curve(
    closes: pd.DataFrame,
    weights: pd.DataFrame,
    rebalance: pd.Series,
    short_rate: pd.Series,
    grid: Sequence[float],
    maint_margin: float = MAINT_MARGIN,
    borrow_spread: float = BORROW_SPREAD,
) -> pd.DataFrame:
    """Endkapital je Hebel - einmal mit und einmal ohne Liquidationsregel.

    Ohne Liquidation sieht man den reinen Vol-Drag (die Kelly-Kurve). Mit
    Liquidation sieht man, dass die Realitaet die Kurve noch weiter links
    abschneidet.
    """
    rows = []
    for L in grid:
        stopped = run_levered_portfolio(
            closes, weights, rebalance, short_rate, leverage=L,
            name=f"L{L}", maint_margin=maint_margin, borrow_spread=borrow_spread,
            stop_on_liquidation=True,
        )
        free = run_levered_portfolio(
            closes, weights, rebalance, short_rate, leverage=L,
            name=f"L{L}_free", maint_margin=maint_margin, borrow_spread=borrow_spread,
            stop_on_liquidation=False, max_gross=np.inf,
        )
        rows.append(
            {
                "leverage": L,
                "final_equity_with_margin": float(stopped.equity_curve.iloc[-1]),
                "final_equity_no_margin": float(free.equity_curve.iloc[-1]),
                "cagr_with_margin": stopped.calendar_cagr,
                "cagr_no_margin": free.calendar_cagr,
                "liquidated": stopped.liquidated,
                "arith_mean_pa": float(free.returns.mean() * TRADING_DAYS_PER_YEAR),
                "vol_pa": float(free.returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)),
            }
        )
    return pd.DataFrame(rows)


def analytic_kelly(returns: pd.Series, short_rate: pd.Series) -> dict:
    """f* = (mu - r) / sigma^2 auf Basis der ungehebelten Strategie.

    Grobe Orientierung, keine Zielgroesse: die Formel unterstellt normalverteilte
    iid-Renditen und kennt weder fette Raender noch Margin-Calls.
    """
    r = returns.dropna()
    rate = short_rate.reindex(r.index).ffill().bfill()
    excess = r - rate / TRADING_DAYS_PER_YEAR
    mu = float(excess.mean()) * TRADING_DAYS_PER_YEAR
    sigma = float(r.std()) * np.sqrt(TRADING_DAYS_PER_YEAR)
    return {
        "mu_excess_pa": mu,
        "sigma_pa": sigma,
        "kelly_f": mu / (sigma**2) if sigma > 0 else float("nan"),
        "half_kelly_f": 0.5 * mu / (sigma**2) if sigma > 0 else float("nan"),
    }


# ---------------------------------------------------------------------------
# Report-Helfer
# ---------------------------------------------------------------------------
PERIODS = {
    "Gesamt": (None, None),
    "Dev 2004-2018": (None, DEV_END),
    "Hold-out 2019-2026": (HOLDOUT_START, None),
    "2008": (pd.Timestamp("2008-01-01"), pd.Timestamp("2008-12-31")),
    "2020": (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-12-31")),
    "2022": (pd.Timestamp("2022-01-01"), pd.Timestamp("2022-12-31")),
}


def leverage_table(
    closes: pd.DataFrame,
    weights: pd.DataFrame,
    rebalance: pd.Series,
    short_rate: pd.Series,
    grid: Sequence[float] = LEVERAGE_GRID,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    maint_margin: float = MAINT_MARGIN,
    borrow_spread: float = BORROW_SPREAD,
    lows: pd.DataFrame | None = None,
    max_gross: float | None = None,
    stop_on_liquidation: bool = True,
) -> tuple[pd.DataFrame, dict]:
    sub = closes.loc[slice(start, end)]
    w = weights.loc[sub.index]
    reb = rebalance.loc[sub.index].copy()
    reb.iloc[0] = True
    sub_lows = lows.loc[sub.index] if lows is not None else None
    rows, results = {}, {}
    for L in grid:
        res = run_levered_portfolio(
            sub, w, reb, short_rate, leverage=L, name=f"{L:.1f}x",
            maint_margin=maint_margin, borrow_spread=borrow_spread, lows=sub_lows,
            max_gross=max_gross, stop_on_liquidation=stop_on_liquidation,
        )
        results[L] = res
        m = res.metrics
        rows[f"{L:.1f}x"] = {
            "CAGR%": 100 * m["cagr"],
            "Sharpe": m["sharpe"],
            "Sharpe_ex": m["sharpe_excess"],
            "Vol%": 100 * m["vol"],
            "MDD%": 100 * m["max_drawdown"],
            "Calmar": m["calmar"],
            "Verdopp.J": m["doubling_years"],
            "Endkapital": m["final_equity"],
            "AvgExp": m["avg_gross_exposure"],
            "CapBind%": 100 * m["cap_binding_share"],
            "MinMargin": m["min_margin_ratio"],
            "Fin%pa": 100 * m["financing_pa"],
            "Kosten%pa": 100 * m["cost_pa"],
            "TageGelebt": len(res.returns),
            "Liquidiert": (
                res.liquidation_date.date().isoformat() if res.liquidated else "-"
            ),
        }
    return pd.DataFrame(rows).T, results


def max_sustainable_leverage(
    closes: pd.DataFrame,
    weights: pd.DataFrame,
    rebalance: pd.Series,
    short_rate: pd.Series,
    maint_margin: float = MAINT_MARGIN,
    max_gross: float = PORTFOLIO_MARGIN_MAX_GROSS,
    grid: Sequence[float] | None = None,
) -> tuple[float, pd.DataFrame]:
    """Groesster Hebel, der im Fenster NIE einen Margin-Call ausgeloest haette.

    Das ist die praktisch relevante Obergrenze: nicht "was maximiert das
    Endkapital", sondern "was haette der Broker nicht abgeschaltet".
    """
    grid = grid if grid is not None else np.round(np.arange(1.0, 6.01, 0.05), 2)
    rows = []
    best = 0.0
    for L in grid:
        res = run_levered_portfolio(
            closes, weights, rebalance, short_rate, leverage=float(L),
            maint_margin=maint_margin, max_gross=max_gross, name=f"s{L}",
        )
        rows.append({
            "leverage": float(L),
            "liquidated": res.liquidated,
            "liq_date": res.liquidation_date,
            "min_margin": float(res.margin_ratio.min()),
            "cagr": res.calendar_cagr,
            "max_dd": performance_metrics(res.returns)["max_drawdown"],
        })
        if not res.liquidated:
            best = float(L)
    return best, pd.DataFrame(rows)


LEV_FLOAT_COLS = ("CAGR%", "Vol%", "MDD%", "Fin%pa", "Kosten%pa", "CapBind%")
LEV_RATIO_COLS = ("Sharpe", "Sharpe_ex", "Calmar", "AvgExp", "MinMargin")


def _show_lev(tbl: pd.DataFrame) -> pd.DataFrame:
    show = tbl.copy()
    for c in LEV_FLOAT_COLS:
        if c in show:
            show[c] = show[c].astype(float).round(2)
    for c in LEV_RATIO_COLS:
        if c in show:
            show[c] = show[c].astype(float).round(3)
    if "Verdopp.J" in show:
        show["Verdopp.J"] = show["Verdopp.J"].astype(float).round(1)
    if "Endkapital" in show:
        show["Endkapital"] = show["Endkapital"].astype(float).round(2)
    if "TageGelebt" in show:
        show["TageGelebt"] = show["TageGelebt"].astype(int)
    return show


def _p(x, nd=2):
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def main() -> None:
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)
    np.random.seed(RNG_SEED)

    print("=" * 100)
    print("GEHEBELTES VOL-TARGETING - HEBEL, FINANZIERUNG, LIQUIDATION")
    print("=" * 100)

    closes = load_closes(CLASSIC_UNIVERSE)
    lows = pd.DataFrame(
        {
            s: fetch_ohlcv(s, 12000, cache_dir="data_cache")["low"].reindex(closes.index).ffill()
            for s in closes.columns
        }
    )
    short_rate = load_short_rate(closes.index)

    base_w = apply_vol_target(closes, weights_equal(closes, CLASSIC_UNIVERSE), VT_LOOKBACK, TARGET_VOL)
    daily = rebalance_flags(closes.index, "daily")

    print(f"\nBasis: {BASE_NAME}, Universum {CLASSIC_UNIVERSE}")
    print(f"  Vol-Ziel {TARGET_VOL:.0%}, Lookback {VT_LOOKBACK}d, taegliche Vol-Skalierung")
    print(f"  Zeitraum {closes.index.min().date()} .. {closes.index.max().date()} ({len(closes)} Bars)")
    print(f"  Ungehebeltes Brutto-Exposure: Mittel {base_w.sum(axis=1).mean():.3f}, "
          f"Anteil Tage am Cap 1.0: {(base_w.sum(axis=1) > 0.999).mean():.1%}, "
          f"Cash im Mittel {1 - base_w.sum(axis=1).mean():.1%}")
    print(f"  Geldmarktsatz ^IRX: Mittel {short_rate.mean():.2%}, "
          f"2009-2021 {short_rate.loc['2009':'2021'].mean():.2%}, "
          f"2023-2026 {short_rate.loc['2023':].mean():.2%}")
    print(f"  Broker-Aufschlag {BORROW_SPREAD:.1%} -> Sollzins heute ca. "
          f"{short_rate.iloc[-1] + BORROW_SPREAD:.2%}, 2021 ca. "
          f"{short_rate.loc['2021'].mean() + BORROW_SPREAD:.2%}")
    print(f"  Wartungsmarge {MAINT_MARGIN:.0%} -> maximale Brutto-Exposure {1/MAINT_MARGIN:.1f}x")
    print(f"  Reg-T-Ersteinschuss erlaubt {REG_T_MAX_GROSS:.1f}x brutto, "
          f"Portfolio Margin ca. {PORTFOLIO_MARGIN_MAX_GROSS:.2f}x")

    # ------------------------------------------------------------------
    # TEIL 1+2: Hebeltabelle je Zeitraum
    # ------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("GROESSENLIMIT: WELCHER HEBEL IST UEBERHAUPT DARSTELLBAR?")
    print("=" * 100)
    exp = base_w.sum(axis=1)
    print("  Das Konto haelt L x die Strategiegewichte. Brutto-Exposure = L x Exposure.")
    print(f"  {'Hebel':>6} {'AvgExp':>8} {'MaxExp':>8} {'>2.0x (Reg-T)':>15} "
          f"{'>4.0x (Maint 25%)':>19} {'>6.67x (PM)':>13}")
    for L in LEVERAGE_GRID:
        g = L * exp
        print(f"  {L:5.1f}x {g.mean():8.3f} {g.max():8.3f} "
              f"{(g > REG_T_MAX_GROSS).mean():14.1%} "
              f"{(g > 1 / MAINT_MARGIN).mean():18.1%} "
              f"{(g > PORTFOLIO_MARGIN_MAX_GROSS).mean():12.1%}")
    print("\n  Lesart: bei 25% Wartungsmarge ist 4.0x Brutto-Exposure die harte")
    print("  Obergrenze. 4x und 5x auf die Strategie verlangen an vielen Tagen mehr,")
    print("  weil der Vol-Skalar in Ruhephasen bei 1.0 klebt (30% aller Tage).")
    print("  Zwei getrennte Grenzen:")
    print(f"    Ersteinschuss  -> maximale Positionsgroesse beim Eroeffnen")
    print(f"                      (Reg-T {REG_T_MAX_GROSS:.1f}x, "
          f"Portfolio Margin {PORTFOLIO_MARGIN_MAX_GROSS:.2f}x)")
    print(f"    Wartungsmarge  -> Zwangsliquidation, sobald Brutto-Exposure "
          f"{1/MAINT_MARGIN:.1f}x ueberschreitet")
    print("  Die Tabellen unten nutzen Portfolio Margin als Groessenlimit")
    print("  (CapBind% = Anteil Tage, an denen es greift) und 25% Wartungsmarge")
    print("  fuer die Liquidation. Unter normalem Reg-T waere bei dieser Strategie")
    print(f"  schon {REG_T_MAX_GROSS / exp.max():.2f}x der Schluss.")

    all_tables = {}
    all_results = {}
    for label, (s, e) in PERIODS.items():
        print("\n" + "=" * 100)
        print(f"HEBELSTUFEN - {label}  (Portfolio Margin 6.67x, Wartungsmarge 25%)")
        print("=" * 100)
        tbl, res = leverage_table(closes, base_w, daily, short_rate, start=s, end=e)
        all_tables[label] = tbl
        all_results[label] = res
        print(_show_lev(tbl).to_string())
        liq = {L: r for L, r in res.items() if r.liquidated}
        if liq:
            print("\n  Zwangsliquidationen:")
            for L, r in sorted(liq.items()):
                print(f"    {L:.1f}x  {r.liquidation_date.date()}  Restkapital "
                      f"{r.liquidation_equity:.3f}  Ursache: {r.liquidation_cause}")
        else:
            print("\n  Keine Zwangsliquidation in diesem Fenster.")

    # Hoechster Hebel, der nie einen Margin-Call ausgeloest haette.
    print("\n" + "=" * 100)
    print("MAXIMAL NACHHALTIGER HEBEL (nie ein Margin-Call im Fenster)")
    print("=" * 100)
    for mm in (MAINT_MARGIN, MAINT_MARGIN_STRICT):
        line = [f"  Wartungsmarge {mm:.0%} (max. Brutto {1/mm:.2f}x):"]
        for label in ("Gesamt", "Dev 2004-2018", "Hold-out 2019-2026", "2008", "2020", "2022"):
            s, e = PERIODS[label]
            sub = closes.loc[slice(s, e)]
            reb = daily.loc[sub.index].copy()
            reb.iloc[0] = True
            best_L, _ = max_sustainable_leverage(
                sub, base_w.loc[sub.index], reb, short_rate, maint_margin=mm
            )
            line.append(f"{label} = {best_L:.2f}x")
        print(line[0])
        for x in line[1:]:
            print(f"      {x}")
    print("\n  Die Grenze liegt dort, wo L x Exposure im Verlustfall ueber 1/m")
    print("  laeuft. Weil der Vol-Skalar 30% der Tage bei 1.0 klebt, ist das")
    print("  effektiv L x 1.0 - der Puffer kommt NICHT vom Vol-Targeting, sondern")
    print("  nur davon, dass 1/m ueber dem Hebel liegt.")

    # Ohne Broker-Restriktionen: die reine Oekonomie von 4x/5x.
    print("\n" + "=" * 100)
    print("OHNE BROKER-RESTRIKTIONEN - reine Oekonomie (Vol-Drag + Finanzierung)")
    print("=" * 100)
    print("  Kein Groessenlimit, keine Liquidation. Zeigt, was 4x/5x oekonomisch")
    print("  taeten, wenn ein Broker sie zuliesse (Futures/Portfolio Margin).")
    for label in ("Gesamt", "Dev 2004-2018", "Hold-out 2019-2026", "2022"):
        s, e = PERIODS[label]
        tbl, _ = leverage_table(
            closes, base_w, daily, short_rate, start=s, end=e,
            max_gross=np.inf, stop_on_liquidation=False,
        )
        print(f"\n  {label}:")
        print(_show_lev(tbl).drop(columns=["CapBind%", "Liquidiert"]).to_string())

    # Konservative Intraday-Schranke.
    print("\n" + "=" * 100)
    print("LIQUIDATION - KONSERVATIVE INTRADAY-SCHRANKE")
    print("=" * 100)
    print("  Zusatzpruefung mit den Tagestiefs ALLER Assets gleichzeitig. Das ist")
    print("  eine Worst-Case-Schranke (Assets erreichen ihr Tief nie synchron),")
    print("  gibt aber die Obergrenze des Liquidationsrisikos.")
    for label in ("Gesamt", "2008", "2020", "2022"):
        s, e = PERIODS[label]
        _, res_lo = leverage_table(
            closes, base_w, daily, short_rate, start=s, end=e, lows=lows
        )
        events = [
            f"{L:.1f}x -> {r.liquidation_date.date()} ({r.liquidation_cause}, "
            f"Rest {r.liquidation_equity:.2f})"
            for L, r in sorted(res_lo.items()) if r.liquidated
        ]
        print(f"\n  {label}: " + ("; ".join(events) if events else "keine"))

    # ------------------------------------------------------------------
    # Referenz: B&H QQQ, ungehebelt und gehebelt
    # ------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("REFERENZ: B&H QQQ mit Hebel (gleiche Kosten-/Zins-/Margin-Regeln)")
    print("=" * 100)
    qqq_w = weights_buy_and_hold(closes, "QQQ")
    never = rebalance_flags(closes.index, "never")
    for label in ("Gesamt", "Hold-out 2019-2026", "2022"):
        s, e = PERIODS[label]
        tbl, _ = leverage_table(
            closes, qqq_w, never, short_rate, grid=(1.0, 1.5, 2.0, 3.0), start=s, end=e
        )
        cols = ["CAGR%", "Sharpe", "Sharpe_ex", "MDD%", "Calmar", "Verdopp.J",
                "Endkapital", "Liquidiert"]
        print(f"\n  {label}:")
        print(_show_lev(tbl)[cols].to_string())

    # ------------------------------------------------------------------
    # TEIL 2b: 60-Tage-Fenster
    # ------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("60-HANDELSTAGE-FENSTER (rollierend, gesamte Historie) - VERDOPPELN ODER HALBIEREN?")
    print("=" * 100)
    print("  Jedes Fenster = frisches Konto. Ohne globalen Abbruch, damit nach dem")
    print("  ersten Margin-Call noch Fenster existieren; die Liquidationsquote")
    print("  wird separat als Anteil der Fenster mit Margin-Bruch ausgewiesen.")
    rows = []
    for L in (1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 8.0, 12.0):
        free = run_levered_portfolio(
            closes, base_w, daily, short_rate, leverage=L, name=f"free{L}",
            stop_on_liquidation=False, max_gross=np.inf,
        )
        d = rolling_window_distribution(free.returns, free.gross_exposure, window=60)
        d["p_margin_break"] = window_liquidation_rate(free, window=60)
        d["leverage"] = L
        rows.append(d)
    win = pd.DataFrame(rows).set_index("leverage")
    show = win[["windows", "median", "mean", "p95", "best", "worst", "p_positive",
                "p_double", "p_half", "p_margin_break"]].copy()
    for c in ("median", "mean", "p95", "best", "worst"):
        show[c] = (100 * show[c]).round(1)
    for c in ("p_positive", "p_double", "p_half", "p_margin_break"):
        show[c] = (100 * show[c]).round(2)
    show = show.rename(columns={
        "median": "Median%", "mean": "Mittel%", "p95": "P95%", "best": "Best%",
        "worst": "Worst%", "p_positive": "P(>0)%", "p_double": "P(>=+100%)%",
        "p_half": "P(<=-50%)%", "p_margin_break": "P(MarginBruch)%",
    })
    print("\n" + show.to_string())

    print("\n  Block-Bootstrap (stationaer, mean_block=10, 20k Pfade, seed "
          f"{RNG_SEED}) fuer 60 Tage:")
    boot_rows = []
    for L in (1.0, 2.0, 3.0, 5.0, 8.0, 12.0):
        free = run_levered_portfolio(
            closes, base_w, daily, short_rate, leverage=L, name=f"b{L}",
            stop_on_liquidation=False, max_gross=np.inf,
        )
        b = stationary_bootstrap_double(free.returns, horizon=60, paths=20_000)
        b["leverage"] = L
        boot_rows.append(b)
    bt = pd.DataFrame(boot_rows).set_index("leverage")
    bt["median"] = (100 * bt["median"]).round(1)
    bt["mean"] = (100 * bt["mean"]).round(1)
    bt["p_double"] = (100 * bt["p_double"]).round(2)
    bt["p_half"] = (100 * bt["p_half"]).round(2)
    print(bt.rename(columns={"median": "Median%", "mean": "Mittel%",
                             "p_double": "P(>=+100%)%", "p_half": "P(<=-50%)%"}).to_string())

    # ------------------------------------------------------------------
    # TEIL 3: endkapital-maximierender Hebel
    # ------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("OPTIMALE GROESSE: WELCHER HEBEL MAXIMIERT DAS ENDKAPITAL? (Gesamtzeitraum)")
    print("=" * 100)
    grid = np.round(np.arange(0.5, 12.01, 0.25), 3)
    curve = terminal_equity_curve(closes, base_w, daily, short_rate, grid)
    best_m = curve.loc[curve.final_equity_with_margin.idxmax()]
    best_f = curve.loc[curve.final_equity_no_margin.idxmax()]
    show = curve[curve.leverage.isin(
        [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0]
    )].copy()
    show["arith_mean_pa"] = (100 * show["arith_mean_pa"]).round(2)
    show["vol_pa"] = (100 * show["vol_pa"]).round(2)
    show["cagr_with_margin"] = (100 * show["cagr_with_margin"]).round(2)
    show["cagr_no_margin"] = (100 * show["cagr_no_margin"]).round(2)
    show["final_equity_with_margin"] = show["final_equity_with_margin"].round(3)
    show["final_equity_no_margin"] = show["final_equity_no_margin"].round(3)
    print("\n" + show.set_index("leverage").rename(columns={
        "final_equity_with_margin": "EndK(Margin)",
        "final_equity_no_margin": "EndK(ohne Margin)",
        "cagr_with_margin": "CAGR%(Margin)",
        "cagr_no_margin": "CAGR%(ohne)",
        "arith_mean_pa": "ArithMittel%pa",
        "vol_pa": "Vol%pa",
    }).to_string())
    print(f"\n  Endkapital-Maximum MIT Margin-Regel:  L = {best_m.leverage:.2f}  "
          f"(Endkapital {best_m.final_equity_with_margin:.2f}, CAGR {best_m.cagr_with_margin:.2%})")
    print(f"  Endkapital-Maximum OHNE Margin-Regel: L = {best_f.leverage:.2f}  "
          f"(Endkapital {best_f.final_equity_no_margin:.2f}, CAGR {best_f.cagr_no_margin:.2%})")
    print("  Das arithmetische Mittel pro Tag steigt linear mit L weiter - das")
    print("  Endkapital faellt jenseits des Maximums. Genau das ist der Vol-Drag.")

    k = analytic_kelly(
        run_levered_portfolio(closes, base_w, daily, short_rate, leverage=1.0,
                              stop_on_liquidation=False, max_gross=np.inf).returns,
        short_rate,
    )
    print(f"\n  Analytisches Kelly (Orientierung, iid-Normal unterstellt): "
          f"mu_excess {k['mu_excess_pa']:.2%}, sigma {k['sigma_pa']:.2%}, "
          f"f* = {k['kelly_f']:.2f}x, halbes Kelly {k['half_kelly_f']:.2f}x")

    # ------------------------------------------------------------------
    # Zinsregime-Abhaengigkeit
    # ------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("WIE STARK HAENGT DAS AN DER NULLZINSPHASE?")
    print("=" * 100)
    regimes = {
        "Echte Zinsen (^IRX)": short_rate,
        "Konstant 0.1% (ZIRP)": pd.Series(0.001, index=closes.index),
        "Konstant 4.3% (heute)": pd.Series(0.043, index=closes.index),
    }
    rows = []
    for rlabel, rs in regimes.items():
        for L in (1.0, 2.0, 3.0):
            res = run_levered_portfolio(
                closes, base_w, daily, rs, leverage=L, name=f"{rlabel}-{L}",
                stop_on_liquidation=True,
            )
            m = res.metrics
            rows.append({
                "Zinsregime": rlabel, "Hebel": f"{L:.1f}x",
                "CAGR%": round(100 * m["cagr"], 2),
                "Sharpe": round(m["sharpe"], 3),
                "Fin%pa": round(100 * m["financing_pa"], 2),
                "Cash%pa": round(100 * m["cash_income_pa"], 2),
                "Endkapital": round(m["final_equity"], 2),
            })
    print("\n" + pd.DataFrame(rows).to_string(index=False))

    print("\n  Teilperioden-CAGR bei 2x, getrennt nach Zinsumfeld (echte Zinsen):")
    sub_windows = {
        "2004-2008 (Zinsen 1-5%)": ("2004", "2008"),
        "2009-2015 (ZIRP)": ("2009", "2015"),
        "2016-2021 (ZIRP-Ausklang)": ("2016", "2021"),
        "2022-2026 (Zinswende)": ("2022", None),
    }
    rows = []
    for wl, (a, b) in sub_windows.items():
        sub = closes.loc[a:b] if b else closes.loc[a:]
        reb = daily.loc[sub.index].copy()
        reb.iloc[0] = True
        for L in (1.0, 2.0, 3.0):
            res = run_levered_portfolio(
                sub, base_w.loc[sub.index], reb, short_rate, leverage=L, name=wl,
            )
            m = res.metrics
            rows.append({
                "Fenster": wl, "Hebel": f"{L:.1f}x",
                "CAGR%": round(100 * m["cagr"], 2), "Sharpe": round(m["sharpe"], 3),
                "MDD%": round(100 * m["max_drawdown"], 2),
                "Fin%pa": round(100 * m["financing_pa"], 2),
                "Liq": res.liquidation_date.date().isoformat() if res.liquidated else "-",
            })
    print("\n" + pd.DataFrame(rows).to_string(index=False))

    # ------------------------------------------------------------------
    # Sensitivitaeten
    # ------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("SENSITIVITAET: BROKER-AUFSCHLAG UND WARTUNGSMARGE")
    print("=" * 100)
    rows = []
    for spread in (BORROW_SPREAD_LOW, BORROW_SPREAD, BORROW_SPREAD_HIGH):
        for mm in (MAINT_MARGIN, MAINT_MARGIN_STRICT):
            for L in (2.0, 3.0):
                res = run_levered_portfolio(
                    closes, base_w, daily, short_rate, leverage=L,
                    borrow_spread=spread, maint_margin=mm, name="sens",
                )
                m = res.metrics
                rows.append({
                    "Spread": f"{spread:.1%}", "MaintMargin": f"{mm:.0%}",
                    "Hebel": f"{L:.1f}x", "CAGR%": round(100 * m["cagr"], 2),
                    "Sharpe": round(m["sharpe"], 3),
                    "MDD%": round(100 * m["max_drawdown"], 2),
                    "Liq": res.liquidation_date.date().isoformat() if res.liquidated else "-",
                })
    print("\n" + pd.DataFrame(rows).to_string(index=False))

    print("\n" + "=" * 100)
    print("ALTERNATIVE LESART: VOL-TARGET DARF SELBST HEBELN (konstantes 6%-Ziel)")
    print("=" * 100)
    print("  Oben wird die fertige Strategie L-fach gehalten (Vol-Ziel wird L*6%).")
    print("  Hier darf der Vol-Skalar stattdessen bis max_exposure laufen, das")
    print("  Vol-Ziel bleibt 6% - risikokontrolliert, aber mit Hebel in Ruhephasen.")
    rows = []
    for cap in (1.0, 1.5, 2.0, 3.0, 4.0):
        w = apply_vol_target(
            closes, weights_equal(closes, CLASSIC_UNIVERSE), VT_LOOKBACK, TARGET_VOL,
            max_exposure=cap,
        )
        res = run_levered_portfolio(closes, w, daily, short_rate, leverage=1.0,
                                    name=f"cap{cap}", lows=lows)
        m = res.metrics
        rows.append({
            "max_exposure": f"{cap:.1f}x", "CAGR%": round(100 * m["cagr"], 2),
            "Sharpe": round(m["sharpe"], 3), "Vol%": round(100 * m["vol"], 2),
            "MDD%": round(100 * m["max_drawdown"], 2), "Calmar": round(m["calmar"], 3),
            "AvgExp": round(m["avg_gross_exposure"], 3),
            "Verdopp.J": round(m["doubling_years"], 1),
            "Liq": res.liquidation_date.date().isoformat() if res.liquidated else "-",
        })
    print("\n" + pd.DataFrame(rows).to_string(index=False))

    print("\n" + "=" * 100)
    print("FAZIT (Zahlen oben, Interpretation im Bericht)")
    print("=" * 100)
    print(f"  Endkapital-maximierender Hebel (Gesamtzeitraum, mit Margin): "
          f"{best_m.leverage:.2f}x")
    print(f"  Hoechster Hebel ohne Liquidation im Gesamtzeitraum: siehe Tabelle 'Gesamt'.")
    print("  Der Hold-out wurde NICHT zur Hebelwahl benutzt - nur berichtet.")


if __name__ == "__main__":
    main()
