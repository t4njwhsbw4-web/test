"""Periodische Auswertung abgeschlossener Trades - der "immer lernen"-Teil.

WICHTIGE DISZIPLIN, die dieses Modul bewusst NICHT verletzt: Es passt
KEINE Parameter automatisch an. Genau das war die Falle bei der Krypto-
Strategie-Suche in diesem Projekt (Cross-Sectional-Momentum: Dev-Sharpe
1.10, Hold-out -1.10) - bei kleiner Stichprobe "lernt" man Rauschen, nicht
Signal. Dieses Modul liefert nur die Datengrundlage (Kennzahlen je
Strategie, Stichprobengröße, Exit-Gründe) für eine informierte, manuelle
Entscheidung - keine automatische Selbstmodifikation der Filter-Regeln.

MIN_TRADES_FOR_ANY_CONCLUSION: unterhalb dieser Schwelle wird explizit
"zu wenig Daten" ausgegeben statt einer (bedeutungslosen) Prozentzahl.
"""
from __future__ import annotations

import csv
import dataclasses
import pathlib
from collections import defaultdict

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
TRADES_CSV_PATH = REPO_ROOT / "artifacts" / "memecoin_paper" / "trades.csv"

MIN_TRADES_FOR_ANY_CONCLUSION = 20  # unter dieser Zahl: "zu wenig Daten", keine Prozentangabe


@dataclasses.dataclass
class ClosedTrade:
    strategy: str
    symbol: str
    token_address: str
    entry_price: float
    exit_price: float
    return_pct: float
    exit_reason: str


@dataclasses.dataclass
class StrategyReview:
    strategy: str
    n_closed: int
    n_open_still: int
    enough_data: bool
    win_rate: float | None
    median_return_pct: float | None
    best_return_pct: float | None
    worst_return_pct: float | None
    exit_reason_counts: dict[str, int]


def _read_rows() -> list[dict]:
    if not TRADES_CSV_PATH.exists():
        return []
    with open(TRADES_CSV_PATH, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _pair_trades(rows: list[dict]) -> tuple[list[ClosedTrade], dict[str, int]]:
    """Paart BUY->SELL pro (token_address, strategy) in chronologischer
    Reihenfolge (FIFO - für diesen Prototyp ausreichend, da MAX_OPEN_POSITIONS
    pro Strategie je Token ohnehin nur eine offene Position gleichzeitig
    erlaubt, siehe run_loop.py). Liefert geschlossene Trades + Anzahl noch
    offener (BUY ohne zugehöriges SELL) je Strategie."""
    open_buys: dict[tuple[str, str], dict] = {}
    closed: list[ClosedTrade] = []
    still_open: dict[str, int] = defaultdict(int)

    for row in rows:
        strategy = row.get("strategy", "-")
        if strategy in ("-", "", None):
            continue  # SKIP-Zeilen ohne Strategie-Zuordnung
        key = (row.get("token_address", ""), strategy)
        action = row.get("action")

        if action == "BUY":
            try:
                price = float(row["price"])
            except (TypeError, ValueError):
                continue
            open_buys[key] = {"entry_price": price, "symbol": row.get("symbol", "?")}
        elif action == "SELL" and key in open_buys:
            entry = open_buys.pop(key)
            try:
                exit_price = float(row["price"])
            except (TypeError, ValueError):
                continue
            if entry["entry_price"] <= 0:
                continue
            return_pct = (exit_price - entry["entry_price"]) / entry["entry_price"]
            reason_raw = row.get("reason", "") or ""
            exit_reason = reason_raw.split("(")[0].strip() or "unbekannt"
            closed.append(ClosedTrade(
                strategy=strategy, symbol=entry["symbol"], token_address=key[0],
                entry_price=entry["entry_price"], exit_price=exit_price,
                return_pct=return_pct, exit_reason=exit_reason,
            ))

    for (_token, strategy) in open_buys:
        still_open[strategy] += 1

    return closed, dict(still_open)


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    s = sorted(values)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0


def review_all_strategies() -> dict[str, StrategyReview]:
    rows = _read_rows()
    closed, still_open = _pair_trades(rows)

    by_strategy: dict[str, list[ClosedTrade]] = defaultdict(list)
    for t in closed:
        by_strategy[t.strategy].append(t)

    all_strategies = set(by_strategy) | set(still_open)
    result: dict[str, StrategyReview] = {}

    for strategy in sorted(all_strategies):
        trades = by_strategy.get(strategy, [])
        n = len(trades)
        enough = n >= MIN_TRADES_FOR_ANY_CONCLUSION

        returns = [t.return_pct for t in trades]
        wins = [r for r in returns if r > 0]

        exit_counts: dict[str, int] = defaultdict(int)
        for t in trades:
            exit_counts[t.exit_reason] += 1

        result[strategy] = StrategyReview(
            strategy=strategy,
            n_closed=n,
            n_open_still=still_open.get(strategy, 0),
            enough_data=enough,
            win_rate=(len(wins) / n) if n > 0 else None,
            median_return_pct=_median(returns),
            best_return_pct=max(returns) if returns else None,
            worst_return_pct=min(returns) if returns else None,
            exit_reason_counts=dict(exit_counts),
        )

    return result


def format_report(reviews: dict[str, StrategyReview]) -> str:
    lines = [
        "=" * 90,
        "TRADE-REVIEW - abgeschlossene Paper-Trades je Strategie",
        "=" * 90,
        "",
        f"Schwelle fuer eine belastbare Aussage: >= {MIN_TRADES_FOR_ANY_CONCLUSION} geschlossene Trades.",
        "Unterhalb dieser Schwelle wird bewusst KEINE Prozentzahl interpretiert -",
        "zu wenig Daten ist ein gueltiges Ergebnis, keine Luecke im Bericht.",
        "",
    ]
    for strategy, r in reviews.items():
        lines.append(f"--- {strategy} " + "-" * (60 - len(strategy)))
        lines.append(f"  Geschlossene Trades: {r.n_closed}  |  noch offen: {r.n_open_still}")
        if not r.enough_data:
            lines.append(
                f"  ZU WENIG DATEN (< {MIN_TRADES_FOR_ANY_CONCLUSION}) - keine Interpretation, "
                "nur Rohzahlen zur Beobachtung:"
            )
        if r.n_closed > 0:
            lines.append(f"  Win-Rate: {r.win_rate:.1%}  |  Median-Rendite: {r.median_return_pct:+.1%}")
            lines.append(f"  Beste: {r.best_return_pct:+.1%}  |  Schlechteste: {r.worst_return_pct:+.1%}")
            lines.append(f"  Exit-Gruende: {r.exit_reason_counts}")
        else:
            lines.append("  Noch keine geschlossenen Trades.")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    reviews = review_all_strategies()
    print(format_report(reviews))
