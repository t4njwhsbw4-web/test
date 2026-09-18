import csv

from src.paper_memecoin.trade_review import MIN_TRADES_FOR_ANY_CONCLUSION, _pair_trades, review_all_strategies


def _rows(*tuples):
    """tuples: (token_address, strategy, action, price, reason)"""
    return [
        {"token_address": t, "strategy": s, "action": a, "price": str(p), "symbol": "X", "reason": r}
        for t, s, a, p, r in tuples
    ]


def test_pairs_buy_and_sell_fifo():
    rows = _rows(
        ("mintA", "baseline", "BUY", 1.0, ""),
        ("mintA", "baseline", "SELL", 1.5, "trailing_stop (+50%)"),
    )
    closed, still_open = _pair_trades(rows)
    assert len(closed) == 1
    assert closed[0].return_pct == 0.5
    assert closed[0].exit_reason == "trailing_stop"
    assert still_open == {}


def test_unmatched_buy_counts_as_still_open():
    rows = _rows(("mintA", "baseline", "BUY", 1.0, ""))
    closed, still_open = _pair_trades(rows)
    assert closed == []
    assert still_open == {"baseline": 1}


def test_skip_rows_without_strategy_are_ignored():
    rows = _rows(("mintA", "-", "SKIP", 1.0, "irrelevant"))
    closed, still_open = _pair_trades(rows)
    assert closed == []
    assert still_open == {}


def test_below_threshold_flags_not_enough_data(tmp_path, monkeypatch):
    csv_path = tmp_path / "trades.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["timestamp", "token_address", "symbol", "strategy", "action", "price", "qty", "reason", "equity_after"])
        w.writeheader()
        w.writerow({"timestamp": "t", "token_address": "m1", "symbol": "X", "strategy": "baseline", "action": "BUY", "price": "1.0", "qty": "1", "reason": "", "equity_after": ""})
        w.writerow({"timestamp": "t", "token_address": "m1", "symbol": "X", "strategy": "baseline", "action": "SELL", "price": "0.5", "qty": "1", "reason": "stop_loss (-50%)", "equity_after": ""})

    import src.paper_memecoin.trade_review as tr
    monkeypatch.setattr(tr, "TRADES_CSV_PATH", csv_path)

    reviews = review_all_strategies()
    assert reviews["baseline"].n_closed == 1
    assert reviews["baseline"].n_closed < MIN_TRADES_FOR_ANY_CONCLUSION
    assert reviews["baseline"].enough_data is False
    assert reviews["baseline"].win_rate == 0.0
    assert reviews["baseline"].median_return_pct == -0.5
