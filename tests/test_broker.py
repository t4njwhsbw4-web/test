import pytest

from src.execution.broker import PaperBroker


def test_buy_reduces_cash_and_creates_position():
    broker = PaperBroker(starting_cash=10000)
    broker.submit_order("AAPL", qty=10, side="buy", price=100.0)
    pos = broker.get_position("AAPL")
    assert pos is not None
    assert pos.qty == 10
    assert pos.entry_price == 100.0
    assert broker.get_equity() == pytest.approx(10000.0)  # Cash -> Position, Equity unverändert


def test_sell_without_position_raises():
    broker = PaperBroker(starting_cash=10000)
    with pytest.raises(ValueError):
        broker.submit_order("AAPL", qty=1, side="sell", price=100.0)


def test_buy_without_enough_cash_raises():
    broker = PaperBroker(starting_cash=100)
    with pytest.raises(ValueError):
        broker.submit_order("AAPL", qty=10, side="buy", price=100.0)


def test_full_round_trip_closes_position():
    broker = PaperBroker(starting_cash=10000)
    broker.submit_order("AAPL", qty=10, side="buy", price=100.0)
    broker.submit_order("AAPL", qty=10, side="sell", price=110.0)
    assert broker.get_position("AAPL") is None
    assert broker.get_equity() == pytest.approx(10100.0)  # 100 Profit realisiert
