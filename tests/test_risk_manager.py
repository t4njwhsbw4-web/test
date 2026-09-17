import pytest

from src.risk.manager import AccountState, KillSwitchTriggered, RiskLimits, RiskManager


def _limits(**overrides):
    defaults = dict(
        max_position_pct=0.10,
        max_daily_loss_pct=0.03,
        max_drawdown_pct=0.15,
        stop_loss_pct=0.05,
        take_profit_pct=0.10,
        max_open_positions=5,
    )
    defaults.update(overrides)
    return RiskLimits(**defaults)


def test_kill_switch_triggers_on_daily_loss():
    rm = RiskManager(_limits())
    account = AccountState(equity=9600, starting_equity_today=10000, peak_equity=10000, open_positions=0)
    with pytest.raises(KillSwitchTriggered):
        rm.check_kill_switch(account)


def test_kill_switch_triggers_on_drawdown():
    rm = RiskManager(_limits())
    account = AccountState(equity=8400, starting_equity_today=9900, peak_equity=10000, open_positions=0)
    with pytest.raises(KillSwitchTriggered):
        rm.check_kill_switch(account)


def test_no_kill_switch_within_limits():
    rm = RiskManager(_limits())
    account = AccountState(equity=9900, starting_equity_today=10000, peak_equity=10000, open_positions=0)
    rm.check_kill_switch(account)  # darf nicht werfen
    assert not rm.halted


def test_kill_switch_stays_halted_until_manual_reset():
    rm = RiskManager(_limits())
    bad_account = AccountState(equity=9600, starting_equity_today=10000, peak_equity=10000, open_positions=0)
    with pytest.raises(KillSwitchTriggered):
        rm.check_kill_switch(bad_account)

    good_account = AccountState(equity=10000, starting_equity_today=10000, peak_equity=10000, open_positions=0)
    with pytest.raises(KillSwitchTriggered):
        rm.check_kill_switch(good_account)


def test_position_size_respects_max_position_pct():
    rm = RiskManager(_limits(max_position_pct=0.10))
    account = AccountState(equity=10000, starting_equity_today=10000, peak_equity=10000, open_positions=0)
    qty = rm.position_size(account, price=100.0)
    assert qty == pytest.approx(10.0)  # 10% von 10000 / 100


def test_position_size_zero_when_max_open_positions_reached():
    rm = RiskManager(_limits(max_open_positions=2))
    account = AccountState(equity=10000, starting_equity_today=10000, peak_equity=10000, open_positions=2)
    assert rm.position_size(account, price=100.0) == 0.0


def test_should_exit_stop_loss():
    rm = RiskManager(_limits(stop_loss_pct=0.05, take_profit_pct=0.10))
    assert rm.should_exit(entry_price=100.0, current_price=94.0) == "stop_loss"


def test_should_exit_take_profit():
    rm = RiskManager(_limits(stop_loss_pct=0.05, take_profit_pct=0.10))
    assert rm.should_exit(entry_price=100.0, current_price=111.0) == "take_profit"


def test_should_exit_none_within_band():
    rm = RiskManager(_limits(stop_loss_pct=0.05, take_profit_pct=0.10))
    assert rm.should_exit(entry_price=100.0, current_price=102.0) is None
