"""Tests for the PnL engine's accounting."""

import numpy as np
import pandas as pd
import pytest

from regime_allocation.backtest import run_backtest
from regime_allocation.config import TRADING_DAYS, BacktestConfig, StrategyConfig

STRAT = StrategyConfig(leverage_max=1.5)  # levered, so financing is exercised
FREE = BacktestConfig(tx_cost=0.0, cash_rate=0.0, borrow_spread=0.0)


def make_frame(n=12, bull=0.9, bear=0.02, ret=0.01):
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    side = 1.0 - bull - bear
    return pd.DataFrame(
        {
            "Close": 100 * (1 + ret) ** np.arange(n),
            "Returns": [ret] * n,
            "Z_Score": [1.0] * n,
            "Volatility": [0.15] * n,
            "P_Bear": [bear] * n,
            "P_Sideways": [side] * n,
            "P_Bull": [bull] * n,
        },
        index=idx,
    )


def test_position_lags_the_target_by_one_bar():
    """No bar's own return may inform the exposure that captures it."""
    res = run_backtest(make_frame(), STRAT, FREE)
    assert res["Position"].iloc[0] == 0.0
    pd.testing.assert_series_equal(
        res["Position"].iloc[1:],
        res["Target"].shift(1).iloc[1:],
        check_names=False,
    )


def test_first_day_is_always_flat():
    """A strategy cannot be invested before it has seen its first signal."""
    res = run_backtest(make_frame(), STRAT, FREE)
    assert res["Gross_Ret"].iloc[0] == 0.0


def test_gross_return_is_position_times_market_return():
    res = run_backtest(make_frame(ret=0.02), STRAT, FREE)
    expected = res["Position"] * res["Returns"]
    pd.testing.assert_series_equal(res["Gross_Ret"], expected, check_names=False)


def test_cash_yield_accrues_only_on_uninvested_capital():
    cfg = BacktestConfig(tx_cost=0.0, cash_rate=0.06, borrow_spread=0.0)
    res = run_backtest(make_frame(bull=0.01, bear=0.9, ret=-0.01), STRAT, cfg)
    assert (res["Position"] == 0.0).all()
    assert res["Cash_Yield"].iloc[0] == pytest.approx(0.06 / TRADING_DAYS)
    assert (res["Borrow_Cost"] == 0.0).all()


def test_leverage_above_one_is_financed():
    """Regression: levered days used to be free money.

    The original engine credited a cash yield but never charged for the
    borrowed portion, so every 1.5x day was flattered by the missing carry.
    """
    cfg = BacktestConfig(tx_cost=0.0, cash_rate=0.06, borrow_spread=0.02)
    res = run_backtest(make_frame(bull=0.95), STRAT, cfg)
    levered = res[res["Position"] > 1.0]
    assert not levered.empty
    expected = 0.5 * (0.06 + 0.02) / TRADING_DAYS
    assert levered["Borrow_Cost"].iloc[0] == pytest.approx(expected)
    assert (levered["Cash_Yield"] == 0.0).all()


def test_transaction_cost_is_charged_on_exposure_turned_over():
    cfg = BacktestConfig(tx_cost=0.001, cash_rate=0.0, borrow_spread=0.0)
    res = run_backtest(make_frame(bull=0.95), STRAT, cfg)
    changes = res["Position"].diff().fillna(res["Position"].abs()).abs()
    pd.testing.assert_series_equal(res["Tx_Cost"], changes * 0.001, check_names=False)
    assert res["Tx_Cost"].sum() > 0


def test_net_return_is_the_sum_of_its_parts():
    cfg = BacktestConfig(tx_cost=0.001, cash_rate=0.06, borrow_spread=0.02)
    res = run_backtest(make_frame(), STRAT, cfg)
    expected = res["Gross_Ret"] + res["Cash_Yield"] - res["Borrow_Cost"] - res["Tx_Cost"]
    pd.testing.assert_series_equal(res["Net_Ret"], expected, check_names=False)


def test_equity_curves_compound_their_return_series():
    res = run_backtest(make_frame(), STRAT, FREE)
    pd.testing.assert_series_equal(
        res["Equity_Market"], (1 + res["Returns"]).cumprod(), check_names=False
    )
    pd.testing.assert_series_equal(
        res["Equity_Strategy"], (1 + res["Net_Ret"]).cumprod(), check_names=False
    )


def test_costs_can_only_reduce_performance():
    frame = make_frame(n=40)
    free = run_backtest(frame, STRAT, BacktestConfig(0.0, 0.0, 0.0))
    charged = run_backtest(frame, STRAT, BacktestConfig(0.005, 0.0, 0.05))
    assert charged["Equity_Strategy"].iloc[-1] < free["Equity_Strategy"].iloc[-1]


def test_missing_returns_column_is_rejected():
    with pytest.raises(KeyError):
        run_backtest(make_frame().drop(columns=["Returns"]), STRAT, FREE)
