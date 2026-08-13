"""Tests for the performance statistics."""

import numpy as np
import pandas as pd
import pytest

from regime_allocation.metrics import (
    comparison_table,
    drawdown_curve,
    longest_drawdown_days,
    max_drawdown,
    summarize,
)


def series(values, start="2020-01-01"):
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq="B"))


def test_drawdown_is_zero_on_a_monotonic_curve():
    assert (drawdown_curve(series([1.0, 1.1, 1.2, 1.3])) == 0).all()


def test_max_drawdown_measures_peak_to_trough():
    assert max_drawdown(series([1.0, 2.0, 1.0, 1.5])) == pytest.approx(-0.5)


def test_drawdown_is_measured_from_the_peak_not_the_start():
    """A curve that ends above its start can still have a real drawdown."""
    assert max_drawdown(series([1.0, 4.0, 2.0, 5.0])) == pytest.approx(-0.5)


def test_longest_underwater_stretch():
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    equity = pd.Series([1.0, 0.9, 0.8, 0.95, 1.5], index=idx)
    assert longest_drawdown_days(equity) == 3
    assert longest_drawdown_days(pd.Series([1.0, 1.1, 1.2], index=idx[:3])) == 0


def test_total_return_compounds_daily_returns():
    daily = 1.10 ** (1 / 365) - 1
    idx = pd.date_range("2021-01-01", periods=366, freq="D")
    stats = summarize(pd.Series([0.0] + [daily] * 365, index=idx), label="test")
    assert stats["total_return"] == pytest.approx(0.10, rel=1e-3)


def test_cagr_equals_total_return_over_exactly_one_year():
    daily = 1.10 ** (1 / 365) - 1
    idx = pd.date_range("2021-01-01", periods=366, freq="D")
    stats = summarize(pd.Series([0.0] + [daily] * 365, index=idx), label="test")
    assert stats["years"] == pytest.approx(1.0, rel=1e-2)
    assert stats["cagr"] == pytest.approx(stats["total_return"], rel=1e-2)


def test_cagr_annualises_a_multi_year_run_downward():
    """21% over two years is ~10% a year, not 21%."""
    idx = pd.date_range("2020-01-01", periods=2, freq="730D")
    stats = summarize(pd.Series([0.0, 0.21], index=idx), label="test")
    assert stats["cagr"] == pytest.approx(0.10, abs=0.005)


def test_sharpe_is_net_of_the_risk_free_rate():
    rets = series(list(np.linspace(0.001, 0.003, 60)))
    gross = summarize(rets, risk_free_rate=0.0, label="s")["sharpe"]
    net = summarize(rets, risk_free_rate=0.06, label="s")["sharpe"]
    assert net < gross


def test_zero_volatility_does_not_raise():
    stats = summarize(series([0.0] * 30), label="flat")
    assert stats["volatility"] == pytest.approx(0.0)
    assert not np.isfinite(stats["sharpe"]) or stats["sharpe"] == 0


def test_position_derived_statistics():
    rets = series([0.01, -0.01, 0.02, 0.0])
    pos = series([0.0, 1.0, 1.0, 0.0])
    stats = summarize(rets, positions=pos, costs=series([0.0, 0.001, 0.0, 0.001]), label="s")
    assert stats["time_in_market"] == pytest.approx(0.5)
    assert stats["avg_exposure"] == pytest.approx(0.5)
    assert stats["trades"] == 2  # flat -> 1x -> flat
    assert stats["total_costs"] == pytest.approx(0.002)


def test_trade_count_ignores_the_opening_nan():
    """Regression: ``diff() != 0`` counted the leading NaN as a trade."""
    rets = series([0.0] * 4)
    pos = series([0.0, 0.0, 0.0, 0.0])
    assert summarize(rets, positions=pos, label="s")["trades"] == 0


def test_hit_rate_counts_strictly_positive_days():
    stats = summarize(series([0.01, -0.01, 0.0, 0.02]), label="s")
    assert stats["hit_rate"] == pytest.approx(0.5)


def test_empty_input_is_rejected():
    with pytest.raises(ValueError):
        summarize(pd.Series(dtype=float), label="empty")


def test_comparison_table_renders_both_columns():
    a = summarize(series([0.01, 0.02, -0.01]), label="Strategy")
    b = summarize(series([0.005, 0.01, 0.0]), label="Buy & hold")
    table = comparison_table(a, b)
    assert "Strategy" in table and "Buy & hold" in table
    assert "Max drawdown" in table and "Sharpe" in table
