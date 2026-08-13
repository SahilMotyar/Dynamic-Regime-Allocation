"""Tests for the allocation state machine."""

import numpy as np
import pandas as pd
import pytest

from regime_allocation.config import StrategyConfig
from regime_allocation.strategy import next_position, signal_label, target_positions

CFG = StrategyConfig()


def test_aggressive_tier_is_reachable_from_cash():
    """Regression: the 1.5x tier used to be dead code.

    The original rule tested ``p_bull > 0.60`` before ``p_bull > 0.80``, so the
    second branch could never fire and the documented "aggressive buy" never
    happened in the backtest.
    """
    assert next_position(0.0, 0.05, 0.05, 0.90, 1.0, CFG) == CFG.leverage_max


def test_aggressive_tier_is_reachable_as_an_upgrade():
    assert next_position(CFG.leverage_base, 0.05, 0.05, 0.90, 1.0, CFG) == CFG.leverage_max


def test_standard_entry_between_thresholds():
    assert next_position(0.0, 0.1, 0.2, 0.70, 1.0, CFG) == CFG.leverage_base


def test_sideways_entry_requires_price_above_the_z_floor():
    assert next_position(0.0, 0.2, 0.70, 0.1, 0.0, CFG) == CFG.leverage_base
    assert next_position(0.0, 0.2, 0.70, 0.1, -2.0, CFG) == 0.0


def test_bear_reading_forces_cash_from_any_exposure():
    for held in (CFG.leverage_base, CFG.leverage_max):
        assert next_position(held, 0.75, 0.15, 0.10, 0.0, CFG) == 0.0


def test_hysteresis_holds_the_top_tier_inside_the_band():
    """Between de-escalate and aggressive the top tier is kept, not toggled."""
    held = next_position(CFG.leverage_max, 0.1, 0.12, 0.78, 1.0, CFG)
    assert held == CFG.leverage_max
    assert next_position(CFG.leverage_max, 0.1, 0.2, 0.70, 1.0, CFG) == CFG.leverage_base


def test_position_is_sticky_while_merely_neutral():
    """Once invested, a lukewarm reading does not trigger an exit."""
    assert next_position(CFG.leverage_base, 0.4, 0.4, 0.2, 0.0, CFG) == CFG.leverage_base


def test_nan_beliefs_leave_the_position_untouched():
    assert next_position(1.0, np.nan, np.nan, np.nan, 0.0, CFG) == 1.0
    assert next_position(0.0, 0.1, 0.1, 0.8, np.nan, CFG) == 0.0


def test_target_positions_runs_the_machine_in_order():
    idx = pd.date_range("2020-01-01", periods=4, freq="D")
    df = pd.DataFrame(
        {
            "P_Bear": [0.1, 0.1, 0.9, 0.1],
            "P_Sideways": [0.1, 0.1, 0.05, 0.6],
            "P_Bull": [0.9, 0.7, 0.05, 0.3],
            "Z_Score": [1.0, 1.0, -2.0, 0.5],
        },
        index=idx,
    )
    got = target_positions(df, CFG).tolist()
    assert got == [CFG.leverage_max, CFG.leverage_base, 0.0, CFG.leverage_base]


def test_target_positions_rejects_missing_columns():
    df = pd.DataFrame({"P_Bull": [0.9]}, index=pd.date_range("2020-01-01", periods=1))
    with pytest.raises(KeyError):
        target_positions(df, CFG)


def test_signal_label_distinguishes_the_tiers():
    assert "CASH" in signal_label(0.0, CFG)
    assert "STRONG" in signal_label(CFG.leverage_max, CFG)
    assert "INVESTED" in signal_label(CFG.leverage_base, CFG)


def test_config_rejects_inconsistent_thresholds():
    with pytest.raises(ValueError):
        StrategyConfig(bull_entry=0.9, bull_aggressive=0.8, bull_deescalate=0.85)
    with pytest.raises(ValueError):
        StrategyConfig(leverage_base=1.0, leverage_max=0.5)
