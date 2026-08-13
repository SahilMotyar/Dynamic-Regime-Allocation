"""The allocation rule.

This is the single source of truth for "given today's beliefs, how much exposure
should I carry tomorrow". Both the backtest and the live trade card call
``next_position``, so the printed recommendation can never drift away from the
behaviour that was actually measured.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import StrategyConfig


def next_position(
    current_pos: float,
    p_bear: float,
    p_sideways: float,
    p_bull: float,
    z_score: float,
    cfg: StrategyConfig,
) -> float:
    """Target exposure for the next bar, given the position held into this one.

    The rule is deliberately sticky: once invested, only a bear reading takes
    the position back to cash. Trend-following exits on "not bullish enough"
    were what produced most of the churn in early versions.
    """
    if any(not np.isfinite(v) for v in (p_bear, p_sideways, p_bull, z_score)):
        return current_pos

    if current_pos <= 0.0:
        if p_bull > cfg.bull_aggressive:
            return cfg.leverage_max
        if p_bull > cfg.bull_entry:
            return cfg.leverage_base
        if p_sideways > cfg.sideways_entry and z_score > cfg.sideways_z_floor:
            return cfg.leverage_base
        return 0.0

    # Already invested.
    if p_bear > cfg.bear_exit:
        return 0.0
    if p_bull > cfg.bull_aggressive:
        return cfg.leverage_max
    if current_pos > cfg.leverage_base and p_bull < cfg.bull_deescalate:
        return cfg.leverage_base
    return current_pos


def signal_label(position: float, cfg: StrategyConfig) -> str:
    """Human-readable name for a target exposure."""
    if position <= 0.0:
        return "CASH (no exposure)"
    if position > cfg.leverage_base:
        return f"STRONG BULL - aggressive, {position:.2f}x"
    return f"INVESTED - standard, {position:.2f}x"


def target_positions(df: pd.DataFrame, cfg: StrategyConfig) -> pd.Series:
    """Run the state machine over ``df``, returning the target set on each close.

    The value at row *t* is decided using information available at the close of
    *t* and is therefore earned over the return of *t+1*; see
    :func:`regime_allocation.backtest.run_backtest`, which applies the shift.
    """
    required = {"P_Bear", "P_Sideways", "P_Bull", "Z_Score"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"missing columns for the strategy: {sorted(missing)}")

    bear = df["P_Bear"].to_numpy(dtype=float)
    side = df["P_Sideways"].to_numpy(dtype=float)
    bull = df["P_Bull"].to_numpy(dtype=float)
    z = df["Z_Score"].to_numpy(dtype=float)

    targets = np.empty(len(df))
    pos = 0.0
    for i in range(len(df)):
        pos = next_position(pos, bear[i], side[i], bull[i], z[i], cfg)
        targets[i] = pos
    return pd.Series(targets, index=df.index, name="Target")
