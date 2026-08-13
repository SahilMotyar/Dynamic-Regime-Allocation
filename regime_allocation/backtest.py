"""Fee-aware PnL engine.

Accounting conventions, stated once so the numbers are auditable:

* A target set on the close of day *t* is held over day *t+1* and earns day
  *t+1*'s return. ``Position`` is therefore ``Target.shift(1)``; there is no way
  for a bar's own return to inform the exposure that captures it.
* Un-invested capital (``1 - Position``, when positive) accrues ``cash_rate``.
* Exposure above 1x is financed at ``cash_rate + borrow_spread``. The original
  version credited cash yield but charged nothing for leverage, which flattered
  every levered day.
* Trading friction is ``tx_cost`` per unit of exposure turned over, charged on
  the day the new exposure takes effect.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import TRADING_DAYS, BacktestConfig, StrategyConfig
from .strategy import target_positions


def run_backtest(
    df: pd.DataFrame,
    strategy_cfg: StrategyConfig,
    backtest_cfg: BacktestConfig,
) -> pd.DataFrame:
    """Return ``df`` extended with positions, per-day PnL components and equity."""
    if "Returns" not in df.columns:
        raise KeyError("frame must contain a 'Returns' column")

    out = df.copy()
    out["Target"] = target_positions(out, strategy_cfg)
    out["Position"] = out["Target"].shift(1).fillna(0.0)

    daily_cash = backtest_cfg.cash_rate / TRADING_DAYS
    daily_borrow = (backtest_cfg.cash_rate + backtest_cfg.borrow_spread) / TRADING_DAYS

    out["Gross_Ret"] = out["Position"] * out["Returns"]
    out["Cash_Yield"] = np.maximum(0.0, 1.0 - out["Position"]) * daily_cash
    out["Borrow_Cost"] = np.maximum(0.0, out["Position"] - 1.0) * daily_borrow
    out["Tx_Cost"] = out["Position"].diff().abs().fillna(out["Position"].abs()) * backtest_cfg.tx_cost

    out["Net_Ret"] = out["Gross_Ret"] + out["Cash_Yield"] - out["Borrow_Cost"] - out["Tx_Cost"]

    out["Equity_Strategy"] = (1.0 + out["Net_Ret"]).cumprod()
    out["Equity_Market"] = (1.0 + out["Returns"]).cumprod()
    return out
