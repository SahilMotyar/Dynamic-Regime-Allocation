"""Risk and performance statistics.

The original script reported only total return, which cannot support a claim
about *risk-adjusted* performance -- a levered strategy can beat the index on
total return while being strictly worse to hold. These are the statistics that
actually settle that question.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import TRADING_DAYS


def drawdown_curve(equity: pd.Series) -> pd.Series:
    """Fractional drawdown from the running peak, as a negative series."""
    return equity / equity.cummax() - 1.0


def max_drawdown(equity: pd.Series) -> float:
    return float(drawdown_curve(equity).min())


def longest_drawdown_days(equity: pd.Series) -> int:
    """Longest stretch, in calendar days, spent below a previous peak."""
    underwater = equity < equity.cummax()
    if not underwater.any():
        return 0
    longest = pd.Timedelta(0)
    start = None
    for timestamp, is_under in underwater.items():
        if is_under and start is None:
            start = timestamp
        elif not is_under and start is not None:
            longest = max(longest, timestamp - start)
            start = None
    if start is not None:
        longest = max(longest, underwater.index[-1] - start)
    return int(longest.days)


def summarize(
    returns: pd.Series,
    *,
    risk_free_rate: float = 0.0,
    positions: pd.Series | None = None,
    costs: pd.Series | None = None,
    label: str = "strategy",
) -> dict[str, float | str]:
    """Compute a full statistics block from a series of daily net returns."""
    returns = returns.dropna()
    if returns.empty:
        raise ValueError(f"no returns to summarise for {label!r}")

    equity = (1.0 + returns).cumprod()
    years = max((returns.index[-1] - returns.index[0]).days / 365.25, 1e-9)
    total_return = float(equity.iloc[-1]) - 1.0
    cagr = (1.0 + total_return) ** (1.0 / years) - 1.0

    vol = float(returns.std(ddof=1)) * np.sqrt(TRADING_DAYS)
    excess = returns - risk_free_rate / TRADING_DAYS
    sharpe = (
        float(excess.mean()) / float(returns.std(ddof=1)) * np.sqrt(TRADING_DAYS)
        if returns.std(ddof=1) > 0
        else float("nan")
    )
    downside = returns[returns < 0]
    downside_vol = float(downside.std(ddof=1)) * np.sqrt(TRADING_DAYS) if len(downside) > 1 else 0.0
    sortino = (
        float(excess.mean()) * TRADING_DAYS / downside_vol if downside_vol > 0 else float("nan")
    )

    mdd = max_drawdown(equity)
    stats: dict[str, float | str] = {
        "label": label,
        "start": str(returns.index[0].date()),
        "end": str(returns.index[-1].date()),
        "years": years,
        "total_return": total_return,
        "cagr": cagr,
        "volatility": vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": mdd,
        "calmar": cagr / abs(mdd) if mdd < 0 else float("nan"),
        "longest_drawdown_days": longest_drawdown_days(equity),
        "hit_rate": float((returns > 0).mean()),
        "best_day": float(returns.max()),
        "worst_day": float(returns.min()),
    }

    if positions is not None:
        positions = positions.reindex(returns.index)
        changes = positions.diff().fillna(positions.abs())
        stats["time_in_market"] = float((positions > 0).mean())
        stats["avg_exposure"] = float(positions.mean())
        stats["trades"] = int((changes.abs() > 1e-12).sum())
        stats["turnover"] = float(changes.abs().sum())
    if costs is not None:
        stats["total_costs"] = float(costs.reindex(returns.index).fillna(0.0).sum())

    return stats


_ROWS: list[tuple[str, str, str]] = [
    ("Period", "start", "range"),
    ("Total return", "total_return", "pct"),
    ("CAGR", "cagr", "pct"),
    ("Volatility (ann.)", "volatility", "pct"),
    ("Sharpe (vs cash)", "sharpe", "num"),
    ("Sortino", "sortino", "num"),
    ("Max drawdown", "max_drawdown", "pct"),
    ("Calmar", "calmar", "num"),
    ("Longest underwater", "longest_drawdown_days", "days"),
    ("Positive days", "hit_rate", "pct"),
    ("Worst day", "worst_day", "pct"),
    ("Time in market", "time_in_market", "pct"),
    ("Average exposure", "avg_exposure", "x"),
    ("Trades", "trades", "int"),
    ("Turnover (sum |dPos|)", "turnover", "num"),
    ("Costs paid (of capital)", "total_costs", "pct"),
]


def _format(value, kind: str, stats: dict) -> str:
    if value is None:
        return "-"
    if kind == "range":
        return f"{stats['start']} to {stats['end']}"
    if isinstance(value, float) and not np.isfinite(value):
        return "n/a"
    if kind == "pct":
        return f"{value * 100:,.2f}%"
    if kind == "num":
        return f"{value:,.2f}"
    if kind == "x":
        return f"{value:,.2f}x"
    if kind == "days":
        return f"{int(value)} d"
    if kind == "int":
        return f"{int(value):,}"
    return str(value)


def comparison_table(*stat_blocks: dict) -> str:
    """Render two or more statistics blocks side by side as a fixed-width table."""
    labels = [str(block.get("label", f"col{i}")) for i, block in enumerate(stat_blocks)]
    width = max(26, *(len(label) + 2 for label in labels))  # 26 fits "YYYY-MM-DD to YYYY-MM-DD"
    lines = ["Metric".ljust(26) + "".join(label.rjust(width) for label in labels)]
    lines.append("-" * len(lines[0]))
    for title, key, kind in _ROWS:
        if not any(key in block for block in stat_blocks):
            continue
        cells = "".join(
            _format(block.get(key), kind, block).rjust(width) for block in stat_blocks
        )
        lines.append(title.ljust(26) + cells)
    return "\n".join(lines)
