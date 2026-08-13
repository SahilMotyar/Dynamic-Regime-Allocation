"""Console output: the live trade card and the equity/risk chart."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import StrategyConfig
from .metrics import drawdown_curve
from .strategy import signal_label


def format_trade_card(results: pd.DataFrame, cfg: StrategyConfig) -> str:
    """Render the actionable signal for the next session.

    The exposure shown is ``Target`` from the final row -- the same number the
    backtest would have carried into the next bar -- rather than a second,
    independently written set of if-statements. When the two were separate the
    card could advertise a 1.5x "aggressive buy" that the backtest never took.
    """
    last = results.iloc[-1]
    held = float(results["Position"].iloc[-1])
    target = float(last["Target"])

    if target > held:
        action = "INCREASE exposure" if held > 0 else "ENTER"
    elif target < held:
        action = "REDUCE exposure" if target > 0 else "EXIT to cash"
    else:
        action = "HOLD"

    width = 58
    lines = [
        "=" * width,
        f"LIVE SIGNAL - {results.index[-1].date()}".center(width),
        "=" * width,
        f"  Last close            {last['Close']:>16,.2f}",
        f"  Trend (Z-score)       {last['Z_Score']:>16.2f} sigma",
        f"  Volatility (ann.)     {last['Volatility'] * 100:>15.1f}%",
        "-" * width,
        f"  Regime beliefs (smoothed):",
        f"    Bear                {last['P_Bear'] * 100:>15.1f}%",
        f"    Sideways            {last['P_Sideways'] * 100:>15.1f}%",
        f"    Bull                {last['P_Bull'] * 100:>15.1f}%",
        "-" * width,
        f"  Currently held        {held:>15.2f}x",
        f"  Target exposure       {target:>15.2f}x",
        f"  Regime               {signal_label(target, cfg):>16}",
        f"  Action               {action:>16}",
        "=" * width,
        "  Educational research output, not financial advice.",
    ]
    return "\n".join(lines)


def plot_results(
    results: pd.DataFrame,
    save_path: Path | str | None = None,
    show: bool = True,
) -> None:
    """Three-panel chart: growth of capital, drawdowns, and regime beliefs."""
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        4, 1, figsize=(13, 11), sharex=True, gridspec_kw={"height_ratios": [3, 2, 2, 1]}
    )

    ax = axes[0]
    ax.plot(results.index, results["Equity_Market"], color="grey", alpha=0.8, label="Buy & hold")
    ax.plot(
        results.index, results["Equity_Strategy"], color="#1f4fd8", linewidth=1.8, label="Strategy (net)"
    )
    ax.set_yscale("log")
    ax.set_ylabel("Growth of 1 (log)")
    ax.set_title("Dynamic Regime Allocation - net of costs")
    ax.legend(loc="upper left")
    ax.grid(True, which="both", alpha=0.25)

    ax = axes[1]
    ax.fill_between(
        results.index, drawdown_curve(results["Equity_Market"]) * 100, 0, color="grey", alpha=0.45,
        label="Buy & hold",
    )
    ax.plot(
        results.index, drawdown_curve(results["Equity_Strategy"]) * 100, color="#1f4fd8",
        linewidth=1.2, label="Strategy",
    )
    ax.set_ylabel("Drawdown (%)")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.25)

    ax = axes[2]
    ax.stackplot(
        results.index,
        results["P_Bear"], results["P_Sideways"], results["P_Bull"],
        colors=["#c0392b", "#e0a80d", "#2e8b57"],
        labels=["Bear", "Sideways", "Bull"],
        alpha=0.65,
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel("Regime belief")
    ax.legend(loc="upper left", ncol=3, fontsize=8)
    ax.grid(True, alpha=0.2)

    # Exposure gets its own panel: overlaid on the belief stack, the 0/1 flips
    # drew a vertical line at every transition and read as a picket fence.
    ax = axes[3]
    ax.fill_between(
        results.index, results["Position"], 0, step="post", color="#1f4fd8", alpha=0.55
    )
    ax.set_ylim(0, max(float(results["Position"].max()) * 1.15, 1.15))
    ax.set_ylabel("Exposure")
    ax.set_xlabel("")
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=130, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
