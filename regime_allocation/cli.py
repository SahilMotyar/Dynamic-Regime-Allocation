"""Command-line entry point: fetch, fit, backtest, report."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .backtest import run_backtest
from .config import BacktestConfig, DataConfig, ModelConfig, StrategyConfig
from .data import load_dataset
from .metrics import comparison_table, summarize
from .model import attach_beliefs
from .reporting import format_trade_card, plot_results

logger = logging.getLogger("regime_allocation")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="regime-allocation",
        description="Regime-switching allocation on a single index, net of costs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    data = p.add_argument_group("data")
    data.add_argument("--ticker", default="^NSEI", help="Yahoo Finance symbol")
    data.add_argument("--start", default="2007-01-01")
    data.add_argument("--end", default=None, help="exclusive end date; defaults to today")
    data.add_argument("--cache-dir", default="data_cache", help="set to '' to disable caching")
    data.add_argument("--refresh", action="store_true", help="ignore cached prices")

    model = p.add_argument_group("model")
    model.add_argument("--train-window", type=int, default=1250)
    model.add_argument("--step", type=int, default=10, help="days between refits")
    model.add_argument("--smoothing", type=int, default=10, help="MA applied to beliefs")
    model.add_argument("--restarts", type=int, default=3, help="EM restarts per window")
    model.add_argument("--seed", type=int, default=42)

    strat = p.add_argument_group("strategy")
    strat.add_argument("--leverage-base", type=float, default=1.0)
    strat.add_argument(
        "--leverage-max",
        type=float,
        default=1.0,
        help="exposure in a strong bull; 1.0 disables the aggressive tier",
    )
    strat.add_argument("--bull-entry", type=float, default=0.60)
    strat.add_argument("--bull-aggressive", type=float, default=0.80)
    strat.add_argument("--bull-deescalate", type=float, default=0.75)
    strat.add_argument("--bear-exit", type=float, default=0.60)

    costs = p.add_argument_group("costs")
    costs.add_argument("--tx-cost", type=float, default=0.001, help="per unit of exposure traded")
    costs.add_argument("--cash-rate", type=float, default=0.06, help="annualised yield on cash")
    costs.add_argument(
        "--borrow-spread", type=float, default=0.02, help="over cash rate, on exposure above 1x"
    )

    out = p.add_argument_group("output")
    out.add_argument("--no-plot", action="store_true")
    out.add_argument("--save-plot", default=None, help="write the chart to this path")
    out.add_argument("--save-csv", default=None, help="write the full result frame to this path")
    out.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    data_cfg = DataConfig(ticker=args.ticker, start=args.start, end=args.end)
    model_cfg = ModelConfig(
        train_window=args.train_window,
        step=args.step,
        smoothing_window=args.smoothing,
        n_restarts=args.restarts,
        seed=args.seed,
    )
    strategy_cfg = StrategyConfig(
        leverage_base=args.leverage_base,
        leverage_max=args.leverage_max,
        bull_entry=args.bull_entry,
        bull_aggressive=args.bull_aggressive,
        bull_deescalate=args.bull_deescalate,
        bear_exit=args.bear_exit,
    )
    backtest_cfg = BacktestConfig(
        tx_cost=args.tx_cost, cash_rate=args.cash_rate, borrow_spread=args.borrow_spread
    )

    try:
        df = load_dataset(data_cfg, cache_dir=args.cache_dir or None, refresh=args.refresh)
        df = attach_beliefs(df, model_cfg)
    except (RuntimeError, ValueError, KeyError) as exc:
        logger.error("%s", exc)
        return 1

    results = run_backtest(df, strategy_cfg, backtest_cfg)

    strategy_stats = summarize(
        results["Net_Ret"],
        risk_free_rate=backtest_cfg.cash_rate,
        positions=results["Position"],
        costs=results["Tx_Cost"],
        label="Strategy",
    )
    market_stats = summarize(
        results["Returns"], risk_free_rate=backtest_cfg.cash_rate, label="Buy & hold"
    )

    print()
    print(comparison_table(strategy_stats, market_stats))
    print()
    print(format_trade_card(results, strategy_cfg))

    if args.save_csv:
        path = Path(args.save_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(path)
        logger.info("Wrote %s", path)

    if args.save_plot or not args.no_plot:
        plot_results(results, save_path=args.save_plot, show=not args.no_plot)

    return 0


if __name__ == "__main__":
    sys.exit(main())
