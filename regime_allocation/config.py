"""Configuration objects for the regime-allocation pipeline.

Every tunable number in the strategy lives here rather than being hard-coded at
its point of use, so that a backtest and the live signal are guaranteed to run
under identical assumptions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

TRADING_DAYS = 252


@dataclass(frozen=True)
class DataConfig:
    """Where the price series comes from and how features are derived."""

    ticker: str = "^NSEI"
    start: str = "2007-01-01"
    end: str | None = None
    vol_window: int = 20
    trend_window: int = 50
    momentum_window: int = 14

    def __post_init__(self) -> None:
        for name in ("vol_window", "trend_window", "momentum_window"):
            if getattr(self, name) < 2:
                raise ValueError(f"{name} must be >= 2")


@dataclass(frozen=True)
class ModelConfig:
    """Walk-forward Gaussian HMM settings."""

    n_states: int = 3
    train_window: int = 1250  # ~5 years of trading days
    step: int = 10  # refit cadence, in trading days
    n_iter: int = 100
    n_restarts: int = 3  # best-of-N by in-sample log-likelihood
    covariance_type: str = "diag"
    smoothing_window: int = 10  # moving average applied to regime beliefs
    seed: int = 42

    def __post_init__(self) -> None:
        if self.n_states < 2:
            raise ValueError("n_states must be >= 2")
        if self.step < 1:
            raise ValueError("step must be >= 1")
        if self.train_window <= self.n_states:
            raise ValueError("train_window must be larger than n_states")
        if self.n_restarts < 1:
            raise ValueError("n_restarts must be >= 1")
        if self.smoothing_window < 1:
            raise ValueError("smoothing_window must be >= 1")


@dataclass(frozen=True)
class StrategyConfig:
    """Probability thresholds that map regime beliefs onto a target exposure.

    The two bull thresholds form a hysteresis band: exposure is levered up once
    ``P(Bull) > bull_aggressive`` and only de-levered once it falls back below
    ``bull_deescalate``. Without that gap the top tier would flip on and off
    around a single threshold and pay transaction costs each time.

    ``leverage_max`` defaults to 1.0, i.e. the aggressive tier is *off*. It is
    supported and correct -- pass ``--leverage-max 1.5`` to enable it -- but on
    Nifty it measures worse on every risk-adjusted basis once financing and the
    extra turnover it creates are charged for. See the README.
    """

    leverage_base: float = 1.0
    leverage_max: float = 1.0
    bull_entry: float = 0.60
    bull_aggressive: float = 0.80
    bull_deescalate: float = 0.75
    bear_exit: float = 0.60
    sideways_entry: float = 0.60
    sideways_z_floor: float = -0.5

    def __post_init__(self) -> None:
        if not self.bull_entry < self.bull_deescalate <= self.bull_aggressive:
            raise ValueError(
                "thresholds must satisfy bull_entry < bull_deescalate <= bull_aggressive"
            )
        if self.leverage_max < self.leverage_base:
            raise ValueError("leverage_max must be >= leverage_base")
        for name in ("bull_entry", "bull_aggressive", "bear_exit", "sideways_entry"):
            value = getattr(self, name)
            if not 0.0 < value < 1.0:
                raise ValueError(f"{name} must lie strictly between 0 and 1")


@dataclass(frozen=True)
class BacktestConfig:
    """Cost and carry assumptions for the PnL engine."""

    tx_cost: float = 0.001  # round-trip friction per unit of exposure traded
    cash_rate: float = 0.06  # annualised yield on un-invested capital
    borrow_spread: float = 0.02  # charged above cash_rate on exposure beyond 1x

    def __post_init__(self) -> None:
        if self.tx_cost < 0 or self.cash_rate < 0 or self.borrow_spread < 0:
            raise ValueError("cost parameters must be non-negative")


@dataclass(frozen=True)
class RunConfig:
    """Everything needed to reproduce a single run."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
