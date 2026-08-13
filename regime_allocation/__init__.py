"""Dynamic Regime Allocation: a fee-aware, regime-switching allocation strategy."""

from .config import BacktestConfig, DataConfig, ModelConfig, StrategyConfig
from .strategy import next_position, signal_label

__version__ = "0.2.0"

__all__ = [
    "BacktestConfig",
    "DataConfig",
    "ModelConfig",
    "StrategyConfig",
    "next_position",
    "signal_label",
    "__version__",
]
