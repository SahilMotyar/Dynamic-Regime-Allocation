"""Tests for feature construction and the walk-forward model wiring.

These run offline against a synthetic price path -- no network access.
"""

import numpy as np
import pandas as pd
import pytest

from regime_allocation.config import TRADING_DAYS, DataConfig, ModelConfig
from regime_allocation.data import FEATURE_COLUMNS, build_features
from regime_allocation.model import attach_beliefs, rolling_regime_beliefs, state_columns

CFG = DataConfig()


def synthetic_prices(n=1600, seed=0):
    """A path that genuinely switches between a calm uptrend and a violent selloff."""
    rng = np.random.default_rng(seed)
    blocks = []
    while sum(len(b) for b in blocks) < n:
        if len(blocks) % 2 == 0:
            blocks.append(rng.normal(0.0008, 0.006, 300))  # bull: drifting, calm
        else:
            blocks.append(rng.normal(-0.0015, 0.025, 150))  # bear: falling, wild
    rets = np.concatenate(blocks)[:n]
    idx = pd.date_range("2010-01-01", periods=n, freq="B")
    return pd.DataFrame({"Close": 1000 * np.cumprod(1 + rets)}, index=idx)


def test_features_are_present_and_finite():
    df = build_features(synthetic_prices(), CFG)
    for col in FEATURE_COLUMNS:
        assert col in df.columns
        assert np.isfinite(df[col]).all()


def test_warmup_rows_are_dropped():
    """The longest lookback (the 50-day trend window) sets the first usable row."""
    prices = synthetic_prices(400)
    df = build_features(prices, CFG)
    assert len(df) == len(prices) - (CFG.trend_window - 1)
    assert df.index[0] == prices.index[CFG.trend_window - 1]


def test_momentum_is_scale_free():
    """Regression: momentum used to be a raw price difference.

    A 14-day *price* change scales with the index level, so the same market
    behaviour produced a feature ten times larger in 2025 than in 2008. Using a
    trailing return makes the two windows comparable.
    """
    prices = synthetic_prices(400)
    base = build_features(prices, CFG)
    scaled = build_features(prices * 1000, CFG)
    pd.testing.assert_series_equal(base["Momentum"], scaled["Momentum"], rtol=1e-9)
    pd.testing.assert_series_equal(base["Z_Score"], scaled["Z_Score"], rtol=1e-9)
    pd.testing.assert_series_equal(base["Volatility"], scaled["Volatility"], rtol=1e-9)


def test_volatility_is_annualised():
    prices = synthetic_prices(400)
    df = build_features(prices, CFG)
    raw = prices["Close"].pct_change()
    manual = (raw.rolling(CFG.vol_window).std() * np.sqrt(TRADING_DAYS)).loc[df.index]
    pd.testing.assert_series_equal(df["Volatility"], manual, check_names=False)


def test_missing_close_column_is_rejected():
    with pytest.raises(KeyError):
        build_features(pd.DataFrame({"Open": [1.0, 2.0]}), CFG)


def test_beliefs_are_a_probability_distribution():
    df = build_features(synthetic_prices(1600), CFG)
    model_cfg = ModelConfig(train_window=750, step=50, n_restarts=1, n_iter=20)
    beliefs = rolling_regime_beliefs(df, model_cfg).dropna()
    assert not beliefs.empty
    assert ((beliefs >= 0) & (beliefs <= 1)).all().all()
    np.testing.assert_allclose(beliefs.sum(axis=1), 1.0, atol=1e-8)


def test_warmup_window_has_no_beliefs():
    """Regression: the warm-up used to be padded with a uniform 1/3 prior.

    That is indistinguishable from a real "no strong view" reading, so the
    strategy could act on days the model had never been trained for.
    """
    df = build_features(synthetic_prices(1600), CFG)
    model_cfg = ModelConfig(train_window=750, step=50, n_restarts=1, n_iter=20)
    beliefs = rolling_regime_beliefs(df, model_cfg)
    assert beliefs.iloc[:750].isna().all().all()


def test_states_are_ordered_bear_to_bull():
    """The bull state must carry the higher average trend, in every refit."""
    df = build_features(synthetic_prices(1600), CFG)
    model_cfg = ModelConfig(train_window=750, step=50, n_restarts=2, n_iter=30)
    out = attach_beliefs(df, model_cfg)
    bull_z = np.average(out["Z_Score"], weights=out["P_Bull"])
    bear_z = np.average(out["Z_Score"], weights=out["P_Bear"])
    assert bull_z > bear_z


def test_too_short_a_history_is_rejected_with_a_useful_message():
    df = build_features(synthetic_prices(400), CFG)
    with pytest.raises(ValueError, match="train-window"):
        rolling_regime_beliefs(df, ModelConfig(train_window=5000))


def test_state_columns_naming():
    assert state_columns(3) == ["P_Bear", "P_Sideways", "P_Bull"]
    assert state_columns(4) == [f"P_State_{i}" for i in range(4)]


def test_model_config_validates_its_inputs():
    with pytest.raises(ValueError):
        ModelConfig(step=0)
    with pytest.raises(ValueError):
        ModelConfig(n_states=1)
    with pytest.raises(ValueError):
        ModelConfig(n_restarts=0)
