"""Walk-forward Gaussian HMM producing daily regime beliefs.

The model is refit every ``step`` days on a trailing window and is only ever
asked to score the ``step`` days that follow that window. Nothing in this module
looks at a bar it was not trained before: the scaler, the HMM parameters and the
bear/sideways/bull state ordering are all derived from the training slice alone.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .config import ModelConfig
from .data import FEATURE_COLUMNS, TREND_FEATURE

logger = logging.getLogger(__name__)

STATE_NAMES_3 = ["P_Bear", "P_Sideways", "P_Bull"]


def state_columns(n_states: int) -> list[str]:
    """Column names for the belief frame, ordered from most bearish to most bullish."""
    if n_states == 3:
        return list(STATE_NAMES_3)
    return [f"P_State_{i}" for i in range(n_states)]


def _fit_best_hmm(x: np.ndarray, cfg: ModelConfig):
    """Fit ``n_restarts`` HMMs from different seeds, keep the highest likelihood.

    EM on a Gaussian HMM is only locally optimal, so a single seed can land on a
    degenerate solution that then propagates into the next 10 days of signals.
    """
    from hmmlearn.hmm import GaussianHMM

    best, best_score = None, -np.inf
    for offset in range(cfg.n_restarts):
        model = GaussianHMM(
            n_components=cfg.n_states,
            covariance_type=cfg.covariance_type,
            n_iter=cfg.n_iter,
            random_state=cfg.seed + offset,
        )
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(x)
                score = model.score(x)
        except (ValueError, np.linalg.LinAlgError) as exc:
            logger.debug("HMM restart %d failed: %s", offset, exc)
            continue
        if np.isfinite(score) and score > best_score:
            best, best_score = model, score
    return best


def _order_states(model, feature_names: list[str]) -> list[int]:
    """Return state indices sorted from most bearish to most bullish.

    Latent state labels are arbitrary and change between refits, so they are
    re-identified every window by the state's mean trend feature. Ranking uses
    the fitted means rather than the empirical means of assigned points, which
    keeps the ordering well defined even when a state is never the Viterbi
    winner in-sample. Standardisation is strictly increasing per feature, so the
    ordering is the same in scaled and raw space.
    """
    trend_idx = feature_names.index(TREND_FEATURE)
    return list(np.argsort(model.means_[:, trend_idx], kind="stable"))


def rolling_regime_beliefs(
    df: pd.DataFrame,
    cfg: ModelConfig,
    feature_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Compute out-of-sample regime probabilities for every row of ``df``.

    Rows inside the initial training window -- and any block whose fit failed --
    are returned as NaN rather than filled with a uniform prior, so that a
    warm-up period is never mistaken for a genuine "no opinion" signal.
    """
    feature_cols = feature_cols or FEATURE_COLUMNS
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise KeyError(f"missing feature columns: {missing}")
    if len(df) <= cfg.train_window:
        raise ValueError(
            f"need more than {cfg.train_window} rows to train; got {len(df)}. "
            "Use a longer history or a smaller --train-window."
        )

    features = df[feature_cols].to_numpy(dtype=float)
    beliefs = np.full((len(df), cfg.n_states), np.nan)
    failures = 0

    for i in range(cfg.train_window, len(df), cfg.step):
        train = features[i - cfg.train_window : i]
        scaler = StandardScaler().fit(train)

        model = _fit_best_hmm(scaler.transform(train), cfg)
        if model is None:
            failures += 1
            logger.warning("HMM fit failed for window ending %s", df.index[i - 1].date())
            continue

        end = min(i + cfg.step, len(df))
        probs = model.predict_proba(scaler.transform(features[i:end]))
        beliefs[i:end] = probs[:, _order_states(model, feature_cols)]

    if failures:
        logger.warning("%d of the walk-forward windows failed to fit", failures)

    out = pd.DataFrame(beliefs, index=df.index, columns=state_columns(cfg.n_states))
    return out.rolling(cfg.smoothing_window, min_periods=cfg.smoothing_window).mean()


def attach_beliefs(df: pd.DataFrame, cfg: ModelConfig) -> pd.DataFrame:
    """Run the walk-forward model and return ``df`` joined with its beliefs.

    Warm-up rows are dropped, so the returned frame starts on the first day the
    strategy could actually have traded.
    """
    beliefs = rolling_regime_beliefs(df, cfg)
    out = df.join(beliefs).dropna(subset=list(beliefs.columns))
    if out.empty:
        raise ValueError("no rows with valid regime beliefs; check the model settings")
    logger.info(
        "Regime beliefs available for %d rows (%s to %s)",
        len(out),
        out.index[0].date(),
        out.index[-1].date(),
    )
    return out
