"""Price download, on-disk caching, and feature engineering."""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .config import TRADING_DAYS, DataConfig

logger = logging.getLogger(__name__)

FEATURE_COLUMNS = ["Volatility", "Z_Score", "Momentum"]
TREND_FEATURE = "Z_Score"  # the feature used to order latent states bear -> bull


def _cache_path(cache_dir: Path, cfg: DataConfig) -> Path:
    ticker = cfg.ticker.replace("^", "").replace("/", "_")
    end = cfg.end or "latest"
    return cache_dir / f"{ticker}_{cfg.start}_{end}.csv"


def download_prices(
    cfg: DataConfig,
    cache_dir: Path | str | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Return daily OHLCV for ``cfg.ticker`` with a split/dividend-adjusted Close.

    Results are cached to ``cache_dir`` and reused for the rest of the calendar
    day, so repeated runs (and the test suite) do not hammer the data provider.
    """
    cache_file = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = _cache_path(cache_dir, cfg)
        if cache_file.exists() and not refresh:
            stale = date.fromtimestamp(cache_file.stat().st_mtime) < date.today()
            if not stale:
                logger.info("Loading cached prices from %s", cache_file)
                return pd.read_csv(cache_file, index_col=0, parse_dates=True)

    import yfinance as yf  # imported lazily so the package works offline

    logger.info("Fetching %s from Yahoo Finance...", cfg.ticker)
    df = yf.download(
        cfg.ticker,
        start=cfg.start,
        end=cfg.end,
        interval="1d",
        auto_adjust=True,  # 'Close' is then already total-return adjusted
        progress=False,
    )
    if df is None or df.empty:
        raise RuntimeError(
            f"No data returned for {cfg.ticker!r} between {cfg.start} and {cfg.end or 'today'}"
        )

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.ffill().dropna(subset=["Close"])

    if cache_file is not None:
        df.to_csv(cache_file)
        logger.info("Cached %d rows to %s", len(df), cache_file)
    return df


def build_features(prices: pd.DataFrame, cfg: DataConfig) -> pd.DataFrame:
    """Derive the three model inputs from an adjusted close series.

    All three are deliberately scale-free, so that a window from 2008 (Nifty
    ~3,000) and one from 2025 (Nifty ~25,000) live on the same axes:

    * ``Volatility`` -- annualised rolling stdev of daily returns (risk proxy).
    * ``Z_Score``    -- distance of price from its moving average, in sigmas.
    * ``Momentum``   -- trailing return over the momentum window, not a raw
      price difference; a 14-day *price* change grows with the index level and
      would make the feature non-stationary across the sample.
    """
    if "Close" not in prices.columns:
        raise KeyError("price frame must contain a 'Close' column")

    df = prices.copy()
    close = df["Close"].astype(float)

    df["Returns"] = close.pct_change()
    df["Volatility"] = df["Returns"].rolling(cfg.vol_window).std() * np.sqrt(TRADING_DAYS)

    sma = close.rolling(cfg.trend_window).mean()
    sd = close.rolling(cfg.trend_window).std()
    df["Z_Score"] = (close - sma) / sd.replace(0.0, np.nan)

    df["Momentum"] = close.pct_change(cfg.momentum_window)

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["Returns", *FEATURE_COLUMNS])
    if df.empty:
        raise ValueError("no rows survived feature construction; check the date range")
    return df


def load_dataset(
    cfg: DataConfig,
    cache_dir: Path | str | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Download (or load from cache) and featurise in one call."""
    return build_features(download_prices(cfg, cache_dir=cache_dir, refresh=refresh), cfg)
