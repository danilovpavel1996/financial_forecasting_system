"""Feature matrix builder — assembles price + macro features and caches to parquet.

This is the Phase 3 entry point. It loads cached raw data, delegates to
price_features and macro_features, joins the results, and caches to
data/processed/.

Usage
-----
    from src.config import load_config
    from src.features import build_feature_matrix

    cfg = load_config()
    feature_df = build_feature_matrix(cfg, target_ticker="GC=F")
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.config import Config
from src.data.prices import fetch_all_tickers
from src.data.macro import fetch_all_series, _get_api_key
from src.data import universe
from src.features.price_features import build_price_features
from src.features.macro_features import build_macro_features

logger = logging.getLogger(__name__)


def build_feature_matrix(
    cfg: Config,
    target_ticker: str = "GC=F",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Build and cache the full feature matrix for a target ticker.

    Steps
    -----
    1. Load (or fetch) all price and macro data from cache.
    2. Compute price features for the target ticker plus context assets.
    3. Compute macro features aligned to the target's trading-day index.
    4. Join price + macro features on the common index.
    5. Assert no duplicate dates, assert index alignment.
    6. Cache result to data/processed/.

    Returns
    -------
    pd.DataFrame indexed by the target ticker's trading-day DatetimeIndex.
    Columns are raw feature values — no normalization applied.
    """
    cache_path = _feature_cache_path(cfg, target_ticker)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists() and not force_refresh:
        logger.info("Feature matrix cache hit: %s", cache_path)
        df = pd.read_parquet(cache_path)
        logger.info("Loaded feature matrix: %d rows, %d cols", len(df), df.shape[1])
        return df

    logger.info("Building feature matrix for %s ...", target_ticker)

    # --- Load raw data ---
    all_tickers = universe.price_tickers(cfg)
    prices = fetch_all_tickers(
        all_tickers, cfg.dates["start"], cfg.dates["end"],
        cfg.paths.data_raw, force_refresh=False,
    )

    import os
    api_key = os.environ.get("FRED_API_KEY", "").strip()
    macro_raw = fetch_all_series(
        universe.fred_series(cfg),
        cfg.dates["start"], cfg.dates["end"],
        cfg.paths.data_raw,
        api_key=api_key or None,
        force_refresh=False,
    )

    if target_ticker not in prices:
        raise KeyError(
            f"Target ticker {target_ticker!r} not in fetched prices. "
            f"Available: {list(prices.keys())}"
        )

    trading_index = prices[target_ticker].index

    # --- Price features ---
    context_tickers = [t for t in all_tickers if t != target_ticker]
    # ETFs and equity-market indices close at 4pm ET, after GC=F (~2:30pm ET).
    # Their same-day 1-day return contains post-futures-close price movement, which
    # is forward-looking relative to the futures settlement price.
    late_close = universe.equity_context_tickers(cfg)
    price_feats = build_price_features(
        prices,
        target_ticker=target_ticker,
        context_tickers=context_tickers,
        late_close_tickers=late_close,
        ratio_pair=("GC=F", "SI=F"),
        ratio_name="gold_silver_ratio",
    )

    # --- Macro features ---
    macro_feats = build_macro_features(macro_raw, trading_index=trading_index)

    # --- Join ---
    feature_df = price_feats.join(macro_feats, how="left")
    feature_df = feature_df.sort_index()

    # --- Assertions ---
    dupes = feature_df.index[feature_df.index.duplicated()]
    if len(dupes):
        raise ValueError(f"Duplicate dates in feature matrix: {dupes.tolist()[:5]}")

    if not feature_df.index.equals(trading_index):
        raise ValueError("Feature matrix index does not match target ticker's index.")

    # --- Cache ---
    feature_df.to_parquet(cache_path)
    logger.info(
        "Cached feature matrix: %d rows, %d cols → %s",
        len(feature_df), feature_df.shape[1], cache_path,
    )

    _log_nan_summary(feature_df)
    return feature_df


def _feature_cache_path(cfg: Config, target_ticker: str) -> Path:
    safe = target_ticker.replace("=", "_eq_").replace("^", "_hat_").replace("-", "_")
    start = cfg.dates["start"]
    end = cfg.dates["end"]
    return cfg.paths.data_processed / f"features_{safe}_{start}_{end}.parquet"


def _log_nan_summary(df: pd.DataFrame) -> None:
    """Log the NaN count and first valid index for each column."""
    for col in df.columns:
        n_nan = df[col].isna().sum()
        if n_nan > 0:
            first_valid = df[col].first_valid_index()
            logger.debug(
                "  %s: %d NaN rows (first valid: %s)",
                col, n_nan,
                first_valid.date() if first_valid is not None else "never",
            )
