"""Phase 4 pipeline: load cached data → features → target → backtest.

Entry points
------------
    load_aligned_data(cfg, target_ticker) → (features_df, target_series)
    run_pipeline(cfg, target_ticker)      → (results_dict, oos_index)

Both functions assume that raw data and the feature matrix have already been
cached by scripts/fetch_data.py and the Phase 3 build. Pass force_refresh=True
to rebuild from the network.
"""
from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import pandas as pd

from src.config import Config
from src.data.prices import fetch_all_tickers
from src.eval.backtester import Backtester, BacktestResult
from src.eval.splitter import WalkForwardSplitter
from src.features import build_feature_matrix
from src.models.baselines import Drift, Momentum, RandomWalk
from src.models.gbm import LightGBMModel, LightGBMQuantileModel
from src.models.linear import ElasticNetModel
from src.targets import forward_log_return

logger = logging.getLogger(__name__)


def load_aligned_data(
    cfg: Config,
    target_ticker: str = "GC=F",
    force_refresh: bool = False,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load feature matrix and forward-return target, NaN-trimmed.

    The warm-up rows (NaN features from rolling windows) and the last
    `horizon` rows (NaN target — no future close available) are dropped
    so the returned arrays are fully finite and share the same index.

    Returns
    -------
    features : pd.DataFrame  — shape (T, F), all finite
    target   : pd.Series     — shape (T,), named "fwd_ret_{horizon}d"
    """
    feature_df = build_feature_matrix(
        cfg, target_ticker=target_ticker, force_refresh=force_refresh
    )

    prices = fetch_all_tickers(
        [target_ticker],
        cfg.dates["start"],
        cfg.dates["end"],
        cfg.paths.data_raw,
        force_refresh=False,
    )
    close = prices[target_ticker]["Close"]

    target = forward_log_return(close, horizon=cfg.horizon)
    target = target.reindex(feature_df.index)

    valid = feature_df.notna().all(axis=1) & target.notna()
    feature_df = feature_df.loc[valid]
    target = target.loc[valid]

    logger.info(
        "Aligned data: %d rows × %d features (horizon=%d)",
        len(feature_df), feature_df.shape[1], cfg.horizon,
    )
    return feature_df, target


def run_pipeline(
    cfg: Config,
    target_ticker: str = "GC=F",
    force_refresh: bool = False,
) -> tuple[Dict[str, BacktestResult], pd.DatetimeIndex]:
    """Run the walk-forward baseline backtest.

    Returns
    -------
    results   : dict[model_name → BacktestResult]
    oos_index : DatetimeIndex aligned to the concatenated OOS predictions
                (use for equity-curve x-axis)
    """
    features, target = load_aligned_data(
        cfg, target_ticker=target_ticker, force_refresh=force_refresh
    )
    X = features.values
    y = target.values

    splitter = WalkForwardSplitter.from_config(cfg)
    bt = Backtester(splitter=splitter, cost_bps=cfg.cost_bps, horizon=cfg.horizon)

    rng_seed = cfg.random_seed
    factories = {
        "RandomWalk": RandomWalk,
        "Drift": Drift,
        "Momentum": Momentum,
        "ElasticNet": lambda: ElasticNetModel(random_state=rng_seed),
        "LightGBM": lambda: LightGBMModel(random_state=rng_seed),
        "LightGBM_q90": lambda: LightGBMQuantileModel(quantile_alpha=0.9, random_state=rng_seed),
    }

    results: Dict[str, BacktestResult] = {}
    for name, factory in factories.items():
        logger.info("Running %s ...", name)
        results[name] = bt.run(X, y, factory, model_name=name)
        r = results[name]
        sharpe = r.strategy.sharpe
        logger.info(
            "  %s: pooled_IC=%.4f  Sharpe=%.2f  turnover=%.3f",
            name,
            r.pooled_ic,
            sharpe if np.isfinite(sharpe) else float("nan"),
            r.strategy.turnover,
        )

    # OOS date index: test indices from each fold (same order as backtester)
    folds = splitter.split(len(y))
    oos_idx = np.concatenate([te for _, te in folds])
    oos_index = target.index[oos_idx]

    return results, oos_index
