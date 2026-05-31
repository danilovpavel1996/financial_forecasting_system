"""Cross-sectional ranking pipeline.

Entry point: run_ranking_pipeline(cfg, horizon) → dict[name → RankingResult]

This mirrors pipeline.py but operates on the pooled (date × asset) dataset
and uses RankingBacktester instead of the single-asset Backtester.
"""
from __future__ import annotations

import logging
import os
from typing import Dict

import numpy as np

from src.config import Config
from src.data import universe
from src.data.macro import fetch_all_series
from src.data.prices import fetch_all_tickers
from src.eval.rank_backtester import (
    EqualWeightRanker,
    MeanReversionRanker,
    MomentumRankRanker,
    RankingBacktester,
    RankingResult,
)
from src.eval.splitter import WalkForwardSplitter
from src.features.pooled_dataset import build_pooled_dataset, feature_cols
from src.models.gbm import LightGBMModel
from src.models.linear import ElasticNetModel

logger = logging.getLogger(__name__)


def run_ranking_pipeline(
    cfg: Config,
    horizon: int,
    force_refresh: bool = False,
) -> Dict[str, RankingResult]:
    """Build pooled dataset and run walk-forward ranking backtest.

    Parameters
    ----------
    cfg:           project config.
    horizon:       forecast horizon in trading days (1 or 5).
    force_refresh: if True, re-fetch data from network (ignores cache).

    Returns
    -------
    dict mapping model name → RankingResult.
    """
    # ── Load raw data ────────────────────────────────────────────────────────
    all_tickers = universe.price_tickers(cfg)
    prices = fetch_all_tickers(
        all_tickers,
        cfg.dates["start"],
        cfg.dates["end"],
        cfg.paths.data_raw,
        force_refresh=force_refresh,
    )

    api_key = os.environ.get("FRED_API_KEY", "").strip()
    macro_raw = fetch_all_series(
        universe.fred_series(cfg),
        cfg.dates["start"],
        cfg.dates["end"],
        cfg.paths.data_raw,
        api_key=api_key or None,
        force_refresh=False,
    )

    # ── Build pooled dataset ─────────────────────────────────────────────────
    pooled = build_pooled_dataset(cfg, prices, macro_raw, horizon=horizon)

    fcols = feature_cols(pooled)
    n_feats = len(fcols)
    logger.info(
        "Pooled dataset: %d rows × %d features, horizon=%d",
        len(pooled), n_feats, horizon,
    )

    # ── Feature-column indices for ranking baselines ─────────────────────────
    mom_21d_idx = fcols.index("mom_21d") if "mom_21d" in fcols else 0
    ret_5d_idx  = fcols.index("ret_5d")  if "ret_5d"  in fcols else 1

    # ── Splitter ─────────────────────────────────────────────────────────────
    splitter = WalkForwardSplitter.from_config(cfg)

    metals = universe.metal_tickers(cfg)
    bt = RankingBacktester(
        splitter=splitter,
        cost_bps=cfg.cost_bps,
        horizon=horizon,
        assets=sorted(metals),
    )

    rng_seed = cfg.random_seed
    factories = {
        "EqualWeight":   EqualWeightRanker,
        "MomentumRank":  lambda: MomentumRankRanker(feature_idx=mom_21d_idx),
        "MeanReversion": lambda: MeanReversionRanker(feature_idx=ret_5d_idx),
        "ElasticNet":    lambda: ElasticNetModel(random_state=rng_seed),
        "LightGBM":      lambda: LightGBMModel(random_state=rng_seed),
    }

    results: Dict[str, RankingResult] = {}
    for name, factory in factories.items():
        logger.info("Running %s (horizon=%d) ...", name, horizon)
        r = bt.run(pooled, factory, model_name=name)
        results[name] = r
        logger.info(
            "  %s: CS_RIC=%.4f  stability=%.2f  Sharpe=%.2f  turnover=%.3f",
            name,
            r.mean_cs_ric if np.isfinite(r.mean_cs_ric) else float("nan"),
            r.cs_ric_stability if np.isfinite(r.cs_ric_stability) else float("nan"),
            r.ls_sharpe if np.isfinite(r.ls_sharpe) else float("nan"),
            r.turnover,
        )

    return results
