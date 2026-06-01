"""Forex cross-sectional ranking pipeline.

Entry point: run_forex_pipeline(cfg, horizon) → dict[name → RankingResult]

Mirrors pipeline_ranking_sectors.py but operates on the forex universe
configured under forex in config.yaml.

No COT (not configured for forex).
No carry_proxy (no ETF pairs for forex).
No late-close lag — forex closes at 5pm ET for all pairs simultaneously,
so there is no asymmetric timing issue.
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
    MeanReversionRanker,
    RankingBacktester,
    RankingResult,
)
from src.eval.splitter import WalkForwardSplitter
from src.features.pooled_dataset import build_pooled_dataset, feature_cols
from src.models.gbm import LightGBMModel
from src.models.lambdamart import LambdaMARTModel

logger = logging.getLogger(__name__)


def run_forex_pipeline(
    cfg: Config,
    horizon: int,
    force_refresh: bool = False,
    vol_target: float | None = None,
    max_leverage: float = 2.0,
    vol_lookback: int = 21,
    embargo: int | None = None,
    model_names: list[str] | None = None,
    pred_avg_window: int = 1,
) -> Dict[str, RankingResult]:
    """Build forex pooled dataset and run walk-forward ranking backtest.

    Parameters
    ----------
    cfg:              project config.
    horizon:          forecast horizon in trading days (5, 21, or 63).
    force_refresh:    if True, re-fetch data from network.
    embargo:          override splitter embargo (defaults to horizon).
    model_names:      restrict which models to run (default: all).
    pred_avg_window:  rolling mean window applied to predictions before ranking.
                      1 = no averaging (default). 21 = B3 variant.

    Returns
    -------
    dict mapping model name → RankingResult.
    """
    forex_tkrs = universe.forex_tickers(cfg)
    if not forex_tkrs:
        raise ValueError("forex.ranked_assets is empty in config.yaml")

    context_tkrs = universe.forex_context_tickers(cfg)
    all_tickers = universe.forex_price_tickers(cfg)
    start_date = universe.forex_start_date(cfg)

    logger.info(
        "Forex universe: %d pairs, context: %s, start_date: %s",
        len(forex_tkrs), context_tkrs, start_date,
    )

    # ── Load raw data ────────────────────────────────────────────────────────
    prices = fetch_all_tickers(
        all_tickers,
        start_date,
        cfg.dates["end"],
        cfg.paths.data_raw,
        force_refresh=force_refresh,
    )

    for tkr in forex_tkrs:
        if tkr in prices:
            first = prices[tkr].index[0].date()
            n_rows = len(prices[tkr])
            logger.info("Forex pair %s: %d rows, first date %s", tkr, n_rows, first)
        else:
            logger.warning("Forex pair %s not in prices — will be missing!", tkr)

    api_key = os.environ.get("FRED_API_KEY", "").strip()
    macro_raw = fetch_all_series(
        universe.fred_series(cfg),
        start_date,
        cfg.dates["end"],
        cfg.paths.data_raw,
        api_key=api_key or None,
        force_refresh=False,
    )

    # ── Build pooled dataset ─────────────────────────────────────────────────
    # No COT (not configured for forex).
    # No carry_proxy (no ETF pairs).
    # No late-close lag (all forex pairs record 5pm ET close; same timestamp).
    pooled = build_pooled_dataset(
        cfg,
        prices,
        macro_raw,
        horizon=horizon,
        cot_raw=None,
        ranked_override=forex_tkrs,
        context_override=context_tkrs,
        ref_ticker_override=forex_tkrs[0],  # e.g., EURUSD=X as trading calendar
        late_close_override=set(),           # no lag: all pairs same timestamp
        carry_pairs_override={},             # basis_momentum only; no carry_proxy
    )

    fcols = feature_cols(pooled)
    logger.info(
        "Forex pooled dataset: %d rows × %d features, horizon=%d",
        len(pooled), len(fcols), horizon,
    )

    # ── Feature-column indices for ranking baselines ─────────────────────────
    ret_5d_idx = fcols.index("ret_5d") if "ret_5d" in fcols else 1

    # ── Splitter ─────────────────────────────────────────────────────────────
    _embargo = embargo if embargo is not None else horizon
    splitter = WalkForwardSplitter(
        min_train=int(cfg.splitter.train_years * 252),
        test_size=int(cfg.splitter.test_years * 252),
        embargo=_embargo,
        n_splits=cfg.splitter.n_splits,
        expanding=True,
    )

    n_assets = len(forex_tkrs)
    n_long = 2 if n_assets >= 6 else 1
    n_short = 2 if n_assets >= 6 else 1
    logger.info(
        "Forex ranking: %d pairs, n_long=%d, n_short=%d", n_assets, n_long, n_short
    )

    bt = RankingBacktester(
        splitter=splitter,
        cost_bps=cfg.cost_bps,
        horizon=horizon,
        assets=sorted(forex_tkrs),
        n_long=n_long,
        n_short=n_short,
        vol_target=vol_target,
        max_leverage=max_leverage,
        vol_lookback=vol_lookback,
        pred_avg_window=pred_avg_window,
    )

    rng_seed = cfg.random_seed
    all_factories = {
        "MeanReversion": lambda: MeanReversionRanker(feature_idx=ret_5d_idx),
        "LightGBM":      lambda: LightGBMModel(random_state=rng_seed),
        "LambdaMART":    lambda: LambdaMARTModel(random_state=rng_seed),
    }
    if model_names is not None:
        factories = {k: v for k, v in all_factories.items() if k in model_names}
    else:
        factories = all_factories

    results: Dict[str, RankingResult] = {}
    for name, factory in factories.items():
        logger.info("Running %s (horizon=%d) on forex …", name, horizon)
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
