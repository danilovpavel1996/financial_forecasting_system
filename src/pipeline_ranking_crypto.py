"""Crypto cross-sectional ranking pipeline.

Entry point: run_crypto_pipeline(cfg, horizon) → dict[name → RankingResult]

Mirrors pipeline_ranking_forex.py but operates on the crypto universe
configured under crypto in config.yaml.

Key differences from forex:
- Crypto trades 24/7; yfinance includes weekends.  We filter to business days
  (Mon-Fri) so weekend returns are absorbed into Monday's bar.  This keeps the
  pipeline compatible with the walk-forward splitter without major refactoring.
- cost_bps = 20 (Binance/Kraken ~0.1% per side = 20 bps round-trip).
- No COT, no carry_proxy, no late-close lag.
- Macro features forward-filled across holidays / weekends (same as forex).
"""
from __future__ import annotations

import logging
import os
from typing import Dict

import numpy as np
import pandas as pd

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


def _filter_to_business_days(
    prices: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Drop Saturday/Sunday rows from each price DataFrame.

    Crypto trades 24/7 so yfinance includes weekends.  Filtering to Mon-Fri
    (dayofweek 0-4) makes the calendar compatible with the walk-forward splitter
    and the rest of the pipeline.  Weekend returns are absorbed into Monday.
    """
    out: dict[str, pd.DataFrame] = {}
    for tkr, df in prices.items():
        bday_mask = df.index.dayofweek < 5
        out[tkr] = df.loc[bday_mask].copy()
    return out


def run_crypto_pipeline(
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
    """Build crypto pooled dataset and run walk-forward ranking backtest.

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
    crypto_tkrs = universe.crypto_tickers(cfg)
    if not crypto_tkrs:
        raise ValueError("crypto.ranked_assets is empty in config.yaml")

    context_tkrs = universe.crypto_context_tickers(cfg)
    all_tickers  = universe.crypto_price_tickers(cfg)
    start_date   = universe.crypto_start_date(cfg)
    cost_bps     = universe.crypto_cost_bps(cfg)

    logger.info(
        "Crypto universe: %d tokens, context: %s, start_date: %s, cost_bps=%.0f",
        len(crypto_tkrs), context_tkrs, start_date, cost_bps,
    )

    # ── Load raw data ────────────────────────────────────────────────────────
    prices_raw = fetch_all_tickers(
        all_tickers,
        start_date,
        cfg.dates["end"],
        cfg.paths.data_raw,
        force_refresh=force_refresh,
    )

    # Filter all data to business days (Mon-Fri)
    prices = _filter_to_business_days(prices_raw)

    for tkr in crypto_tkrs:
        if tkr in prices:
            first  = prices[tkr].index[0].date()
            n_rows = len(prices[tkr])
            logger.info("Crypto token %s: %d business-day rows, first date %s", tkr, n_rows, first)
        else:
            logger.warning("Crypto token %s not in prices — will be missing!", tkr)

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
    # No COT, no carry_proxy, no late-close lag.
    # ref_ticker_override: use the first available crypto ticker after bday filter.
    ref_ticker = next((t for t in crypto_tkrs if t in prices), crypto_tkrs[0])
    pooled = build_pooled_dataset(
        cfg,
        prices,
        macro_raw,
        horizon=horizon,
        cot_raw=None,
        ranked_override=crypto_tkrs,
        context_override=context_tkrs,
        ref_ticker_override=ref_ticker,
        late_close_override=set(),   # crypto: all tokens share the same daily bar
        carry_pairs_override={},     # no carry proxy for crypto
    )

    fcols = feature_cols(pooled)
    logger.info(
        "Crypto pooled dataset: %d rows × %d features, horizon=%d",
        len(pooled), len(fcols), horizon,
    )

    # ── Feature-column indices for ranking baselines ─────────────────────────
    ret_5d_idx = fcols.index("ret_5d") if "ret_5d" in fcols else 1

    # ── Splitter ─────────────────────────────────────────────────────────────
    # Crypto section may override the global splitter for shorter history.
    _embargo = embargo if embargo is not None else horizon
    crypto_spl = cfg.crypto.get("splitter", {})
    _train_years  = float(crypto_spl.get("train_years",  cfg.splitter.train_years))
    _test_years   = float(crypto_spl.get("test_years",   cfg.splitter.test_years))
    _n_splits_cfg = int(crypto_spl.get("n_splits",       cfg.splitter.n_splits))
    splitter = WalkForwardSplitter(
        min_train=int(_train_years * 252),
        test_size=int(_test_years  * 252),
        embargo=_embargo,
        n_splits=_n_splits_cfg,
        expanding=True,
    )

    n_assets = len(crypto_tkrs)
    n_long  = 3 if n_assets >= 9 else (2 if n_assets >= 6 else 1)
    n_short = 3 if n_assets >= 9 else (2 if n_assets >= 6 else 1)
    logger.info(
        "Crypto ranking: %d tokens, n_long=%d, n_short=%d, cost_bps=%.0f",
        n_assets, n_long, n_short, cost_bps,
    )

    bt = RankingBacktester(
        splitter=splitter,
        cost_bps=cost_bps,
        horizon=horizon,
        assets=sorted(crypto_tkrs),
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
        logger.info("Running %s (horizon=%d) on crypto …", name, horizon)
        r = bt.run(pooled, factory, model_name=name)
        results[name] = r
        logger.info(
            "  %s: CS_RIC=%.4f  stability=%.2f  Sharpe=%.2f  turnover=%.3f",
            name,
            r.mean_cs_ric   if np.isfinite(r.mean_cs_ric)        else float("nan"),
            r.cs_ric_stability if np.isfinite(r.cs_ric_stability) else float("nan"),
            r.ls_sharpe     if np.isfinite(r.ls_sharpe)          else float("nan"),
            r.turnover,
        )

    return results
