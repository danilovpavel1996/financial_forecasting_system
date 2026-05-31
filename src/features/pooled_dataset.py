"""Pooled (date × asset) dataset builder for cross-sectional ranking.

Stacks per-asset feature matrices into one tall DataFrame.  Each row is a
(date, asset) pair.  The walk-forward splitter operates on DATES ONLY — all
4 metals at the same date must be in the same fold, never split across
train/test.

Column layout
-------------
  own price features   (10 cols: ret_1d, ret_5d, ..., dist_ma_60)
  cross-asset features (8 cols: rel_mom_*, rel_vol_*, ratios)
  context ETF/macro    (~16 cols: same for every asset on the same date)
  asset one-hot        (3 cols: is_gc, is_si, is_pl;  PA=F = reference)
  target               (1 col:  forward log return at horizon)

The 'asset' column is NOT included in the feature matrix — it is encoded
via the one-hot columns so sklearn models can ingest raw numpy arrays.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from src.config import Config
from src.data import universe
from src.features.cross_features import build_cross_features
from src.features.macro_features import build_macro_features
from src.features.price_features import build_price_features
from src.targets import forward_log_return

logger = logging.getLogger(__name__)

# Asset-level one-hot column names (PA=F is the reference, gets all zeros)
_ONEHOT_MAP = {
    "GC=F": "is_gc",
    "SI=F": "is_si",
    "PL=F": "is_pl",
    # PA=F → no indicator column (reference category)
}


def build_pooled_dataset(
    cfg: Config,
    prices: dict[str, pd.DataFrame],
    macro_raw: dict[str, pd.Series],
    horizon: int,
) -> pd.DataFrame:
    """Build the pooled (date × asset) feature + target DataFrame.

    Parameters
    ----------
    cfg:       project config (for late-close tickers, random seed, paths).
    prices:    dict mapping ticker → OHLCV DataFrame (all tickers loaded).
    macro_raw: dict mapping FRED series ID → daily Series.
    horizon:   forecast horizon in trading days.

    Returns
    -------
    pd.DataFrame with:
      - MultiIndex: (date [DatetimeIndex], asset [str])
      - feature columns (float) + 'target' column (float)

    Only dates where ALL 4 metals have valid features AND a valid target are
    included; the final set is the intersection across all 4 assets.
    """
    metals = universe.metal_tickers(cfg)
    if len(metals) != 4:
        raise ValueError(f"Expected 4 metals, got {len(metals)}: {metals}")

    late_close = set(universe.equity_context_tickers(cfg))
    context_tickers = [t for t in universe.price_tickers(cfg) if t not in metals]

    # ── Use GC=F's calendar as the shared reference ──────────────────────────
    ref_index: pd.DatetimeIndex = prices["GC=F"].index

    # ── Common context features (same for all assets on the same date) ───────
    # Build the ETF/macro context exactly once with GC=F as target.
    all_context_prices = {t: prices[t] for t in context_tickers if t in prices}
    # Build a GC=F price dict that includes context tickers for build_price_features
    ctx_prices = dict(prices)  # shallow copy; includes all tickers

    macro_feats = build_macro_features(macro_raw, trading_index=ref_index)

    # ── Cross-asset features (all 4 metals together) ─────────────────────────
    cross_feats = build_cross_features(prices, metals, ref_index)

    # ── Per-asset own features + target ──────────────────────────────────────
    asset_dfs: list[pd.DataFrame] = []

    for ticker in metals:
        # Own price features for this metal
        own_feats = build_price_features(
            ctx_prices,
            target_ticker=ticker,
            context_tickers=context_tickers,
            late_close_tickers=list(late_close),
        )
        # own_feats is indexed by GC=F's calendar when context tickers include GC=F;
        # metals futures share the same CME calendar so the index matches.

        # Target: forward log return
        close = prices[ticker]["Close"]
        target = forward_log_return(close, horizon=horizon)
        target.name = "target"

        # Cross-asset features for this metal
        cross = cross_feats[ticker]

        # Asset one-hot encoding
        onehot = pd.DataFrame(
            {col: 0 for col in _ONEHOT_MAP.values()},
            index=ref_index,
            dtype=float,
        )
        if ticker in _ONEHOT_MAP:
            onehot[_ONEHOT_MAP[ticker]] = 1.0

        # Assemble all feature columns + macro + target
        df = (
            own_feats
            .join(cross, how="left")
            .join(macro_feats, how="left")
            .join(onehot, how="left")
            .join(target.reindex(ref_index), how="left")
        )

        # Tag with asset label
        df.index.name = "date"
        df["asset"] = ticker

        asset_dfs.append(df)

    # ── Intersect valid dates across all 4 assets ─────────────────────────────
    # For each asset: rows where ALL features and target are finite.
    valid_dates_per_asset: list[pd.Index] = []
    for df in asset_dfs:
        feature_cols = [c for c in df.columns if c != "asset"]
        ok = df[feature_cols].notna().all(axis=1)
        valid_dates_per_asset.append(df.index[ok])

    common_dates = valid_dates_per_asset[0]
    for vd in valid_dates_per_asset[1:]:
        common_dates = common_dates.intersection(vd)

    logger.info(
        "Pooled dataset: %d common dates across all %d metals (horizon=%d)",
        len(common_dates), len(metals), horizon,
    )

    # ── Stack into MultiIndex DataFrame ──────────────────────────────────────
    frames: list[pd.DataFrame] = []
    for df in asset_dfs:
        sub = df.loc[common_dates].copy()
        sub.index = pd.MultiIndex.from_arrays(
            [sub.index, sub["asset"]],
            names=["date", "asset"],
        )
        sub = sub.drop(columns=["asset"])
        frames.append(sub)

    pooled = pd.concat(frames, axis=0).sort_index()

    # Verify — should have exactly 4 rows per date
    n_per_date = pooled.groupby(level="date").size()
    bad_dates = n_per_date[n_per_date != len(metals)]
    if len(bad_dates):
        raise RuntimeError(
            f"Pooled dataset has dates with != {len(metals)} assets: "
            f"{bad_dates.head().to_dict()}"
        )

    logger.info(
        "Pooled dataset shape: %d rows × %d cols  (%d dates × %d assets)",
        len(pooled), pooled.shape[1], len(common_dates), len(metals),
    )
    return pooled


def feature_cols(pooled: pd.DataFrame) -> list[str]:
    """Return the feature column names (everything except 'target')."""
    return [c for c in pooled.columns if c != "target"]


def get_asset_order(pooled: pd.DataFrame) -> list[str]:
    """Return the sorted unique asset labels from the pooled dataset."""
    return sorted(pooled.index.get_level_values("asset").unique().tolist())
