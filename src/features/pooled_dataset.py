"""Pooled (date × asset) dataset builder for cross-sectional ranking.

Stacks per-asset feature matrices into one tall DataFrame.  Each row is a
(date, asset) pair.  The walk-forward splitter operates on DATES ONLY — all
N assets at the same date must be in the same fold, never split across
train/test.

Column layout
-------------
  own price features   (~10 cols: ret_1d, ret_5d, ..., dist_ma_60)
  cross-asset features (~8 cols: rel_mom_*, rel_vol_*, ratios)
  context ETF/macro    (~16 cols: same for every asset on the same date)
  asset one-hot        (N-1 cols: is_<ticker>; last sorted ticker = reference)
  target               (1 col:  forward log return at horizon)

The 'asset' column is NOT included in the feature matrix — it is encoded
via the one-hot columns so sklearn models can ingest raw numpy arrays.
"""
from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd

from src.config import Config
from src.data import universe
from src.features.carry_features import build_carry_features
from src.features.cot_features import build_cot_features, build_cross_cot_features
from src.features.cross_features import build_cross_features
from src.features.macro_features import build_macro_features
from src.features.price_features import build_price_features
from src.features.seasonal_features import build_seasonal_features
from src.targets import forward_log_return

logger = logging.getLogger(__name__)


def _build_onehot_map(tickers: list[str]) -> dict[str, str]:
    """Build {ticker: column_name} for N-1 one-hot columns.

    Tickers are sorted; the last in sorted order is the reference category
    (all-zero row).  Column names: 'is_' + ticker with '=^-' removed, lowercased.
    """
    tickers_sorted = sorted(tickers)
    return {
        t: "is_" + t.replace("=", "").replace("^", "").replace("-", "").lower()
        for t in tickers_sorted[:-1]
    }


def build_pooled_dataset(
    cfg: Config,
    prices: dict[str, pd.DataFrame],
    macro_raw: dict[str, pd.Series],
    horizon: int,
    cot_raw: dict[str, pd.DataFrame] | None = None,
    ranked_override: list[str] | None = None,
    context_override: list[str] | None = None,
    ref_ticker_override: str | None = None,
    late_close_override: set[str] | None = None,
    carry_pairs_override: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Build the pooled (date × asset) feature + target DataFrame.

    Parameters
    ----------
    cfg:                project config (for late-close tickers, random seed, paths).
    prices:             dict mapping ticker → OHLCV DataFrame (all tickers loaded).
    macro_raw:          dict mapping FRED series ID → daily Series.
    horizon:            forecast horizon in trading days.
    cot_raw:            optional dict from cot.fetch_all_cot() — {ticker → weekly DataFrame}.
                        If None or empty, COT features are omitted (backward compatible).
    ranked_override:    override the ranked tickers list (default: from cfg).
    context_override:   override the context tickers list (default: from cfg).
    ref_ticker_override: override the reference ticker for the trading calendar.
    late_close_override: override the set of late-close tickers (pass set() for all-ETF universes).
    carry_pairs_override: override carry pairs ({} disables carry_proxy, keeps basis_momentum).

    Returns
    -------
    pd.DataFrame with:
      - MultiIndex: (date [DatetimeIndex], asset [str])
      - feature columns (float) + 'target' column (float)

    Only dates where ALL N ranked assets have valid features AND a valid target
    are included; the final set is the intersection across all assets.
    """
    metals = ranked_override if ranked_override is not None else universe.ranked_tickers(cfg)
    if not metals:
        raise ValueError(
            "No ranked_assets configured in config.yaml. "
            "Add a universe.ranked_assets section for Phase 8+."
        )

    n_assets = len(metals)
    metals_set = set(metals)

    if late_close_override is not None:
        late_close = late_close_override
    else:
        late_close = set(universe.equity_context_tickers(cfg))

    if context_override is not None:
        context_tickers = [t for t in context_override if t not in metals_set]
    else:
        context_tickers = [t for t in universe.price_tickers(cfg) if t not in metals_set]

    # One-hot: N-1 indicator columns, last sorted ticker is reference category
    onehot_map = _build_onehot_map(metals)
    onehot_cols = list(onehot_map.values())

    # Reference calendar: use override ticker, or fall back to GC=F
    _ref_ticker = ref_ticker_override if ref_ticker_override is not None else "GC=F"
    if _ref_ticker not in prices:
        _ref_ticker = metals[0]
    ref_index: pd.DatetimeIndex = prices[_ref_ticker].index

    # Log per-asset row counts to flag coverage issues
    for ticker in metals:
        if ticker not in prices:
            logger.warning("Ranked ticker %s not in prices dict — it will be missing.", ticker)
            continue
        n_rows = len(prices[ticker])
        if n_rows < 2000:
            logger.warning(
                "Ticker %s has only %d rows (< 2000) — data coverage may be insufficient.",
                ticker, n_rows,
            )
        else:
            logger.info("Ticker %s: %d rows", ticker, n_rows)

    # ── Common context features (same for all assets on the same date) ───────
    macro_feats = build_macro_features(macro_raw, trading_index=ref_index)

    # ── Seasonal features (same for all assets on the same date) ─────────────
    seasonal_feats = build_seasonal_features(ref_index)

    # ── Carry / term-structure features (per-asset) ───────────────────────────
    carry_feats = build_carry_features(
        prices, metals, ref_index,
        carry_pairs=carry_pairs_override,  # None = use defaults; {} = basis_momentum only
    )

    # ── Cross-asset features (all N assets together) ─────────────────────────
    cross_feats = build_cross_features(prices, metals, ref_index)

    # ── COT features (per-asset positioning data) ─────────────────────────────
    use_cot = bool(cot_raw)
    if use_cot:
        cot_feats = build_cot_features(cot_raw, metals, ref_index)
        cross_cot_feats = build_cross_cot_features(cot_feats, metals, ref_index)
        logger.info("COT features enabled: 6 per-asset + 2 cross-sectional columns added.")
    else:
        logger.info("COT features disabled (cot_raw is None or empty).")

    # ── Per-asset own features + target ──────────────────────────────────────
    asset_dfs: list[pd.DataFrame] = []

    for ticker in metals:
        # Own price features for this asset
        own_feats = build_price_features(
            prices,
            target_ticker=ticker,
            context_tickers=context_tickers,
            late_close_tickers=list(late_close),
        )

        # Target: forward log return
        # Sanitize non-positive closes (e.g. CL=F on 2020-04-20) → NaN target
        # so those dates are naturally dropped by the valid-date intersection.
        close = prices[ticker]["Close"].copy()
        n_nonpos = int((close <= 0).sum())
        if n_nonpos:
            logger.warning(
                "Ticker %s: %d non-positive close price(s) → NaN target; "
                "date(s) will be excluded from pooled dataset.",
                ticker, n_nonpos,
            )
            close = close.where(close > 0)
        target = forward_log_return(close, horizon=horizon)
        target.name = "target"

        # Cross-asset features for this asset
        cross = cross_feats[ticker]

        # Asset one-hot encoding: N-1 columns, reference = last sorted ticker
        onehot = pd.DataFrame(
            {col: 0.0 for col in onehot_cols},
            index=ref_index,
            dtype=float,
        )
        if ticker in onehot_map:
            onehot[onehot_map[ticker]] = 1.0

        # Assemble all feature columns + macro + seasonal + carry + (COT if enabled) + target
        df = (
            own_feats
            .join(cross, how="left")
            .join(macro_feats, how="left")
            .join(seasonal_feats, how="left")
            .join(carry_feats[ticker].reindex(ref_index), how="left")
        )
        if use_cot:
            df = (
                df
                .join(cot_feats[ticker].reindex(ref_index), how="left")
                .join(cross_cot_feats[ticker].reindex(ref_index), how="left")
            )
        df = df.join(onehot, how="left").join(target.reindex(ref_index), how="left")

        # Tag with asset label
        df.index.name = "date"
        df["asset"] = ticker

        asset_dfs.append(df)

    # ── Intersect valid dates across all N assets ─────────────────────────────
    valid_dates_per_asset: list[pd.Index] = []
    for df in asset_dfs:
        feature_cols_here = [c for c in df.columns if c != "asset"]
        ok = df[feature_cols_here].notna().all(axis=1)
        valid_dates_per_asset.append(df.index[ok])
        n_valid = ok.sum()
        ticker = df["asset"].iloc[0]
        logger.info("Ticker %s: %d valid dates (of %d total)", ticker, n_valid, len(df))

    common_dates = valid_dates_per_asset[0]
    for vd in valid_dates_per_asset[1:]:
        common_dates = common_dates.intersection(vd)

    n_dropped = len(valid_dates_per_asset[0]) - len(common_dates)
    logger.info(
        "Pooled dataset: %d common dates across all %d assets (horizon=%d); "
        "%d dates dropped by intersection (calendar/coverage gaps).",
        len(common_dates), n_assets, horizon, n_dropped,
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

    # Verify — should have exactly n_assets rows per date
    n_per_date = pooled.groupby(level="date").size()
    bad_dates = n_per_date[n_per_date != n_assets]
    if len(bad_dates):
        raise RuntimeError(
            f"Pooled dataset has dates with != {n_assets} assets: "
            f"{bad_dates.head().to_dict()}"
        )

    logger.info(
        "Pooled dataset shape: %d rows × %d cols  (%d dates × %d assets)",
        len(pooled), pooled.shape[1], len(common_dates), n_assets,
    )
    return pooled


def feature_cols(pooled: pd.DataFrame) -> list[str]:
    """Return the feature column names (everything except 'target')."""
    return [c for c in pooled.columns if c != "target"]


def get_asset_order(pooled: pd.DataFrame) -> list[str]:
    """Return the sorted unique asset labels from the pooled dataset."""
    return sorted(pooled.index.get_level_values("asset").unique().tolist())
