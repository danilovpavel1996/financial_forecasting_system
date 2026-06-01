"""Live feature builder — today's feature vector for all ranked assets.

CRITICAL DESIGN RULE: This module NEVER reimplements feature logic.
It calls the SAME functions from src/features/ used by the backtest.

The column structure here is the exact same structure produced by
build_pooled_dataset() in src/features/pooled_dataset.py, minus
the 'target' column (which is unknown for today's date).

This means: if the model was trained on pooled_dataset output, the
feature vector returned here is directly comparable.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.config import Config
from src.data import universe
from src.features.cot_features import build_cot_features, build_cross_cot_features
from src.features.cross_features import build_cross_features
from src.features.macro_features import build_macro_features
from src.features.price_features import build_price_features
from src.live.data import LiveData

logger = logging.getLogger(__name__)


def build_live_features(
    cfg: Config,
    live_data: LiveData,
    use_cot: bool = False,
    ranked_override: list[str] | None = None,
    context_override: list[str] | None = None,
    late_close_override: set[str] | None = None,
    ref_ticker_override: str | None = None,
) -> dict[str, pd.Series]:
    """Build today's feature vector for every ranked asset.

    Returns dict mapping ticker → pd.Series (index = feature names),
    representing the LATEST valid row of each asset's feature matrix.

    Column order and names match build_pooled_dataset() so a model
    trained on the pooled dataset can predict directly.

    Parameters
    ----------
    cfg:                project config.
    live_data:          output of fetch_live_data().
    use_cot:            include COT features (must match training config).
    ranked_override:    override the ranked tickers list.
    context_override:   override the context tickers list.
    late_close_override: override the late-close tickers set.
    ref_ticker_override: override the reference calendar ticker.
    """
    if ranked_override is not None:
        ranked = ranked_override
    else:
        ranked = universe.ranked_tickers(cfg)

    metals_set = set(ranked)

    if late_close_override is not None:
        late_close = late_close_override
    else:
        late_close = set(universe.equity_context_tickers(cfg))

    if context_override is not None:
        context_tickers = [t for t in context_override if t not in metals_set]
    else:
        context_tickers = [t for t in universe.price_tickers(cfg) if t not in metals_set]

    prices = live_data.prices

    # Reference calendar: use override ticker, fall back to GC=F, or first ranked asset
    _ref = ref_ticker_override or "GC=F"
    if _ref not in prices:
        _ref = ranked[0] if ranked else None
    if _ref is None or _ref not in prices:
        raise RuntimeError("Reference ticker not in prices — cannot build live features.")
    ref_index: pd.DatetimeIndex = prices[_ref].index

    # ── Common features (same for all assets on the same date) ───────────────
    macro_feats = build_macro_features(live_data.macro_raw, trading_index=ref_index)
    cross_feats = build_cross_features(prices, ranked, ref_index)

    cot_feats: Optional[dict] = None
    cross_cot_feats: Optional[dict] = None
    if use_cot and live_data.cot_raw:
        cot_feats = build_cot_features(live_data.cot_raw, ranked, ref_index)
        cross_cot_feats = build_cross_cot_features(cot_feats, ranked, ref_index)

    # ── One-hot encoding (deterministic from sorted ticker list) ─────────────
    onehot_map = _build_onehot_map(ranked)
    onehot_cols = list(onehot_map.values())

    # ── Per-asset: own price features + assembly ─────────────────────────────
    result: dict[str, pd.Series] = {}
    for ticker in ranked:
        own = build_price_features(
            prices,
            target_ticker=ticker,
            context_tickers=context_tickers,
            late_close_tickers=list(late_close),
        )

        # Assemble feature DataFrame (same join order as build_pooled_dataset)
        df = own.join(cross_feats[ticker], how="left").join(macro_feats, how="left")
        if use_cot and cot_feats and cross_cot_feats:
            df = (df
                  .join(cot_feats[ticker].reindex(ref_index), how="left")
                  .join(cross_cot_feats[ticker].reindex(ref_index), how="left"))

        # One-hot columns
        oh = pd.DataFrame(
            {col: 0.0 for col in onehot_cols},
            index=ref_index, dtype=float,
        )
        if ticker in onehot_map:
            oh[onehot_map[ticker]] = 1.0
        df = df.join(oh, how="left")

        # Take last row where all PRICE-BASED features are finite
        # (macro may have NaN near the end; we tolerate that)
        price_cols = list(own.columns)
        valid_mask = df[price_cols].notna().all(axis=1)
        valid = df.loc[valid_mask]

        if valid.empty:
            logger.warning("%s: no valid feature rows — skipping.", ticker)
            continue

        last_row = valid.iloc[-1]
        result[ticker] = last_row
        logger.info(
            "%s: live features as of %s (%d NaN of %d cols)",
            ticker,
            valid.index[-1].date(),
            int(last_row.isna().sum()),
            len(last_row),
        )

    return result


def get_feature_date(live_features: dict[str, pd.Series], ref_ticker: str | None = None) -> pd.Timestamp:
    """Return the index date of the live feature row (the 'as-of' date)."""
    if ref_ticker and ref_ticker in live_features:
        return live_features[ref_ticker].name
    if live_features:
        return next(iter(live_features.values())).name
    raise ValueError("No live features available.")


def align_to_training_cols(
    live_features: dict[str, pd.Series],
    training_cols: list[str],
) -> dict[str, np.ndarray]:
    """Reindex each asset's feature Series to exactly match training_cols order.

    Missing columns (e.g. COT columns not present in live data) → 0.0.
    Extra columns (not in training_cols) are dropped.
    Returns dict: ticker → 1-D float array of length len(training_cols).
    """
    result: dict[str, np.ndarray] = {}
    for ticker, series in live_features.items():
        aligned = series.reindex(training_cols, fill_value=np.nan)
        result[ticker] = aligned.values.astype(float)
    return result


# ── Private helper (mirrors pooled_dataset._build_onehot_map) ────────────────

def _build_onehot_map(tickers: list[str]) -> dict[str, str]:
    tickers_sorted = sorted(tickers)
    return {
        t: "is_" + t.replace("=", "").replace("^", "").replace("-", "").lower()
        for t in tickers_sorted[:-1]
    }
