"""Cross-sectional features for the ranked commodity basket.

All features at time t use ONLY data available at or before t — the same
invariant as price_features.py.  The basket-average subtraction is also
strictly lagged: for date t, the basket mean uses each asset's own trailing
computation at t, not any future value.

Works for any basket size N ≥ 2.  Relative features sum to zero across the
basket at every date by construction.

Shift-and-compare leakage tests live in tests/test_cross_features.py.
"""
from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd

from src.features.price_features import (
    distance_from_ma,
    rolling_momentum,
    rolling_volatility,
    log_ratio,
)

logger = logging.getLogger(__name__)

# Default windows — kept consistent with price_features.py
_MOM_WINDOWS: tuple[int, ...] = (10, 21)
_VOL_WINDOWS: tuple[int, ...] = (21,)
_MA_WINDOWS: tuple[int, ...] = (20, 60)

# Metals for which ratio features are defined
_RATIO_PAIRS = [
    ("GC=F", "SI=F", "gold_silver_ratio"),
    ("GC=F", "PL=F", "gold_platinum_ratio"),
    ("SI=F", "PL=F", "silver_platinum_ratio"),
]


def build_cross_features(
    prices: dict[str, pd.DataFrame],
    basket_tickers: Sequence[str],
    ref_index: pd.DatetimeIndex,
) -> dict[str, pd.DataFrame]:
    """Build cross-sectional relative features for each asset in the basket.

    Works for any basket size N ≥ 2.

    For each asset ``a`` at date ``t`` the returned features are:

    - ``rel_mom_10d``, ``rel_mom_21d``
          asset's trailing 10d/21d momentum minus basket-average momentum
    - ``rel_vol_21d``
          asset's annualised 21d vol minus basket-average vol
    - ``rel_dist_ma_20``, ``rel_dist_ma_60``
          asset's distance-from-MA minus basket-average distance
    - ``gold_silver_ratio``, ``gold_platinum_ratio``, ``silver_platinum_ratio``
          log price-ratios for precious metals pairs (skipped if tickers absent)

    Parameters
    ----------
    prices:         dict mapping ticker → OHLCV DataFrame (from cache).
    basket_tickers: ordered list of N assets in the basket.
    ref_index:      common trading-day DatetimeIndex to align all series.

    Returns
    -------
    dict mapping ticker → pd.DataFrame of cross-asset features,
    all sharing ``ref_index``.
    """
    basket_tickers = list(basket_tickers)
    # keep internal variable name for minimal diff
    metal_tickers = basket_tickers

    # ── Step 1: per-asset own features on the common calendar ──────────────
    own: dict[str, dict[str, pd.Series]] = {}
    for ticker in metal_tickers:
        if ticker not in prices:
            raise KeyError(f"Ticker {ticker!r} missing from prices dict.")
        close = prices[ticker]["Close"].reindex(ref_index).ffill()

        mom = rolling_momentum(close, windows=list(_MOM_WINDOWS))
        vol = rolling_volatility(close, windows=list(_VOL_WINDOWS))
        dma = distance_from_ma(close, windows=list(_MA_WINDOWS))

        own[ticker] = {
            "mom_10d":    mom["mom_10d"],
            "mom_21d":    mom["mom_21d"],
            "vol_21d":    vol["vol_21d"],
            "dist_ma_20": dma["dist_ma_20"],
            "dist_ma_60": dma["dist_ma_60"],
        }

    # ── Step 2: basket means (no future data — each col computed at t) ─────
    feat_names = ("mom_10d", "mom_21d", "vol_21d", "dist_ma_20", "dist_ma_60")
    basket_mean: dict[str, pd.Series] = {}
    for feat in feat_names:
        cols = pd.concat(
            [own[t][feat].rename(t) for t in metal_tickers], axis=1
        )
        basket_mean[feat] = cols.mean(axis=1)  # NaN-safe: ignores NaN columns

    # ── Step 3: ratio features (same for all assets on a given date) ────────
    ratio_df = _build_ratios(prices, ref_index)

    # ── Step 4: assemble per-asset cross-feature DataFrame ──────────────────
    result: dict[str, pd.DataFrame] = {}
    for ticker in metal_tickers:
        parts: dict[str, pd.Series] = {}
        for feat in feat_names:
            parts[f"rel_{feat}"] = own[ticker][feat] - basket_mean[feat]
        df = pd.DataFrame(parts, index=ref_index)
        df = df.join(ratio_df)
        result[ticker] = df

    logger.info(
        "Cross-sectional features built: %d assets × %d dates × %d columns",
        len(metal_tickers), len(ref_index), result[metal_tickers[0]].shape[1],
    )
    return result


def _build_ratios(
    prices: dict[str, pd.DataFrame],
    ref_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Build log price-ratio features. Returns DataFrame indexed by ref_index."""
    parts: list[pd.Series] = []
    for ta, tb, name in _RATIO_PAIRS:
        if ta not in prices or tb not in prices:
            logger.warning("Ratio pair (%s/%s) not available; skipping.", ta, tb)
            continue
        parts.append(
            log_ratio(
                prices[ta]["Close"],
                prices[tb]["Close"],
                name=name,
                ref_index=ref_index,
            )
        )
    if not parts:
        return pd.DataFrame(index=ref_index)
    return pd.concat(parts, axis=1)
