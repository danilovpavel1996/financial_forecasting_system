"""COT (Commitment of Traders) feature engineering.

Leakage guarantee
-----------------
COT reports are released on Friday at 3:30pm ET for Tuesday positions.
Raw data is indexed by release_date (Friday).  Feature pipeline:
  1. Compute derived features on the WEEKLY series (indexed by release_date).
  2. Align to daily trading calendar via align_to_trading_index()
     using forward-fill with max 10 trading days (≈ 2 missed weekly reports).
  3. Lag by 1 trading day (lag_series(lag=1)).

After step 3, the value at date t is the value of the last report whose
release_date is strictly < t.  Monday receives the most-recent Friday's release.
Friday receives the previous Friday's release (today's report is after market close).

The 52-week percentile (mm_net_pct) is computed on the WEEKLY series BEFORE
forward-filling to daily.  This respects the weekly data cadence and avoids
inflating the window with repeated daily values.

Shift-and-compare leakage tests: tests/test_cot_features.py.
"""
from __future__ import annotations

import logging
from typing import Dict, Sequence

import numpy as np
import pandas as pd

from src.features.macro_features import align_to_trading_index, lag_series

logger = logging.getLogger(__name__)

# Maximum consecutive trading days to forward-fill weekly COT data.
# 2 missed weekly reports ≈ 14 calendar days ≈ 10 trading days.
_MAX_FFILL_COT = 10

# 52-week rolling window for mm_net_pct; require at least half as warmup.
_PCT_WINDOW   = 52
_PCT_MIN_PERIODS = 26

# Small epsilon to avoid division by zero in percentile computation.
_EPS = 1e-10


def build_cot_features(
    cot_raw: Dict[str, pd.DataFrame],
    basket_tickers: Sequence[str],
    trading_index: pd.DatetimeIndex,
    lag: int = 1,
) -> Dict[str, pd.DataFrame]:
    """Build per-asset COT feature matrices aligned to a daily trading calendar.

    For each ticker in basket_tickers, computes 6 features:
      mm_net, mm_net_chg, mm_net_pct, mm_long_ratio, prod_net, oi_chg_pct

    Parameters
    ----------
    cot_raw:        dict from cot.fetch_all_cot() — {ticker → weekly DataFrame}.
    basket_tickers: ordered list of tickers in the basket.
    trading_index:  common daily DatetimeIndex for alignment.
    lag:            trading-day shift after release_date alignment (default 1).

    Returns
    -------
    dict mapping ticker → pd.DataFrame (index=trading_index, 6 float columns).
    Tickers with no COT data return a DataFrame of NaNs (not an error).
    """
    result: Dict[str, pd.DataFrame] = {}
    for ticker in basket_tickers:
        if ticker not in cot_raw:
            logger.warning("COT features: no raw data for %s — returning NaN block.", ticker)
            result[ticker] = _nan_block(trading_index)
            continue
        result[ticker] = _build_one(cot_raw[ticker], ticker, trading_index, lag)

    _log_feature_summary(result, basket_tickers)
    return result


def build_cross_cot_features(
    cot_daily: Dict[str, pd.DataFrame],
    basket_tickers: Sequence[str],
    trading_index: pd.DatetimeIndex,
) -> Dict[str, pd.DataFrame]:
    """Build cross-sectional (relative) COT features for each asset.

    For each ticker, computes:
      rel_mm_net_pct = ticker mm_net_pct  − basket average mm_net_pct
      rel_mm_net_chg = ticker mm_net_chg  − basket average mm_net_chg

    Parameters
    ----------
    cot_daily:      dict from build_cot_features() — {ticker → daily DataFrame}.
    basket_tickers: ordered list of N tickers.
    trading_index:  shared DatetimeIndex.

    Returns
    -------
    dict mapping ticker → pd.DataFrame with 2 cross-sectional COT columns.
    """
    basket_tickers = list(basket_tickers)

    # Basket averages (NaN-safe mean across assets at each date)
    pct_cols = pd.concat(
        [cot_daily[t]["mm_net_pct"].rename(t) for t in basket_tickers
         if t in cot_daily],
        axis=1,
    )
    chg_cols = pd.concat(
        [cot_daily[t]["mm_net_chg"].rename(t) for t in basket_tickers
         if t in cot_daily],
        axis=1,
    )

    basket_pct_mean = pct_cols.mean(axis=1)   # NaN-safe
    basket_chg_mean = chg_cols.mean(axis=1)

    result: Dict[str, pd.DataFrame] = {}
    for ticker in basket_tickers:
        if ticker not in cot_daily:
            result[ticker] = pd.DataFrame(
                {"rel_mm_net_pct": np.nan, "rel_mm_net_chg": np.nan},
                index=trading_index,
            )
            continue
        df = pd.DataFrame(
            {
                "rel_mm_net_pct": cot_daily[ticker]["mm_net_pct"] - basket_pct_mean,
                "rel_mm_net_chg": cot_daily[ticker]["mm_net_chg"] - basket_chg_mean,
            },
            index=trading_index,
        )
        result[ticker] = df

    return result


# ── Per-ticker feature builder ───────────────────────────────────────────────

def _build_one(
    raw: pd.DataFrame,
    ticker: str,
    trading_index: pd.DatetimeIndex,
    lag: int,
) -> pd.DataFrame:
    """Compute 6 COT features for one ticker; align + lag to trading_index."""

    # ── Step 1: raw weekly series (indexed by release_date = Friday) ─────────
    mm_long  = raw["mm_long"].sort_index().astype(float)
    mm_short = raw["mm_short"].sort_index().astype(float)
    pm_long  = raw["prod_long"].sort_index().astype(float)
    pm_short = raw["prod_short"].sort_index().astype(float)
    oi       = raw["oi"].sort_index().astype(float)

    # ── Step 2: derive features on WEEKLY series ─────────────────────────────
    mm_net     = mm_long - mm_short
    mm_net_chg = mm_net.diff()                             # week-over-week change
    mm_net_pct = _rolling_percentile(mm_net)               # 52-week range percentile
    mm_long_sum = (mm_long + mm_short).replace(0, np.nan)  # guard zero denominator
    mm_long_ratio = mm_long / mm_long_sum
    prod_net   = pm_long - pm_short
    oi_safe    = oi.replace(0, np.nan)
    oi_chg_pct = oi.pct_change()                           # week-over-week % change in OI

    weekly_feats = {
        "mm_net":       mm_net,
        "mm_net_chg":   mm_net_chg,
        "mm_net_pct":   mm_net_pct,
        "mm_long_ratio":mm_long_ratio,
        "prod_net":     prod_net,
        "oi_chg_pct":  oi_chg_pct,
    }

    # ── Step 3: align each weekly series to daily, then lag ──────────────────
    daily_cols: dict[str, pd.Series] = {}
    for name, s in weekly_feats.items():
        s.name = f"{ticker}_cot_{name}"
        aligned = align_to_trading_index(s, trading_index, max_ffill_days=_MAX_FFILL_COT)
        lagged  = lag_series(aligned, lag)
        lagged.name = name
        daily_cols[name] = lagged

    result = pd.DataFrame(daily_cols, index=trading_index)

    n_nan = result.isnull().any(axis=1).sum()
    logger.debug(
        "COT features %s: %d rows, %d NaN rows (warm-up + gaps expected)",
        ticker, len(result), n_nan,
    )
    return result


def _rolling_percentile(s: pd.Series) -> pd.Series:
    """52-week rolling min-max percentile of mm_net (computed on weekly series).

    Values in [0, 1] after the warm-up period (min_periods=26 half-year).
    Returns NaN during warm-up and when min==max (flat positioning).
    """
    roll_min = s.rolling(_PCT_WINDOW, min_periods=_PCT_MIN_PERIODS).min()
    roll_max = s.rolling(_PCT_WINDOW, min_periods=_PCT_MIN_PERIODS).max()
    denom = (roll_max - roll_min).replace(0, np.nan)   # NaN when range is zero
    pct = (s - roll_min) / denom
    return pct.clip(0.0, 1.0)


def _nan_block(trading_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Return a DataFrame of NaNs with the 6 COT feature columns."""
    cols = ["mm_net", "mm_net_chg", "mm_net_pct", "mm_long_ratio", "prod_net", "oi_chg_pct"]
    return pd.DataFrame(np.nan, index=trading_index, columns=cols)


def _log_feature_summary(
    result: Dict[str, pd.DataFrame],
    basket_tickers: Sequence[str],
) -> None:
    for ticker in basket_tickers:
        if ticker not in result:
            continue
        df = result[ticker]
        n_nan = df.isnull().any(axis=1).sum()
        logger.info(
            "COT features %s: %d rows, %d cols, %d NaN rows",
            ticker, len(df), df.shape[1], n_nan,
        )
        pct_col = df.get("mm_net_pct")
        if pct_col is not None:
            valid = pct_col.dropna()
            if len(valid):
                out_of_bounds = ((valid < 0) | (valid > 1)).sum()
                if out_of_bounds:
                    logger.warning(
                        "COT features %s: mm_net_pct has %d values outside [0,1]",
                        ticker, out_of_bounds,
                    )
