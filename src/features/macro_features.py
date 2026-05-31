"""Macro feature engineering — raw values only, NO normalization.

All features are lagged by at least 1 trading day before use.

Why the lag is mandatory
------------------------
FRED series (real yields, breakevens, DXY, VIX) are published with a
lag relative to their observation date. Using today's FRED value as a
feature for today's return model would assume we know that value in real
time, which we do not. Lagging by 1 trading day makes the feature
conservative and avoids look-ahead bias.

Additionally, forward-filling calendar-day FRED data onto a trading-day
index must use only past observations. The implementation here:
  1. Reindexes the raw FRED series to the trading-day index using ffill()
     (fills each trading day with the last known FRED value on or before
      that day — this is safe, uses no future data).
  2. Shifts by `lag` trading days (default 1) — at day t, the feature
     value is the aligned value from day t-lag.
  3. Daily changes are computed on the lagged series, so the change at t
     is (value[t-lag] - value[t-lag-1]).

TODO (point-in-time): use ALFRED vintage data for rigorous publication-lag
handling. The 1-day lag is a reasonable approximation but not fully
point-in-time correct for all FRED series.

Normalization (z-scoring) is intentionally absent — that belongs inside
the model pipeline fitted on train data only.
"""
from __future__ import annotations

import logging
from typing import Sequence

import pandas as pd

logger = logging.getLogger(__name__)


# Series IDs expected in the macro dict
REAL_YIELD = "DFII10"
BREAKEVEN = "T10YIE"
NOM_YIELD = "DGS10"
DXY = "DTWEXBGS"
VIX = "VIXCLS"


# --------------------------------------------------------------------------- #
# Core building blocks
# --------------------------------------------------------------------------- #

def align_to_trading_index(
    raw: pd.Series,
    trading_index: pd.DatetimeIndex,
    max_ffill_days: int = 5,
) -> pd.Series:
    """Reindex a calendar-day FRED series to a trading-day index.

    Uses forward-fill to propagate the last known value across weekends and
    holidays. Gaps larger than `max_ffill_days` are left as NaN and logged.

    Parameters
    ----------
    raw:             FRED series with a calendar-day DatetimeIndex.
    trading_index:   target trading-day DatetimeIndex.
    max_ffill_days:  maximum consecutive days to forward-fill.
                     Set to 5 (Mon-Fri span) by default; longer gaps likely
                     indicate a genuine data hole and should be NaN.
    """
    # Align to the full span first (union of calendar and trading days)
    # then ffill with a limit, then select only trading days.
    full_span = pd.DatetimeIndex(
        sorted(set(raw.index).union(set(trading_index)))
    )
    aligned = raw.reindex(full_span).ffill(limit=max_ffill_days)
    result = aligned.reindex(trading_index)

    n_nan = result.isna().sum()
    if n_nan > 0:
        logger.warning(
            "%s: %d NaN after aligning to trading index (gaps > %d days or missing data)",
            raw.name, n_nan, max_ffill_days,
        )
    return result


def lag_series(series: pd.Series, lag: int = 1) -> pd.Series:
    """Shift a series by `lag` periods (positive = look back).

    At trading day t the returned value is series[t-lag].
    lag=1 enforces the FRED publication-lag convention.
    """
    if lag < 1:
        raise ValueError(f"lag must be >= 1 to prevent look-ahead; got {lag}")
    return series.shift(lag)


def daily_change(series: pd.Series) -> pd.Series:
    """First difference: series[t] - series[t-1].

    When applied to an already-lagged series, the result at t is
    (series[t-lag] - series[t-lag-1]) — uses no future data.
    """
    return series.diff()


# --------------------------------------------------------------------------- #
# High-level builder
# --------------------------------------------------------------------------- #

def build_macro_features(
    macro: dict[str, pd.Series],
    trading_index: pd.DatetimeIndex,
    lag: int = 1,
) -> pd.DataFrame:
    """Build the macro feature matrix aligned to a trading-day index.

    Procedure for each FRED series:
      1. Align raw calendar-day series to trading_index via forward-fill.
      2. Shift by `lag` trading days (publication-lag guard).
      3. Compute daily change of the lagged series.

    At trading day t:
      - level features: use FRED value from t-lag
      - change features: FRED value from t-lag minus value from t-lag-1

    Parameters
    ----------
    macro:          dict mapping FRED series ID → raw pd.Series (from macro.py cache).
    trading_index:  trading-day DatetimeIndex to align to.
    lag:            publication-lag shift in trading days (default 1, minimum 1).
    """
    cols: dict[str, pd.Series] = {}

    def _get(sid: str) -> pd.Series | None:
        if sid not in macro:
            logger.warning("Macro series %r not found; skipping its features.", sid)
            return None
        aligned = align_to_trading_index(macro[sid], trading_index)
        return lag_series(aligned, lag)

    # --- Real yield (DFII10) ---
    dfii10 = _get(REAL_YIELD)
    if dfii10 is not None:
        cols["dfii10_level"] = dfii10
        cols["dfii10_chg_1d"] = daily_change(dfii10)

    # --- Breakeven inflation (T10YIE) ---
    t10yie = _get(BREAKEVEN)
    if t10yie is not None:
        cols["t10yie_level"] = t10yie
        cols["t10yie_chg_1d"] = daily_change(t10yie)

    # --- Nominal 10Y yield (DGS10) ---
    dgs10 = _get(NOM_YIELD)
    if dgs10 is not None:
        cols["dgs10_chg_1d"] = daily_change(dgs10)

    # --- DXY / USD index (DTWEXBGS) ---
    dxy = _get(DXY)
    if dxy is not None:
        cols["dxy_level"] = dxy
        cols["dxy_chg_1d"] = daily_change(dxy)

    # --- VIX (VIXCLS) ---
    vix = _get(VIX)
    if vix is not None:
        cols["vix_level"] = vix
        cols["vix_chg_1d"] = daily_change(vix)

    result = pd.DataFrame(cols, index=trading_index)

    _log_feature_summary(result)
    return result


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #

def _log_feature_summary(df: pd.DataFrame) -> None:
    n_nan_rows = df.isnull().any(axis=1).sum()
    logger.info(
        "Macro features: %d rows, %d columns, %d rows with any NaN",
        len(df), df.shape[1], n_nan_rows,
    )
    for col in df.columns:
        n = df[col].isna().sum()
        if n > 0:
            logger.debug("  %s: %d NaN", col, n)
