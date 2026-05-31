"""Price-based feature engineering — raw values only, NO normalization.

All features at time t use ONLY data available at or before t:
  - Lagged log returns:        log(close[t] / close[t-k])
  - Rolling volatility:        std of past k daily returns (trailing window)
  - Rolling momentum:          mean of past k daily returns (trailing window)
  - Distance from MA:          (close[t] - MA_k[t]) / MA_k[t]  (trailing MA)
  - Gold/silver ratio:         log(close_gold[t] / close_silver[t])

None of these computations look forward. The shift-and-compare test in
tests/test_features.py enforces this mechanically.

Normalization (z-scoring) is intentionally absent. It belongs inside the
model pipeline (Phase 5) fitted on train data only. Doing it here would
leak test-period statistics into training features.
"""
from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Default windows used by the high-level builder
_RETURN_LAGS: tuple[int, ...] = (1, 5, 10, 21)
_VOL_WINDOWS: tuple[int, ...] = (5, 21)
_MOM_WINDOWS: tuple[int, ...] = (10, 21)
_MA_WINDOWS: tuple[int, ...] = (20, 60)
_ANNUALISE: int = 252  # trading days per year for vol annualisation


# --------------------------------------------------------------------------- #
# Core building blocks (operate on a single pd.Series of Close prices)
# --------------------------------------------------------------------------- #

def log_returns(close: pd.Series) -> pd.Series:
    """Daily log return: log(close[t] / close[t-1]).

    Uses close[t] (available at end of day t) and close[t-1]. No look-ahead.
    First row is NaN (no prior close available).
    """
    return np.log(close / close.shift(1))


def lagged_returns(
    close: pd.Series,
    lags: Sequence[int] = _RETURN_LAGS,
    prefix: str = "",
) -> pd.DataFrame:
    """Log returns over multiple horizons.

    ret_kd[t] = log(close[t] / close[t-k])  — uses only data <= t.

    Parameters
    ----------
    close:  daily Close price series.
    lags:   list of look-back horizons in trading days.
    prefix: column-name prefix, e.g. ticker symbol.
    """
    r1 = log_returns(close)
    frames: dict[str, pd.Series] = {}
    for k in lags:
        col = f"{prefix}ret_{k}d" if prefix else f"ret_{k}d"
        if k == 1:
            frames[col] = r1
        else:
            # sum of k consecutive 1-day log returns = log(close[t]/close[t-k])
            frames[col] = np.log(close / close.shift(k))
    return pd.DataFrame(frames, index=close.index)


def rolling_volatility(
    close: pd.Series,
    windows: Sequence[int] = _VOL_WINDOWS,
    prefix: str = "",
    annualise: bool = True,
) -> pd.DataFrame:
    """Trailing rolling volatility of daily log returns.

    vol_kd[t] = std(ret_1d[t-k+1 .. t]) * sqrt(252)  (annualised by default).

    Uses only data <= t (trailing window, no center=True). First k-1 rows are NaN.
    """
    r1 = log_returns(close)
    frames: dict[str, pd.Series] = {}
    for w in windows:
        col = f"{prefix}vol_{w}d" if prefix else f"vol_{w}d"
        raw_vol = r1.rolling(window=w, min_periods=w).std()
        frames[col] = raw_vol * np.sqrt(_ANNUALISE) if annualise else raw_vol
    return pd.DataFrame(frames, index=close.index)


def rolling_momentum(
    close: pd.Series,
    windows: Sequence[int] = _MOM_WINDOWS,
    prefix: str = "",
) -> pd.DataFrame:
    """Trailing rolling momentum (mean daily log return).

    mom_kd[t] = mean(ret_1d[t-k+1 .. t]).

    Uses only data <= t. First k-1 rows are NaN.
    """
    r1 = log_returns(close)
    frames: dict[str, pd.Series] = {}
    for w in windows:
        col = f"{prefix}mom_{w}d" if prefix else f"mom_{w}d"
        frames[col] = r1.rolling(window=w, min_periods=w).mean()
    return pd.DataFrame(frames, index=close.index)


def distance_from_ma(
    close: pd.Series,
    windows: Sequence[int] = _MA_WINDOWS,
    prefix: str = "",
) -> pd.DataFrame:
    """Distance of close from its trailing moving average, as a fraction.

    dist_ma_k[t] = (close[t] - MA_k[t]) / MA_k[t]
    where MA_k[t] = mean(close[t-k+1 .. t]).

    Uses only data <= t. First k-1 rows are NaN (MA undefined).
    """
    frames: dict[str, pd.Series] = {}
    for w in windows:
        col = f"{prefix}dist_ma_{w}" if prefix else f"dist_ma_{w}"
        ma = close.rolling(window=w, min_periods=w).mean()
        frames[col] = (close - ma) / ma
    return pd.DataFrame(frames, index=close.index)


def log_ratio(
    close_a: pd.Series,
    close_b: pd.Series,
    name: str,
    ref_index: pd.DatetimeIndex | None = None,
) -> pd.Series:
    """Log ratio of two price series: log(close_a / close_b).

    Both series are aligned to ref_index (or close_a's index if None) by
    forward-filling gaps. No look-ahead: forward-fill is only applied to
    fill weekend/holiday gaps for the cross series, not to introduce future data.

    Common use: gold/silver ratio = log(GC=F / SI=F).
    """
    if ref_index is None:
        ref_index = close_a.index

    a = close_a.reindex(ref_index).ffill()
    b = close_b.reindex(ref_index).ffill()
    return np.log(a / b).rename(name)


# --------------------------------------------------------------------------- #
# High-level builder
# --------------------------------------------------------------------------- #

def build_price_features(
    prices: dict[str, pd.DataFrame],
    target_ticker: str,
    context_tickers: Sequence[str] = (),
    late_close_tickers: Sequence[str] = (),
    ratio_pair: tuple[str, str] | None = None,
    ratio_name: str = "gold_silver_ratio",
) -> pd.DataFrame:
    """Build the full price-feature matrix for a target ticker.

    Parameters
    ----------
    prices:              dict mapping ticker → OHLCV DataFrame (from prices.py cache).
    target_ticker:       the primary asset whose features we're computing (e.g. "GC=F").
    context_tickers:     other tickers whose 1-day return is added as context features.
                         These are forward-filled to the target's trading-day index.
    late_close_tickers:  subset of context_tickers whose market closes AFTER the target.
                         For gold futures (GC=F, ~2:30pm ET close), this means all
                         equity-market ETFs and indices (4pm ET close).  Using today's
                         ETF return as a feature would embed gold price movement from
                         2:30pm→4pm that is not yet reflected in the futures close —
                         i.e. look-ahead relative to the futures settlement price.
                         For these tickers the 1-day return is shifted by one extra
                         trading day so the feature at t uses return from t-2 to t-1.
    ratio_pair:          (ticker_a, ticker_b) for a log-ratio feature.
    ratio_name:          column name for the ratio feature.

    Returns
    -------
    pd.DataFrame indexed by the target ticker's trading-day index.
    All feature columns are raw values — no z-scoring, no imputation.
    Initial NaN rows (rolling-window warm-up) are left as NaN.
    """
    if target_ticker not in prices:
        raise KeyError(f"Target ticker {target_ticker!r} not found in prices dict.")

    close_target = prices[target_ticker]["Close"].copy()
    ref_index = close_target.index

    parts: list[pd.DataFrame] = []

    # --- Own price features ---
    prefix = ""  # no prefix for the primary target
    parts.append(lagged_returns(close_target, prefix=prefix))
    parts.append(rolling_volatility(close_target, prefix=prefix))
    parts.append(rolling_momentum(close_target, prefix=prefix))
    parts.append(distance_from_ma(close_target, prefix=prefix))

    # --- Cross-asset context: 1-day return of each context ticker ---
    late_set = set(late_close_tickers)
    for ticker in context_tickers:
        if ticker not in prices:
            logger.warning("Context ticker %r not found; skipping.", ticker)
            continue
        close_ctx = prices[ticker]["Close"].reindex(ref_index).ffill()
        r1 = log_returns(close_ctx)
        if ticker in late_set:
            # This ticker closes after the target futures settlement (~2:30pm ET).
            # Using today's return would leak the 2:30pm→4pm gold price move.
            # Shift by one extra day: feature at t = return from t-2 to t-1.
            r1 = r1.shift(1)
        safe = ticker.replace("=", "").replace("^", "").replace("-", "").lower()
        parts.append(r1.rename(f"{safe}_ret_1d").to_frame())

    # --- Cross-asset ratio feature ---
    if ratio_pair is not None:
        ta, tb = ratio_pair
        if ta in prices and tb in prices:
            parts.append(
                log_ratio(
                    prices[ta]["Close"],
                    prices[tb]["Close"],
                    name=ratio_name,
                    ref_index=ref_index,
                ).to_frame()
            )
        else:
            missing = [t for t in (ta, tb) if t not in prices]
            logger.warning("Ratio tickers missing: %s; skipping ratio feature.", missing)

    result = pd.concat(parts, axis=1)

    _log_feature_summary(result, target_ticker)
    return result


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #

def _log_feature_summary(df: pd.DataFrame, ticker: str) -> None:
    n_nan_rows = df.isnull().any(axis=1).sum()
    logger.info(
        "Price features for %s: %d rows, %d columns, %d NaN rows (warm-up)",
        ticker, len(df), df.shape[1], n_nan_rows,
    )
    for col in df.columns:
        n = df[col].isna().sum()
        if n > 0:
            logger.debug("  %s: %d NaN", col, n)
