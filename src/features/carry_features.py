"""Carry and term-structure features for commodity futures basket.

Features per asset:
  basis_momentum      = ret_252d - ret_21d  (all 9 commodities)
  carry_proxy         = log(futures / ETF)  (GC=F/GLD and SI=F/SLV only)
  carry_proxy_chg_21d = 21-day change in carry_proxy  (carry_proxy tickers only)
  rel_basis_momentum  = basis_momentum - basket average
  rel_carry_proxy     = carry_proxy - average of carry-proxy tickers  (carry_proxy tickers only)

Leakage-safety notes
--------------------
basis_momentum uses close[t-252] and close[t-21]: both are strictly past data at t.

carry_proxy is shifted by 1 trading day before use.  ETFs (GLD, SLV) close at
4pm ET while futures (GC=F, SI=F) settle at ~2:30pm ET.  Using the same-day
ETF close as a signal for tomorrow's futures return would embed price moves
that happen after futures settlement — a look-ahead relative to the futures
close time.  The 1-day lag eliminates this timing mismatch.

Shift-and-compare leakage tests live in tests/test_carry_features.py.
"""
from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Default carry proxy pairs: {futures_ticker: etf_ticker}
_DEFAULT_CARRY_PAIRS: dict[str, str] = {
    "GC=F": "GLD",
    "SI=F": "SLV",
}


def _basis_momentum_series(close: pd.Series) -> pd.Series:
    """ret_252d - ret_21d = log(close[t-21] / close[t-252]).

    Captures the term-structure component of 12-month momentum (excluding
    the most recent month) following Koijen et al. (2018).  Uses only
    data at or before t; requires 252 trading days of history to be non-NaN.
    """
    ret_252d = np.log(close / close.shift(252))
    ret_21d = np.log(close / close.shift(21))
    return (ret_252d - ret_21d).rename("basis_momentum")


def _carry_proxy_series(
    futures_close: pd.Series,
    etf_close: pd.Series,
    ref_index: pd.DatetimeIndex,
    lag: int = 1,
) -> pd.Series:
    """log(futures / ETF) lagged by `lag` days.

    The raw ratio log(GC=F[t] / GLD[t]) mixes a 2:30pm futures settlement
    with a 4pm ETF close.  Shifting by 1 day means the feature at t uses
    yesterday's ratio, which is fully resolved on both legs before t opens.

    Parameters
    ----------
    futures_close : futures close price series (any index)
    etf_close     : ETF close price series (any index)
    ref_index     : common trading-day DatetimeIndex to align both series
    lag           : number of trading days to shift (must be ≥ 1)
    """
    if lag < 1:
        raise ValueError(f"lag must be >= 1 to prevent look-ahead; got {lag}")
    fut = futures_close.reindex(ref_index).ffill()
    etf = etf_close.reindex(ref_index).ffill()
    raw = np.log(fut / etf)
    return raw.shift(lag).rename("carry_proxy")


def build_carry_features(
    prices: dict[str, pd.DataFrame],
    basket_tickers: Sequence[str],
    ref_index: pd.DatetimeIndex,
    carry_pairs: dict[str, str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Build carry / term-structure features for each basket ticker.

    Parameters
    ----------
    prices:         dict mapping ticker → OHLCV DataFrame (all tickers loaded).
    basket_tickers: ordered list of N tickers in the ranking basket.
    ref_index:      common trading-day DatetimeIndex.
    carry_pairs:    {futures_ticker: etf_ticker} for carry proxy computation.
                    Defaults to {"GC=F": "GLD", "SI=F": "SLV"}.

    Returns
    -------
    dict mapping ticker → pd.DataFrame with carry feature columns.
    Tickers in carry_pairs get: basis_momentum, carry_proxy,
    carry_proxy_chg_21d, rel_basis_momentum, rel_carry_proxy.
    All other tickers get: basis_momentum, rel_basis_momentum.
    """
    if carry_pairs is None:
        carry_pairs = _DEFAULT_CARRY_PAIRS

    basket_tickers = list(basket_tickers)

    # ── Step 1: basis_momentum per asset ─────────────────────────────────────
    bm: dict[str, pd.Series] = {}
    for ticker in basket_tickers:
        if ticker not in prices:
            logger.warning("Carry features: ticker %r not in prices; skipping.", ticker)
            continue
        close = prices[ticker]["Close"].reindex(ref_index).ffill()
        bm[ticker] = _basis_momentum_series(close)

    # Basket mean of basis_momentum (NaN-safe: skips missing assets)
    basket_bm_mean: pd.Series
    if bm:
        bm_df = pd.concat(list(bm.values()), axis=1)
        basket_bm_mean = bm_df.mean(axis=1)
    else:
        basket_bm_mean = pd.Series(np.nan, index=ref_index)

    # ── Step 2: carry_proxy for eligible tickers ──────────────────────────────
    cp: dict[str, pd.Series] = {}
    for fut_ticker, etf_ticker in carry_pairs.items():
        if fut_ticker not in prices:
            logger.warning("Carry proxy: futures ticker %r not in prices; skipping.", fut_ticker)
            continue
        if etf_ticker not in prices:
            logger.warning("Carry proxy: ETF ticker %r not in prices; skipping.", etf_ticker)
            continue
        cp[fut_ticker] = _carry_proxy_series(
            prices[fut_ticker]["Close"],
            prices[etf_ticker]["Close"],
            ref_index,
        )

    # Cross-sectional mean of carry proxy (only over assets that have it)
    cp_basket_tickers = [t for t in basket_tickers if t in cp]
    basket_cp_mean: pd.Series | None = None
    if cp_basket_tickers:
        cp_df = pd.concat([cp[t].rename(t) for t in cp_basket_tickers], axis=1)
        basket_cp_mean = cp_df.mean(axis=1)

    # ── Step 3: assemble per-ticker output ────────────────────────────────────
    # All tickers must share the SAME column set so pd.concat in pooled_dataset
    # produces a consistent feature matrix (no NaN gaps from missing columns).
    # Tickers without a carry proxy get carry_proxy = 0 (constant sentinel).
    include_cp_cols = bool(cp)  # at least one valid carry pair was found
    _zero = pd.Series(0.0, index=ref_index)

    result: dict[str, pd.DataFrame] = {}
    for ticker in basket_tickers:
        cols: dict[str, pd.Series] = {}

        if ticker in bm:
            cols["basis_momentum"] = bm[ticker]
            cols["rel_basis_momentum"] = bm[ticker] - basket_bm_mean
        else:
            cols["basis_momentum"] = _zero.copy()
            cols["rel_basis_momentum"] = _zero.copy()

        if include_cp_cols:
            if ticker in cp:
                cols["carry_proxy"] = cp[ticker]
                cols["carry_proxy_chg_21d"] = cp[ticker].diff(21)
                cols["rel_carry_proxy"] = (
                    cp[ticker] - basket_cp_mean
                    if basket_cp_mean is not None
                    else _zero.copy()
                )
            else:
                # Non-carry ticker: zero fill keeps the column set uniform
                cols["carry_proxy"] = _zero.copy()
                cols["carry_proxy_chg_21d"] = _zero.copy()
                cols["rel_carry_proxy"] = _zero.copy()

        result[ticker] = pd.DataFrame(cols, index=ref_index)

    if basket_tickers:
        sample_ticker = next(t for t in basket_tickers if t in result)
        logger.info(
            "Carry features built: %d assets × %d dates × %d columns",
            len(basket_tickers), len(ref_index), result[sample_ticker].shape[1],
        )
    return result
