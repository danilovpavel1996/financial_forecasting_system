"""Tests for carry / term-structure features.

[REQUIRED] Shift-and-compare leakage tests for:
  - basis_momentum
  - carry_proxy

Additional correctness checks:
  - basis_momentum formula: ret_252d - ret_21d
  - carry_proxy is lagged (using t-1 data at time t)
  - carry_proxy lag=0 raises ValueError
  - rel_basis_momentum sums to zero across basket
  - rel_carry_proxy sums to zero across carry-proxy tickers
  - Tickers without carry pairs have no carry_proxy column
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.carry_features import (
    _basis_momentum_series,
    _carry_proxy_series,
    build_carry_features,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _make_close(n: int = 300, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    prices = 100.0 * np.exp(rng.normal(0, 0.01, n).cumsum())
    idx = pd.bdate_range("2010-01-01", periods=n, freq="B")
    return pd.Series(prices, index=idx, name="Close")


def _make_prices(n: int = 300, seed: int = 0) -> dict:
    close = _make_close(n, seed)
    df = pd.DataFrame({
        "Close": close, "Open": close, "High": close, "Low": close, "Volume": 0
    })
    return df


def _biz_index(n: int, start: str = "2010-01-01") -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=n, freq="B")


# --------------------------------------------------------------------------- #
# [REQUIRED] Shift-and-compare leakage: basis_momentum
# --------------------------------------------------------------------------- #

class TestBasisMomentumNoLookahead:
    """basis_momentum at time t must not depend on data after t."""

    def _check(self, close_full: pd.Series, n_drop: int) -> None:
        close_trunc = close_full.iloc[:-n_drop]

        full = _basis_momentum_series(close_full)
        trunc = _basis_momentum_series(close_trunc)

        shared = full.index.intersection(trunc.index)
        assert len(shared) > 20

        pd.testing.assert_series_equal(
            full.loc[shared].sort_index(),
            trunc.loc[shared].sort_index(),
            check_exact=True,
            obj=f"basis_momentum (n_drop={n_drop})",
        )

    def test_drop_1(self):
        self._check(_make_close(350), 1)

    def test_drop_5(self):
        self._check(_make_close(350), 5)

    def test_drop_21(self):
        self._check(_make_close(350), 21)

    def test_drop_63(self):
        self._check(_make_close(400), 63)

    def test_independent_of_seed(self):
        for seed in [7, 42, 99]:
            self._check(_make_close(350, seed=seed), n_drop=5)


# --------------------------------------------------------------------------- #
# [REQUIRED] Shift-and-compare leakage: carry_proxy
# --------------------------------------------------------------------------- #

class TestCarryProxyNoLookahead:
    """carry_proxy at time t must not depend on data after t."""

    def _check(self, n: int, n_drop: int) -> None:
        ref_full = _biz_index(n)
        ref_trunc = ref_full[:-n_drop]

        close_fut = _make_close(n, seed=10).reindex(ref_full)
        close_etf = _make_close(n, seed=20).reindex(ref_full)

        # Truncate raw data to simulate not having future rows
        close_fut_trunc = close_fut.iloc[:-n_drop]
        close_etf_trunc = close_etf.iloc[:-n_drop]

        cp_full = _carry_proxy_series(close_fut, close_etf, ref_full)
        cp_trunc = _carry_proxy_series(close_fut_trunc, close_etf_trunc, ref_trunc)

        shared = cp_full.index.intersection(cp_trunc.index)
        assert len(shared) > 10

        pd.testing.assert_series_equal(
            cp_full.loc[shared].sort_index(),
            cp_trunc.loc[shared].sort_index(),
            check_exact=True,
            obj=f"carry_proxy (n_drop={n_drop})",
        )

    def test_drop_1(self):
        self._check(100, 1)

    def test_drop_5(self):
        self._check(100, 5)

    def test_drop_21(self):
        self._check(120, 21)


# --------------------------------------------------------------------------- #
# Formula correctness
# --------------------------------------------------------------------------- #

def test_basis_momentum_formula():
    """basis_momentum = log(close[t-21] / close[t-252])."""
    n = 300
    close = _make_close(n)
    bm = _basis_momentum_series(close)

    # Manual: ret_252d - ret_21d = log(close[t]/close[t-252]) - log(close[t]/close[t-21])
    #       = log(close[t-21] / close[t-252])
    expected = np.log(close / close.shift(252)) - np.log(close / close.shift(21))

    pd.testing.assert_series_equal(bm, expected.rename("basis_momentum"), check_exact=True)


def test_basis_momentum_nan_for_first_252_rows():
    """basis_momentum requires 252 days of history; first 252 rows must be NaN."""
    close = _make_close(300)
    bm = _basis_momentum_series(close)
    assert bm.iloc[:252].isna().all(), "First 252 rows should be NaN (warm-up)"
    assert bm.iloc[252:].notna().any(), "Should have valid values after warm-up"


def test_carry_proxy_is_lagged():
    """carry_proxy at t equals log(futures[t-1] / ETF[t-1])."""
    n = 80
    ref = _biz_index(n)
    close_fut = _make_close(n, seed=1).reindex(ref)
    close_etf = _make_close(n, seed=2).reindex(ref)

    cp = _carry_proxy_series(close_fut, close_etf, ref, lag=1)

    raw = np.log(close_fut / close_etf)
    expected_lagged = raw.shift(1)

    pd.testing.assert_series_equal(cp, expected_lagged.rename("carry_proxy"), check_exact=True)


def test_carry_proxy_lag_zero_raises():
    """_carry_proxy_series with lag=0 must raise ValueError."""
    ref = _biz_index(50)
    close = _make_close(50).reindex(ref)
    with pytest.raises(ValueError, match="lag must be >= 1"):
        _carry_proxy_series(close, close, ref, lag=0)


def test_carry_proxy_first_row_is_nan():
    """carry_proxy first row is NaN after lag=1 shift."""
    ref = _biz_index(50)
    close = _make_close(50).reindex(ref)
    cp = _carry_proxy_series(close, close, ref)
    assert np.isnan(cp.iloc[0]), "First row after lag shift must be NaN"


# --------------------------------------------------------------------------- #
# Cross-sectional relative features
# --------------------------------------------------------------------------- #

def test_rel_basis_momentum_sums_to_zero():
    """rel_basis_momentum sums to zero across basket at every valid date."""
    n = 350
    ref = _biz_index(n)
    tickers = ["GC=F", "SI=F", "PL=F", "PA=F", "CL=F"]
    prices = {t: _make_prices(n, seed=i) for i, t in enumerate(tickers)}

    feats = build_carry_features(prices, tickers, ref, carry_pairs={})

    # Stack rel_basis_momentum for all tickers
    rel_bm = pd.concat(
        [feats[t]["rel_basis_momentum"].rename(t) for t in tickers],
        axis=1,
    )
    row_sums = rel_bm.sum(axis=1)
    valid = row_sums.dropna()
    assert len(valid) > 0
    np.testing.assert_allclose(valid.values, 0.0, atol=1e-10,
                               err_msg="rel_basis_momentum must sum to 0 across basket")


def test_rel_carry_proxy_sums_to_zero_for_eligible_tickers():
    """rel_carry_proxy sums to zero across the carry-proxy tickers."""
    n = 200
    ref = _biz_index(n)
    tickers = ["GC=F", "SI=F", "HG=F"]
    prices = {
        "GC=F": _make_prices(n, seed=0),
        "SI=F": _make_prices(n, seed=1),
        "HG=F": _make_prices(n, seed=2),
        "GLD":  _make_prices(n, seed=3),
        "SLV":  _make_prices(n, seed=4),
    }
    carry_pairs = {"GC=F": "GLD", "SI=F": "SLV"}

    feats = build_carry_features(prices, tickers, ref, carry_pairs=carry_pairs)

    # rel_carry_proxy present for all tickers (uniform column set)
    assert "rel_carry_proxy" in feats["GC=F"].columns
    assert "rel_carry_proxy" in feats["SI=F"].columns
    assert "rel_carry_proxy" in feats["HG=F"].columns
    # Non-carry ticker's rel_carry_proxy is 0 (constant sentinel)
    assert (feats["HG=F"]["rel_carry_proxy"] == 0.0).all()

    # Sum of rel_carry_proxy across GC=F and SI=F must be 0 (only 2 tickers)
    rel = pd.concat([
        feats["GC=F"]["rel_carry_proxy"].rename("GC=F"),
        feats["SI=F"]["rel_carry_proxy"].rename("SI=F"),
    ], axis=1)
    row_sums = rel.sum(axis=1).dropna()
    np.testing.assert_allclose(row_sums.values, 0.0, atol=1e-10,
                               err_msg="rel_carry_proxy must sum to 0 across eligible tickers")


def test_non_carry_ticker_has_zero_carry_proxy():
    """Tickers not in carry_pairs get carry_proxy = 0 (uniform column set)."""
    n = 300
    ref = _biz_index(n)
    tickers = ["GC=F", "SI=F", "CL=F"]
    prices = {
        "GC=F": _make_prices(n, seed=0),
        "SI=F": _make_prices(n, seed=1),
        "CL=F": _make_prices(n, seed=2),
        "GLD":  _make_prices(n, seed=3),
        "SLV":  _make_prices(n, seed=4),
    }
    feats = build_carry_features(prices, tickers, ref, carry_pairs={"GC=F": "GLD", "SI=F": "SLV"})

    # CL=F should have carry columns but filled with 0 (consistent column set)
    assert "carry_proxy" in feats["CL=F"].columns
    assert "carry_proxy_chg_21d" in feats["CL=F"].columns
    assert "rel_carry_proxy" in feats["CL=F"].columns
    assert (feats["CL=F"]["carry_proxy"] == 0.0).all()
    assert (feats["CL=F"]["carry_proxy_chg_21d"] == 0.0).all()
    assert (feats["CL=F"]["rel_carry_proxy"] == 0.0).all()
    # basis_momentum should still be there and non-trivial
    assert "basis_momentum" in feats["CL=F"].columns


def test_all_tickers_same_column_set():
    """All tickers returned by build_carry_features have the same columns."""
    n = 300
    ref = _biz_index(n)
    tickers = ["GC=F", "SI=F", "HG=F", "CL=F"]
    prices = {t: _make_prices(n, seed=i) for i, t in enumerate(tickers)}
    prices["GLD"] = _make_prices(n, seed=10)
    prices["SLV"] = _make_prices(n, seed=11)

    feats = build_carry_features(prices, tickers, ref, carry_pairs={"GC=F": "GLD", "SI=F": "SLV"})

    col_sets = [set(feats[t].columns) for t in tickers]
    assert len(set(frozenset(cs) for cs in col_sets)) == 1, (
        "All tickers must have the same column set, got: "
        + str([dict(zip(tickers, col_sets))])
    )


def test_build_carry_features_missing_etf_skips_gracefully(caplog):
    """If ETF ticker is absent from prices, carry proxy cols are all zero."""
    import logging
    n = 300
    ref = _biz_index(n)
    tickers = ["GC=F"]
    prices = {"GC=F": _make_prices(n, seed=0)}  # GLD is missing

    with caplog.at_level(logging.WARNING, logger="src.features.carry_features"):
        feats = build_carry_features(prices, tickers, ref, carry_pairs={"GC=F": "GLD"})

    # No carry pairs were fulfilled → no carry_proxy columns at all
    assert "carry_proxy" not in feats["GC=F"].columns
    assert "basis_momentum" in feats["GC=F"].columns
