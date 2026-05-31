"""Leakage and correctness tests for cross-sectional features.

[REQUIRED] tests:
  1. Shift-and-compare: removing future rows must not change historical values.
  2. Relative features sum to zero across the basket at every date.
  3. Ratio features are symmetric / consistent with price_features.log_ratio.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.cross_features import build_cross_features
from src.features.price_features import log_ratio


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_METALS = ["GC=F", "SI=F", "PL=F", "PA=F"]


def _make_prices(n: int = 200, seed: int = 0) -> dict[str, pd.DataFrame]:
    """Synthetic OHLCV dict for all 4 metals on a common business-day index."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n, freq="B")
    out: dict[str, pd.DataFrame] = {}
    for i, ticker in enumerate(_METALS):
        close = 100.0 * np.exp(rng.normal(0, 0.012, n).cumsum())
        df = pd.DataFrame(
            {"Open": close, "High": close, "Low": close, "Close": close, "Volume": 0},
            index=idx,
        )
        out[ticker] = df
    return out


def _ref_index(n: int = 200) -> pd.DatetimeIndex:
    return pd.bdate_range("2015-01-01", periods=n, freq="B")


# ─────────────────────────────────────────────────────────────────────────────
# [REQUIRED] Shift-and-compare leakage test
# ─────────────────────────────────────────────────────────────────────────────

class TestCrossFeaturesNoLookahead:
    """[REQUIRED] Cross-asset features must not depend on data after time t.

    Method: compute features on full series, then truncate the last n_drop rows
    from every asset and recompute.  All shared-date values must be identical.
    """

    def _check(self, n_drop: int) -> None:
        n = 200
        prices_full = _make_prices(n, seed=42)
        ref_full = _ref_index(n)

        prices_trunc = {
            t: df.iloc[:-n_drop] for t, df in prices_full.items()
        }
        ref_trunc = ref_full[:-n_drop]

        feats_full = build_cross_features(prices_full, _METALS, ref_full)
        feats_trunc = build_cross_features(prices_trunc, _METALS, ref_trunc)

        for ticker in _METALS:
            shared = feats_full[ticker].index.intersection(feats_trunc[ticker].index)
            assert len(shared) > 50, f"Too few shared rows: {len(shared)}"

            pd.testing.assert_frame_equal(
                feats_full[ticker].loc[shared].sort_index(),
                feats_trunc[ticker].loc[shared].sort_index(),
                check_exact=True,
                obj=f"cross_features[{ticker}] (n_drop={n_drop})",
            )

    def test_drop_1(self):
        self._check(1)

    def test_drop_5(self):
        self._check(5)

    def test_drop_21(self):
        self._check(21)


# ─────────────────────────────────────────────────────────────────────────────
# [REQUIRED] Relative features sum to zero across the basket
# ─────────────────────────────────────────────────────────────────────────────

def test_relative_features_sum_to_zero():
    """Sum of rel_mom_10d (and all other rel_ columns) across 4 assets = 0.

    rel_x[a, t] = x[a, t] - basket_mean_x[t].
    Sum over a: (sum_a x[a,t]) - 4 * basket_mean_x[t]
               = 4 * basket_mean - 4 * basket_mean = 0.  [QED]
    """
    n = 200
    prices = _make_prices(n, seed=7)
    ref = _ref_index(n)
    feats = build_cross_features(prices, _METALS, ref)

    rel_cols = [c for c in feats[_METALS[0]].columns if c.startswith("rel_")]
    assert len(rel_cols) > 0, "No rel_ columns found"

    for col in rel_cols:
        stacked = pd.concat(
            [feats[t][col].rename(t) for t in _METALS], axis=1
        )
        row_sums = stacked.dropna().sum(axis=1)
        # All row sums must be ~0 (floating-point)
        max_abs = row_sums.abs().max()
        assert max_abs < 1e-10, (
            f"rel column {col!r}: sum across basket is not 0 (max|sum|={max_abs:.2e})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Ratio features are consistent with log_ratio from price_features
# ─────────────────────────────────────────────────────────────────────────────

def test_gold_silver_ratio_matches_log_ratio():
    """gold_silver_ratio must equal log_ratio(GC=F, SI=F) computed independently."""
    n = 150
    prices = _make_prices(n, seed=3)
    ref = _ref_index(n)

    feats = build_cross_features(prices, _METALS, ref)
    # All assets should share the same ratio on a given date
    gc_ratio = feats["GC=F"]["gold_silver_ratio"]
    si_ratio = feats["SI=F"]["gold_silver_ratio"]
    pd.testing.assert_series_equal(gc_ratio, si_ratio, check_names=False)

    # Must match independent computation
    expected = log_ratio(
        prices["GC=F"]["Close"], prices["SI=F"]["Close"],
        name="gold_silver_ratio", ref_index=ref,
    )
    pd.testing.assert_series_equal(gc_ratio, expected)


def test_ratio_all_assets_identical_on_same_date():
    """Ratio features take the same value for all 4 assets on a given date."""
    n = 120
    prices = _make_prices(n, seed=5)
    ref = _ref_index(n)
    feats = build_cross_features(prices, _METALS, ref)

    ratio_cols = [c for c in feats[_METALS[0]].columns if "ratio" in c]
    assert len(ratio_cols) > 0

    for col in ratio_cols:
        first_vals = feats[_METALS[0]][col]
        for ticker in _METALS[1:]:
            pd.testing.assert_series_equal(
                first_vals, feats[ticker][col], check_names=False,
                obj=f"{col!r} should be identical across assets",
            )


def test_all_four_assets_get_features():
    """build_cross_features returns one DataFrame per metal."""
    prices = _make_prices(100)
    ref = _ref_index(100)
    feats = build_cross_features(prices, _METALS, ref)
    assert set(feats.keys()) == set(_METALS)


def test_cross_features_index_matches_ref():
    """All returned DataFrames are indexed by ref_index."""
    n = 150
    prices = _make_prices(n)
    ref = _ref_index(n)
    feats = build_cross_features(prices, _METALS, ref)
    for ticker in _METALS:
        assert feats[ticker].index.equals(ref), (
            f"{ticker}: feature index does not match ref_index"
        )


def test_missing_ticker_raises():
    """KeyError when a required metal ticker is absent from prices."""
    prices = _make_prices(100)
    del prices["PA=F"]
    ref = _ref_index(100)
    with pytest.raises(KeyError, match="PA=F"):
        build_cross_features(prices, _METALS, ref)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 8: 9-asset basket tests
# ─────────────────────────────────────────────────────────────────────────────

_9_ASSETS = ["GC=F", "SI=F", "PL=F", "PA=F", "CL=F", "NG=F", "HG=F", "ZC=F", "ZS=F"]


def _make_prices_9(n: int = 200, seed: int = 0) -> dict[str, pd.DataFrame]:
    """Synthetic OHLCV dict for all 9 Phase-8 assets."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n, freq="B")
    out: dict[str, pd.DataFrame] = {}
    for ticker in _9_ASSETS:
        close = 100.0 * np.exp(rng.normal(0, 0.012, n).cumsum())
        out[ticker] = pd.DataFrame(
            {"Open": close, "High": close, "Low": close, "Close": close, "Volume": 0},
            index=idx,
        )
    return out


def test_relative_features_sum_to_zero_9_assets():
    """With 9 assets, relative features sum to zero across the full basket."""
    n = 200
    prices = _make_prices_9(n)
    ref = _ref_index(n)
    feats = build_cross_features(prices, _9_ASSETS, ref)

    rel_cols = [c for c in feats[_9_ASSETS[0]].columns if c.startswith("rel_")]
    assert len(rel_cols) > 0, "No rel_ columns found in 9-asset features"

    for col in rel_cols:
        stacked = pd.concat(
            [feats[t][col].rename(t) for t in _9_ASSETS], axis=1
        )
        row_sums = stacked.dropna().sum(axis=1)
        max_abs = row_sums.abs().max()
        assert max_abs < 1e-10, (
            f"9-asset {col!r}: basket sum not zero (max|sum|={max_abs:.2e})"
        )


def test_all_9_assets_get_features():
    """build_cross_features returns one DataFrame per asset for 9-asset basket."""
    prices = _make_prices_9(150)
    ref = _ref_index(150)
    feats = build_cross_features(prices, _9_ASSETS, ref)
    assert set(feats.keys()) == set(_9_ASSETS)


def test_leakage_9_assets_drop_5():
    """Shift-and-compare leakage test: 9-asset basket, drop last 5 rows."""
    n = 200
    prices_full = _make_prices_9(n)
    ref_full = _ref_index(n)

    prices_trunc = {t: df.iloc[:-5] for t, df in prices_full.items()}
    ref_trunc = ref_full[:-5]

    feats_full = build_cross_features(prices_full, _9_ASSETS, ref_full)
    feats_trunc = build_cross_features(prices_trunc, _9_ASSETS, ref_trunc)

    for ticker in _9_ASSETS:
        shared = feats_full[ticker].index.intersection(feats_trunc[ticker].index)
        assert len(shared) > 50
        pd.testing.assert_frame_equal(
            feats_full[ticker].loc[shared].sort_index(),
            feats_trunc[ticker].loc[shared].sort_index(),
            check_exact=True,
            obj=f"9-asset cross_features[{ticker}] leakage test",
        )
