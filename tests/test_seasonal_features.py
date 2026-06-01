"""Tests for seasonal features (month_sin / month_cos).

These features carry no look-ahead risk (they are computed from the date
index alone), so leakage tests are trivial.  Correctness tests verify:
  - Values are in [-1, 1]
  - Formula: sin(2π × month / 12) and cos(2π × month / 12)
  - No NaN for any trading day
  - Cyclical continuity: December and January are adjacent on the circle
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.seasonal_features import build_seasonal_features


def _biz_index(n: int = 252, start: str = "2010-01-01") -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=n, freq="B")


def test_returns_dataframe():
    idx = _biz_index(100)
    feats = build_seasonal_features(idx)
    assert isinstance(feats, pd.DataFrame)
    assert feats.index.equals(idx)


def test_expected_columns():
    feats = build_seasonal_features(_biz_index(50))
    assert "month_sin" in feats.columns
    assert "month_cos" in feats.columns
    assert len(feats.columns) == 2


def test_no_nan():
    """Seasonal features must never be NaN."""
    feats = build_seasonal_features(_biz_index(500))
    assert feats.notna().all().all(), "Seasonal features must have no NaN values"


def test_values_in_unit_range():
    """sin and cos values must be in [-1, 1]."""
    feats = build_seasonal_features(_biz_index(252))
    assert (feats["month_sin"].between(-1.0, 1.0)).all()
    assert (feats["month_cos"].between(-1.0, 1.0)).all()


def test_formula_correctness():
    """Verify month_sin and month_cos match the expected formula."""
    idx = pd.bdate_range("2015-01-01", periods=252)
    feats = build_seasonal_features(idx)

    months = idx.month.astype(float)
    expected_sin = np.sin(2.0 * np.pi * months / 12.0)
    expected_cos = np.cos(2.0 * np.pi * months / 12.0)

    np.testing.assert_allclose(feats["month_sin"].values, expected_sin, rtol=1e-12)
    np.testing.assert_allclose(feats["month_cos"].values, expected_cos, rtol=1e-12)


def test_january_spot_check():
    """January (month=1): sin = sin(π/6) ≈ 0.5, cos = cos(π/6) ≈ 0.866."""
    jan_idx = pd.DatetimeIndex(["2020-01-02"])  # a trading day in January
    feats = build_seasonal_features(jan_idx)

    expected_sin = np.sin(2.0 * np.pi * 1 / 12)  # ≈ 0.5
    expected_cos = np.cos(2.0 * np.pi * 1 / 12)  # ≈ 0.866

    assert abs(feats["month_sin"].iloc[0] - expected_sin) < 1e-12
    assert abs(feats["month_cos"].iloc[0] - expected_cos) < 1e-12


def test_june_spot_check():
    """June (month=6): sin = sin(π) = 0, cos = cos(π) = -1."""
    jun_idx = pd.DatetimeIndex(["2020-06-01"])
    feats = build_seasonal_features(jun_idx)

    expected_sin = np.sin(2.0 * np.pi * 6 / 12)  # = sin(π) ≈ 0
    expected_cos = np.cos(2.0 * np.pi * 6 / 12)  # = cos(π) = -1

    assert abs(feats["month_sin"].iloc[0] - expected_sin) < 1e-12
    assert abs(feats["month_cos"].iloc[0] - expected_cos) < 1e-12


def test_december_january_cyclical_proximity():
    """December and January should be close on the unit circle.

    This verifies the cyclical encoding is continuous at the year boundary.
    The angular distance between December (month=12) and January (month=1)
    should equal the distance between any other adjacent months.
    """
    dec_idx = pd.DatetimeIndex(["2020-12-01"])
    jan_idx = pd.DatetimeIndex(["2020-01-02"])

    dec_feats = build_seasonal_features(dec_idx)
    jan_feats = build_seasonal_features(jan_idx)

    # Angular difference: one month step = 2π/12 radians
    one_month_angle = 2.0 * np.pi / 12.0

    # Euclidean distance on the unit circle for one-month step
    expected_dist = np.sqrt(
        (np.sin(0) - np.sin(one_month_angle))**2 +
        (np.cos(0) - np.cos(one_month_angle))**2
    )

    actual_dist = np.sqrt(
        (dec_feats["month_sin"].iloc[0] - jan_feats["month_sin"].iloc[0])**2 +
        (dec_feats["month_cos"].iloc[0] - jan_feats["month_cos"].iloc[0])**2
    )

    assert abs(actual_dist - expected_dist) < 1e-10, (
        f"Dec-Jan distance {actual_dist:.6f} != one-month step distance {expected_dist:.6f}"
    )


def test_same_month_same_year_identical():
    """All trading days in the same month have the same month_sin / month_cos."""
    idx = pd.bdate_range("2020-03-01", "2020-03-31")
    feats = build_seasonal_features(idx)
    assert (feats["month_sin"] == feats["month_sin"].iloc[0]).all()
    assert (feats["month_cos"] == feats["month_cos"].iloc[0]).all()


def test_extended_history_no_nan():
    """Extended 2005–2024 history produces no NaN."""
    idx = pd.bdate_range("2005-01-01", "2024-12-31")
    feats = build_seasonal_features(idx)
    assert feats.notna().all().all()
    assert len(feats) == len(idx)
