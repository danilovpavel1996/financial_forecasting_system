"""Tests for cross-sectional ranking metrics.

[REQUIRED] tests:
  1. CS-RIC = 1.0 for perfect ranking (predictions agree with realized order).
  2. CS-RIC = -1.0 for reversed ranking.
  3. CS-RIC = NaN for constant predictions (no signal).
  4. Long-short return correct for known positions.
  5. Spread capture is between 0 and 1 when signal is perfect.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.eval.cross_metrics import (
    cs_ric_series,
    cs_ric_stability,
    ls_return_series,
    mean_cs_ric,
    mean_spread_capture,
    spread_capture_series,
    std_cs_ric,
)

_ASSETS = ["GC=F", "SI=F", "PL=F", "PA=F"]
_DATE = pd.bdate_range("2020-01-01", periods=1)[0]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_df(data: dict, n_dates: int = 1, start: str = "2020-01-01") -> pd.DataFrame:
    """Build a (dates × assets) DataFrame for a single date or multiple dates."""
    dates = pd.bdate_range(start, periods=n_dates)
    if n_dates == 1:
        return pd.DataFrame([data], index=dates, columns=_ASSETS)
    # data must be a list of dicts
    return pd.DataFrame(data, index=dates, columns=_ASSETS)


# ─────────────────────────────────────────────────────────────────────────────
# [REQUIRED] CS-RIC on known rankings
# ─────────────────────────────────────────────────────────────────────────────

def test_cs_ric_perfect_ranking():
    """[REQUIRED] When predictions perfectly match realized order, CS-RIC = 1."""
    pred = _make_df({"GC=F": 4.0, "SI=F": 3.0, "PL=F": 2.0, "PA=F": 1.0})
    real = _make_df({"GC=F": 0.04, "SI=F": 0.02, "PL=F": 0.01, "PA=F": 0.005})
    result = cs_ric_series(pred, real)
    assert result.iloc[0] == pytest.approx(1.0, abs=1e-9)


def test_cs_ric_reversed_ranking():
    """[REQUIRED] When predictions perfectly reverse realized order, CS-RIC = -1."""
    pred = _make_df({"GC=F": 1.0, "SI=F": 2.0, "PL=F": 3.0, "PA=F": 4.0})
    real = _make_df({"GC=F": 0.04, "SI=F": 0.02, "PL=F": 0.01, "PA=F": 0.005})
    result = cs_ric_series(pred, real)
    assert result.iloc[0] == pytest.approx(-1.0, abs=1e-9)


def test_cs_ric_constant_predictions_is_nan():
    """[REQUIRED] All identical predictions → CS-RIC = NaN (undefined)."""
    pred = _make_df({"GC=F": 0.0, "SI=F": 0.0, "PL=F": 0.0, "PA=F": 0.0})
    real = _make_df({"GC=F": 0.04, "SI=F": 0.02, "PL=F": 0.01, "PA=F": 0.005})
    result = cs_ric_series(pred, real)
    assert np.isnan(result.iloc[0])


def test_cs_ric_hand_computed():
    """[REQUIRED] CS-RIC matches a hand-computed Spearman value.

    pred = [4, 2, 1, 3]  (GC, SI, PL, PA)
    real = [0.04, 0.01, 0.03, 0.02]

    Pred ranks  (asc): GC=4, SI=2, PL=1, PA=3  → pred_rank = [4,2,1,3]
    Real ranks  (asc): GC=4, SI=1, PL=3, PA=2  → real_rank = [4,1,3,2]

    d = [0, 1, -2, 1], d² = [0, 1, 4, 1], Σd² = 6
    Spearman = 1 - 6*6 / (4*(16-1)) = 1 - 36/60 = 1 - 0.6 = 0.4
    """
    pred = _make_df({"GC=F": 4.0, "SI=F": 2.0, "PL=F": 1.0, "PA=F": 3.0})
    real = _make_df({"GC=F": 0.04, "SI=F": 0.01, "PL=F": 0.03, "PA=F": 0.02})
    result = cs_ric_series(pred, real)
    assert result.iloc[0] == pytest.approx(0.4, abs=1e-9)


def test_cs_ric_series_length_matches_dates():
    """CS-RIC series has one value per date in both DataFrames."""
    n = 10
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2020-01-01", periods=n)
    pred_df = pd.DataFrame(rng.normal(0, 1, (n, 4)), index=dates, columns=_ASSETS)
    real_df = pd.DataFrame(rng.normal(0, 1, (n, 4)), index=dates, columns=_ASSETS)
    result = cs_ric_series(pred_df, real_df)
    assert len(result) == n


def test_mean_cs_ric_nanmean():
    """mean_cs_ric ignores NaN entries."""
    cs = pd.Series([0.4, np.nan, 0.6, 0.2])
    assert mean_cs_ric(cs) == pytest.approx((0.4 + 0.6 + 0.2) / 3, abs=1e-9)


def test_cs_ric_stability_fraction_positive():
    """cs_ric_stability returns fraction of positive (not NaN) values."""
    cs = pd.Series([0.3, -0.1, 0.5, np.nan, 0.2])
    # 3 positive out of 4 non-NaN
    assert cs_ric_stability(cs) == pytest.approx(3 / 4, abs=1e-9)


# ─────────────────────────────────────────────────────────────────────────────
# [REQUIRED] Long-short return
# ─────────────────────────────────────────────────────────────────────────────

def test_ls_return_long_top_short_bottom():
    """[REQUIRED] L/S return = (realized_long - realized_short) / 2 (gross=2)."""
    # Long GC=F (+1), short PA=F (-1), flat others
    pos = _make_df({"GC=F": 1.0, "SI=F": 0.0, "PL=F": 0.0, "PA=F": -1.0})
    real = _make_df({"GC=F": 0.04, "SI=F": 0.01, "PL=F": 0.02, "PA=F": -0.01})
    # gross_pnl = 1*0.04 + 0*0.01 + 0*0.02 + (-1)*(-0.01) = 0.04 + 0.01 = 0.05
    # gross_exp = |1| + 0 + 0 + |-1| = 2
    # ls_return = 0.05 / 2 = 0.025
    result = ls_return_series(pos, real)
    assert result.iloc[0] == pytest.approx(0.025, abs=1e-9)


def test_ls_return_all_zero_positions():
    """All-zero positions → NaN return (gross exposure = 0)."""
    pos = _make_df({"GC=F": 0.0, "SI=F": 0.0, "PL=F": 0.0, "PA=F": 0.0})
    real = _make_df({"GC=F": 0.04, "SI=F": 0.01, "PL=F": 0.02, "PA=F": -0.01})
    result = ls_return_series(pos, real)
    assert np.isnan(result.iloc[0])


# ─────────────────────────────────────────────────────────────────────────────
# Spread capture
# ─────────────────────────────────────────────────────────────────────────────

def test_spread_capture_perfect_signal():
    """Perfect signal captures 100% of the spread.

    Long the best performer, short the worst.
    realized = [0.04, 0.01, 0.02, -0.01]
    Best = GC=F (0.04), worst = PA=F (-0.01)
    Long GC, short PA: pnl = (0.04 - (-0.01)) / 2 = 0.025
    Spread = 0.04 - (-0.01) = 0.05
    Capture = 0.025 / 0.05 = 0.5

    Wait — capture can only be 1.0 if we take unnormalized positions.
    With gross_exp=2, the L/S return is halved, so capture = 0.5.
    With gross_exp=1 (long 0.5, short -0.5), capture = 1.0.

    Using gross_exp=2 (standard ±1 positions), capture = 0.5.
    """
    pos = _make_df({"GC=F": 1.0, "SI=F": 0.0, "PL=F": 0.0, "PA=F": -1.0})
    real = _make_df({"GC=F": 0.04, "SI=F": 0.01, "PL=F": 0.02, "PA=F": -0.01})
    sc = spread_capture_series(pos, real)
    # spread = 0.04 - (-0.01) = 0.05, pnl = 0.025
    assert sc.iloc[0] == pytest.approx(0.5, abs=1e-9)


def test_spread_capture_nan_when_spread_zero():
    """Spread capture = NaN when all assets have the same realized return."""
    pos = _make_df({"GC=F": 1.0, "SI=F": 0.0, "PL=F": 0.0, "PA=F": -1.0})
    real = _make_df({"GC=F": 0.01, "SI=F": 0.01, "PL=F": 0.01, "PA=F": 0.01})
    sc = spread_capture_series(pos, real)
    assert np.isnan(sc.iloc[0])
