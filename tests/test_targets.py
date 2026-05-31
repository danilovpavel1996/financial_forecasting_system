"""Tests for src/targets.py — forward log return construction."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.targets import forward_log_return


def _make_close(prices, start="2020-01-01"):
    idx = pd.date_range(start, periods=len(prices), freq="B")
    return pd.Series(prices, index=idx, name="Close", dtype=float)


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------

class TestForwardLogReturnCorrectness:
    def test_single_step_manual(self):
        """r[t] = log(close[t+1] / close[t]) for horizon=1."""
        close = _make_close([100.0, 110.0, 121.0, 133.1])
        result = forward_log_return(close, horizon=1)
        expected_0 = np.log(110.0 / 100.0)
        expected_1 = np.log(121.0 / 110.0)
        expected_2 = np.log(133.1 / 121.0)
        assert np.isclose(result.iloc[0], expected_0)
        assert np.isclose(result.iloc[1], expected_1)
        assert np.isclose(result.iloc[2], expected_2)

    def test_two_step_manual(self):
        """r[t] = log(close[t+2] / close[t]) for horizon=2."""
        close = _make_close([100.0, 110.0, 121.0, 133.1])
        result = forward_log_return(close, horizon=2)
        expected_0 = np.log(121.0 / 100.0)
        expected_1 = np.log(133.1 / 110.0)
        assert np.isclose(result.iloc[0], expected_0)
        assert np.isclose(result.iloc[1], expected_1)

    def test_flat_prices_give_zero(self):
        close = _make_close([50.0] * 5)
        result = forward_log_return(close, horizon=1)
        assert np.allclose(result.iloc[:-1], 0.0)

    def test_name_includes_horizon(self):
        close = _make_close([1.0, 2.0, 3.0])
        assert forward_log_return(close, horizon=1).name == "fwd_ret_1d"
        assert forward_log_return(close, horizon=5).name == "fwd_ret_5d"

    def test_output_index_matches_input(self):
        close = _make_close([100.0, 110.0, 120.0, 130.0])
        result = forward_log_return(close, horizon=1)
        pd.testing.assert_index_equal(result.index, close.index)


# ---------------------------------------------------------------------------
# NaN handling
# ---------------------------------------------------------------------------

class TestNaNHandling:
    def test_last_row_is_nan_horizon_1(self):
        close = _make_close([100.0, 110.0, 121.0])
        result = forward_log_return(close, horizon=1)
        assert np.isnan(result.iloc[-1])
        assert np.isfinite(result.iloc[-2])

    def test_last_two_rows_nan_horizon_2(self):
        close = _make_close([100.0, 110.0, 121.0, 133.1])
        result = forward_log_return(close, horizon=2)
        assert np.isnan(result.iloc[-1])
        assert np.isnan(result.iloc[-2])
        assert np.isfinite(result.iloc[0])
        assert np.isfinite(result.iloc[1])

    def test_exactly_horizon_nan_rows(self):
        n = 10
        horizon = 3
        close = _make_close(np.linspace(100.0, 200.0, n))
        result = forward_log_return(close, horizon=horizon)
        nan_count = result.isna().sum()
        assert nan_count == horizon

    def test_all_pre_tail_rows_are_finite(self):
        close = _make_close(np.linspace(10.0, 100.0, 20))
        result = forward_log_return(close, horizon=1)
        assert result.iloc[:-1].notna().all()

    def test_series_of_length_one_all_nan(self):
        close = _make_close([100.0])
        result = forward_log_return(close, horizon=1)
        assert result.isna().all()


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------

class TestValidation:
    def test_horizon_zero_raises(self):
        close = _make_close([100.0, 110.0])
        with pytest.raises(ValueError, match="horizon must be >= 1"):
            forward_log_return(close, horizon=0)

    def test_horizon_negative_raises(self):
        close = _make_close([100.0, 110.0])
        with pytest.raises(ValueError, match="horizon must be >= 1"):
            forward_log_return(close, horizon=-3)

    def test_non_datetime_index_raises(self):
        s = pd.Series([100.0, 110.0, 120.0], index=[0, 1, 2])
        with pytest.raises(TypeError, match="DatetimeIndex"):
            forward_log_return(s, horizon=1)

    def test_zero_price_raises(self):
        close = _make_close([100.0, 0.0, 110.0])
        with pytest.raises(ValueError, match="non-positive"):
            forward_log_return(close, horizon=1)

    def test_negative_price_raises(self):
        close = _make_close([100.0, -5.0, 110.0])
        with pytest.raises(ValueError, match="non-positive"):
            forward_log_return(close, horizon=1)

    def test_default_horizon_is_1(self):
        close = _make_close([100.0, 110.0, 121.0])
        result = forward_log_return(close)
        assert result.name == "fwd_ret_1d"
        assert np.isclose(result.iloc[0], np.log(110.0 / 100.0))
