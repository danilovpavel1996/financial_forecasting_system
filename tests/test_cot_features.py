"""Tests for COT feature engineering.

[REQUIRED] tests (per PHASE_9_PLAN §4d):
  1. Shift-and-compare leakage: removing future rows must not change historical values.
  2. Direct date-comparison: feature at t uses only release_date < t.
  3. mm_net_pct in [0, 1] after warm-up (first 26 weekly obs).
  4. Forward-fill gap test: no gap > 10 trading days in aligned features.
  5. Cross-sectional COT features sum to zero across the basket at every date.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.cot_features import (
    build_cot_features,
    build_cross_cot_features,
    _rolling_percentile,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

_TICKERS = ["GC=F", "SI=F", "CL=F"]

_COT_COLS = ["mm_long", "mm_short", "prod_long", "prod_short", "oi"]


def _make_weekly_cot(
    n_weeks: int = 100,
    seed: int = 42,
    start: str = "2015-01-02",
) -> pd.DataFrame:
    """Synthetic weekly COT DataFrame indexed by release_date (Fridays)."""
    rng = np.random.default_rng(seed)

    # Build Friday release dates
    start_ts = pd.Timestamp(start)
    # Find first Friday on or after start
    days_to_friday = (4 - start_ts.weekday()) % 7
    first_friday = start_ts + pd.Timedelta(days=days_to_friday)
    fridays = pd.bdate_range(first_friday, periods=n_weeks, freq="W-FRI")

    mm_long  = rng.integers(50_000, 200_000, n_weeks).astype(float)
    mm_short = rng.integers(20_000, 100_000, n_weeks).astype(float)
    pm_long  = rng.integers(10_000, 50_000,  n_weeks).astype(float)
    pm_short = rng.integers(50_000, 150_000, n_weeks).astype(float)
    oi       = (mm_long + mm_short + pm_long + pm_short +
                rng.integers(10_000, 50_000, n_weeks)).astype(float)

    df = pd.DataFrame(
        {"mm_long": mm_long, "mm_short": mm_short,
         "prod_long": pm_long, "prod_short": pm_short, "oi": oi},
        index=pd.DatetimeIndex(fridays, name="release_date"),
    )
    return df


def _make_cot_raw(
    tickers: list[str] = _TICKERS,
    n_weeks: int = 100,
) -> dict[str, pd.DataFrame]:
    return {t: _make_weekly_cot(n_weeks=n_weeks, seed=i) for i, t in enumerate(tickers)}


def _make_trading_index(n_days: int = 500, start: str = "2015-01-05") -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=n_days, freq="B")


# ── [REQUIRED] 1. Shift-and-compare leakage test ─────────────────────────────

class TestCotFeaturesNoLookahead:
    """Shift-and-compare: truncating future rows must not change past values."""

    def _check(self, n_drop: int) -> None:
        cot_full = _make_cot_raw(n_weeks=120)
        idx_full = _make_trading_index(n_days=600)

        cot_trunc = {t: df.iloc[:-n_drop] for t, df in cot_full.items()}
        idx_trunc  = idx_full[:-n_drop * 5]   # drop ~n_drop weeks worth of trading days

        feats_full  = build_cot_features(cot_full,  _TICKERS, idx_full)
        feats_trunc = build_cot_features(cot_trunc, _TICKERS, idx_trunc)

        for ticker in _TICKERS:
            shared = feats_full[ticker].index.intersection(feats_trunc[ticker].index)
            assert len(shared) > 100, f"Too few shared rows for {ticker}: {len(shared)}"

            pd.testing.assert_frame_equal(
                feats_full[ticker].loc[shared].sort_index(),
                feats_trunc[ticker].loc[shared].sort_index(),
                check_exact=False,
                atol=1e-10,
                obj=f"cot_features[{ticker}] shift-and-compare (n_drop={n_drop})",
            )

    def test_drop_1_week(self):
        self._check(1)

    def test_drop_5_weeks(self):
        self._check(5)

    def test_drop_10_weeks(self):
        self._check(10)


# ── [REQUIRED] 2. Direct date-comparison: feature at t uses release_date < t ─

class TestCotReleaseDate:
    """The feature value at date t must correspond to a release_date strictly < t."""

    def test_feature_uses_only_past_release_dates(self) -> None:
        """After lag=1, the feature at Monday must correspond to the prior Friday's release."""
        n_weeks = 80
        cot_raw = _make_cot_raw(n_weeks=n_weeks)
        # Use a trading index that starts well after the COT data starts so warm-up is done
        idx = _make_trading_index(n_days=400, start="2016-01-04")
        feats = build_cot_features(cot_raw, _TICKERS, idx)

        ticker = _TICKERS[0]
        raw = cot_raw[ticker].sort_index()   # indexed by release_date (Fridays)

        daily = feats[ticker]["mm_net"]
        daily_nonnull = daily.dropna()

        # For each trading day t with a non-NaN feature, find which weekly obs it used:
        # the most recent release_date strictly < t.
        # Then verify the mm_net value matches.
        mm_net_weekly = raw["mm_long"] - raw["mm_short"]

        violations = 0
        checked = 0
        for t in daily_nonnull.index[:50]:   # check first 50 non-NaN days
            # Most recent release_date strictly before t
            eligible = mm_net_weekly.index[mm_net_weekly.index < t]
            if eligible.empty:
                continue
            last_release = eligible[-1]
            expected = mm_net_weekly.loc[last_release]
            actual = daily_nonnull.loc[t]
            if not np.isclose(actual, expected, atol=1e-6):
                violations += 1
            checked += 1

        assert checked > 0, "No dates checked — test is vacuous"
        assert violations == 0, (
            f"{violations}/{checked} dates had feature values NOT matching "
            "the expected (most recent release_date < t) weekly observation."
        )

    def test_friday_does_not_use_same_day_release(self) -> None:
        """On a Friday, the feature must use the PREVIOUS Friday's release, not today's.

        Data layout (mm_long increments by 1000 each week):
          release_date 2016-01-08 (Fri): mm_long=100000, mm_net=60000
          release_date 2016-01-15 (Fri): mm_long=101000, mm_net=61000

        Expected feature values after align + lag=1:
          2016-01-15 (Fri): 60000  — lag shifts today's 61000 to Mon 2016-01-18;
                                       Fri sees Thu's value which is 60000 (2016-01-08 release)
          2016-01-18 (Mon): 61000  — first day the 2016-01-15 release becomes visible
        """
        fridays = pd.bdate_range("2016-01-08", periods=10, freq="W-FRI")
        mm_long  = np.arange(100_000, 110_000, 1_000, dtype=float)
        mm_short = np.full(10, 40_000.0)
        raw = pd.DataFrame(
            {"mm_long": mm_long, "mm_short": mm_short,
             "prod_long": np.full(10, 10_000.0),
             "prod_short": np.full(10, 20_000.0),
             "oi": np.full(10, 200_000.0)},
            index=pd.DatetimeIndex(fridays, name="release_date"),
        )
        cot_raw = {"GC=F": raw}
        idx = pd.bdate_range("2016-01-11", periods=60, freq="B")
        feats = build_cot_features(cot_raw, ["GC=F"], idx)

        # Friday 2016-01-15: feature must show 60000 (2016-01-08 release), NOT 61000 (today's)
        test_friday = pd.Timestamp("2016-01-15")
        test_monday = pd.Timestamp("2016-01-18")
        if test_friday not in feats["GC=F"].index or test_monday not in feats["GC=F"].index:
            pytest.skip("Test dates not in index")

        val_on_friday = feats["GC=F"].loc[test_friday, "mm_net"]
        val_on_monday = feats["GC=F"].loc[test_monday, "mm_net"]

        # Friday must see the PRIOR Friday's release (60000), not today's (61000)
        assert np.isclose(val_on_friday, 60_000.0, atol=1.0), (
            f"Friday feature should use prior Friday's release (60000), got {val_on_friday}"
        )
        # Monday is the first day where the 2016-01-15 release is visible
        assert np.isclose(val_on_monday, 61_000.0, atol=1.0), (
            f"Monday feature should use 2016-01-15 release (61000), got {val_on_monday}"
        )


# ── [REQUIRED] 3. mm_net_pct in [0, 1] after warm-up ────────────────────────

class TestMmNetPct:
    def test_pct_in_unit_interval_after_warmup(self) -> None:
        """mm_net_pct must be in [0, 1] for all non-NaN values after warm-up."""
        cot_raw = _make_cot_raw(n_weeks=200)
        idx = _make_trading_index(n_days=1000)
        feats = build_cot_features(cot_raw, _TICKERS, idx)

        for ticker in _TICKERS:
            col = feats[ticker]["mm_net_pct"].dropna()
            assert len(col) > 0, f"mm_net_pct is all-NaN for {ticker}"
            out_of_bounds = ((col < -1e-9) | (col > 1 + 1e-9)).sum()
            assert out_of_bounds == 0, (
                f"{ticker} mm_net_pct has {out_of_bounds} values outside [0, 1]"
            )

    def test_pct_nan_during_warmup(self) -> None:
        """mm_net_pct should be NaN for the first <26 weekly observations (warm-up)."""
        n_weeks = 30   # just past the 26-week minimum
        cot_raw = {"GC=F": _make_weekly_cot(n_weeks=n_weeks)}
        idx = _make_trading_index(n_days=n_weeks * 7)
        feats = build_cot_features(cot_raw, ["GC=F"], idx)
        pct = feats["GC=F"]["mm_net_pct"]
        # The first aligned date should be NaN (warm-up)
        assert pct.dropna().empty or pct.iloc[0] != pct.iloc[0], (
            "Expected NaN at the start of mm_net_pct during warm-up"
        )

    def test_rolling_percentile_bounds(self) -> None:
        """_rolling_percentile helper returns values in [0, 1] for non-NaN output."""
        rng = np.random.default_rng(0)
        s = pd.Series(rng.standard_normal(200))
        pct = _rolling_percentile(s)
        valid = pct.dropna()
        assert len(valid) > 0
        assert (valid >= 0).all() and (valid <= 1).all()


# ── [REQUIRED] 4. Forward-fill gap test ──────────────────────────────────────

class TestForwardFillGap:
    """No gap > 10 consecutive NaN trading days in aligned COT features.

    Gaps > 10 days indicate either missing data or a misconfigured ffill limit.
    """

    def test_no_gap_exceeds_10_days(self) -> None:
        cot_raw = _make_cot_raw(n_weeks=150)
        idx = _make_trading_index(n_days=750)
        feats = build_cot_features(cot_raw, _TICKERS, idx)

        for ticker in _TICKERS:
            mm_net = feats[ticker]["mm_net"]
            # Find runs of consecutive NaN after the series has had at least one valid value
            first_valid = mm_net.first_valid_index()
            if first_valid is None:
                continue
            post_warmup = mm_net.loc[first_valid:]
            # Count max consecutive NaN after first valid
            is_nan = post_warmup.isna()
            max_run = 0
            current_run = 0
            for v in is_nan:
                if v:
                    current_run += 1
                    max_run = max(max_run, current_run)
                else:
                    current_run = 0
            assert max_run <= 10, (
                f"{ticker} mm_net has a gap of {max_run} consecutive NaN trading days "
                "(exceeds max_ffill_days=10)"
            )


# ── [REQUIRED] 5. Cross-sectional COT features sum to zero ───────────────────

class TestCrossCotFeaturesZeroSum:
    """rel_mm_net_pct and rel_mm_net_chg must sum to zero across the basket at each date."""

    def test_rel_features_sum_to_zero(self) -> None:
        cot_raw = _make_cot_raw(n_weeks=150)
        idx = _make_trading_index(n_days=600)
        cot_feats      = build_cot_features(cot_raw, _TICKERS, idx)
        cross_cot_feats = build_cross_cot_features(cot_feats, _TICKERS, idx)

        for feat in ("rel_mm_net_pct", "rel_mm_net_chg"):
            cols = pd.concat(
                [cross_cot_feats[t][feat].rename(t) for t in _TICKERS],
                axis=1,
            )
            row_sums = cols.sum(axis=1)
            # Only check rows where ALL assets have non-NaN values
            valid_mask = cols.notna().all(axis=1)
            sums_valid = row_sums[valid_mask]
            if len(sums_valid) == 0:
                continue
            max_abs = sums_valid.abs().max()
            assert max_abs < 1e-6, (
                f"{feat}: max cross-basket sum = {max_abs:.2e} (should be ~0)"
            )

    def test_rel_features_shape_matches_index(self) -> None:
        cot_raw = _make_cot_raw(n_weeks=100)
        idx = _make_trading_index(n_days=500)
        cot_feats      = build_cot_features(cot_raw, _TICKERS, idx)
        cross_cot_feats = build_cross_cot_features(cot_feats, _TICKERS, idx)

        for ticker in _TICKERS:
            assert len(cross_cot_feats[ticker]) == len(idx), (
                f"{ticker} cross COT length mismatch"
            )
            assert list(cross_cot_feats[ticker].columns) == ["rel_mm_net_pct", "rel_mm_net_chg"]


# ── Additional: column names and dtypes ──────────────────────────────────────

class TestCotFeatureSchema:
    def test_per_asset_columns_present(self) -> None:
        cot_raw = _make_cot_raw()
        idx = _make_trading_index()
        feats = build_cot_features(cot_raw, _TICKERS, idx)

        expected = {"mm_net", "mm_net_chg", "mm_net_pct", "mm_long_ratio", "prod_net", "oi_chg_pct"}
        for ticker in _TICKERS:
            assert set(feats[ticker].columns) == expected, (
                f"{ticker} columns: got {set(feats[ticker].columns)}"
            )

    def test_all_float_dtype(self) -> None:
        cot_raw = _make_cot_raw()
        idx = _make_trading_index()
        feats = build_cot_features(cot_raw, _TICKERS, idx)

        for ticker in _TICKERS:
            for col in feats[ticker].columns:
                assert feats[ticker][col].dtype == np.float64, (
                    f"{ticker}.{col} dtype={feats[ticker][col].dtype}, expected float64"
                )

    def test_index_matches_trading_index(self) -> None:
        cot_raw = _make_cot_raw()
        idx = _make_trading_index()
        feats = build_cot_features(cot_raw, _TICKERS, idx)

        for ticker in _TICKERS:
            pd.testing.assert_index_equal(feats[ticker].index, idx)

    def test_missing_ticker_returns_nan_block(self) -> None:
        cot_raw = _make_cot_raw(tickers=["GC=F"])
        idx = _make_trading_index()
        feats = build_cot_features(cot_raw, ["GC=F", "SI=F"], idx)

        assert "SI=F" in feats
        assert feats["SI=F"].isnull().all().all(), "Missing ticker should yield all-NaN block"
