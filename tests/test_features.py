"""Phase 3 feature engineering tests.

The MANDATORY test is the shift-and-compare leakage check:
  If removing future rows changes a feature value at a historical date,
  that feature uses look-ahead data — which is a bug, not a quirk.

Test plan
---------
[REQUIRED] test_no_lookahead_price_features
    Shift-and-compare on all price feature functions.
    Drop 1, 5, 10, 21 trailing rows; verify shared-date values are identical.

[REQUIRED] test_no_lookahead_macro_features
    Shift-and-compare on macro feature functions.
    Drop 1, 5, 10 trailing rows; verify shared-date values are identical.

[REQUIRED] test_feature_matrix_aligns_with_target_index
    Feature matrix index == target ticker's trading-day index (1:1).

Additional:
    - Return lag correctness (ret_1d matches manual log-return)
    - Rolling vol / momentum require minimum window (no partial windows)
    - Macro lag enforced (feature at t uses FRED value from t-1, not t)
    - Ratio feature is symmetric log ratio
    - Macro lag < 1 raises ValueError
    - Feature matrix has no duplicate dates
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.price_features import (
    build_price_features,
    distance_from_ma,
    lagged_returns,
    log_ratio,
    log_returns,
    rolling_momentum,
    rolling_volatility,
)
from src.features.macro_features import (
    align_to_trading_index,
    build_macro_features,
    daily_change as macro_daily_change,
    lag_series,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _make_close(n: int = 150, seed: int = 0) -> pd.Series:
    """Synthetic close price series on business-day index."""
    rng = np.random.default_rng(seed)
    prices = 100.0 * np.exp(rng.normal(0, 0.01, n).cumsum())
    idx = pd.bdate_range("2015-01-01", periods=n, freq="B")
    return pd.Series(prices, index=idx, name="Close")


def _make_fred_series(n_calendar: int = 300, seed: int = 1, name: str = "SER") -> pd.Series:
    """Synthetic FRED series: daily observations including weekends (some NaN)."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2015-01-01", periods=n_calendar, freq="D")
    vals = np.cumsum(rng.normal(0, 0.05, n_calendar)) + 2.0
    s = pd.Series(vals, index=idx, name=name)
    # Introduce NaN on weekends (realistic FRED pattern)
    s.iloc[idx.dayofweek >= 5] = np.nan
    return s


def _biz_index(n: int, start: str = "2015-01-01") -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=n, freq="B")


def _all_price_features(close: pd.Series) -> pd.DataFrame:
    """Run every price feature function on a single close series and concat."""
    parts = [
        lagged_returns(close),
        rolling_volatility(close),
        rolling_momentum(close),
        distance_from_ma(close),
    ]
    return pd.concat(parts, axis=1)


def _all_macro_features(
    fred: pd.Series,
    trading_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Run macro features for a single series."""
    return build_macro_features({fred.name: fred}, trading_index)


# --------------------------------------------------------------------------- #
# [REQUIRED] Shift-and-compare leakage tests
# --------------------------------------------------------------------------- #

class TestNoLookaheadPrice:
    """[REQUIRED] Price features must not depend on data after time t."""

    def _check_no_lookahead(self, close_full: pd.Series, n_drop: int) -> None:
        close_trunc = close_full.iloc[:-n_drop]

        feats_full = _all_price_features(close_full)
        feats_trunc = _all_price_features(close_trunc)

        shared = feats_full.index.intersection(feats_trunc.index)
        assert len(shared) > 10, f"Too few shared rows ({len(shared)}) for n_drop={n_drop}"

        pd.testing.assert_frame_equal(
            feats_full.loc[shared].sort_index(),
            feats_trunc.loc[shared].sort_index(),
            check_exact=True,
            obj=f"price features (n_drop={n_drop})",
        )

    def test_drop_1(self):
        self._check_no_lookahead(_make_close(150), 1)

    def test_drop_5(self):
        self._check_no_lookahead(_make_close(150), 5)

    def test_drop_10(self):
        self._check_no_lookahead(_make_close(150), 10)

    def test_drop_21(self):
        self._check_no_lookahead(_make_close(150), 21)

    def test_independent_of_seed(self):
        """Leakage check holds across different random seeds."""
        for seed in [42, 99, 123]:
            self._check_no_lookahead(_make_close(150, seed=seed), n_drop=5)

    def test_log_ratio_no_lookahead(self):
        """Gold/silver ratio must not look ahead."""
        close_a = _make_close(120, seed=0)
        close_b = _make_close(120, seed=1)

        ratio_full = log_ratio(close_a, close_b, "ratio")
        ratio_trunc = log_ratio(close_a.iloc[:-5], close_b.iloc[:-5], "ratio")

        shared = ratio_full.index.intersection(ratio_trunc.index)
        pd.testing.assert_series_equal(
            ratio_full.loc[shared].sort_index(),
            ratio_trunc.loc[shared].sort_index(),
            check_exact=True,
        )


class TestNoLookaheadMacro:
    """[REQUIRED] Macro features must not depend on data after time t."""

    def _check_no_lookahead(self, fred: pd.Series, trading_index: pd.DatetimeIndex, n_drop: int) -> None:
        trading_trunc = trading_index[:-n_drop]

        # Drop the last n_drop trading days from the FRED series too
        # (remove calendar days beyond the last kept trading day)
        last_date = trading_trunc[-1]
        fred_trunc = fred.loc[fred.index <= last_date]

        feats_full = _all_macro_features(fred, trading_index)
        feats_trunc = _all_macro_features(fred_trunc, trading_trunc)

        shared = feats_full.index.intersection(feats_trunc.index)
        assert len(shared) > 5

        pd.testing.assert_frame_equal(
            feats_full.loc[shared].sort_index(),
            feats_trunc.loc[shared].sort_index(),
            check_exact=True,
            obj=f"macro features (n_drop={n_drop})",
        )

    def test_drop_1(self):
        n = 200
        trading = _biz_index(n)
        fred = _make_fred_series(n_calendar=280, name="DFII10")
        self._check_no_lookahead(fred, trading, 1)

    def test_drop_5(self):
        n = 200
        trading = _biz_index(n)
        fred = _make_fred_series(n_calendar=280, name="DFII10")
        self._check_no_lookahead(fred, trading, 5)

    def test_drop_10(self):
        n = 200
        trading = _biz_index(n)
        fred = _make_fred_series(n_calendar=280, name="DFII10")
        self._check_no_lookahead(fred, trading, 10)


# --------------------------------------------------------------------------- #
# [REQUIRED] Feature matrix aligns 1:1 with target index
# --------------------------------------------------------------------------- #

def test_feature_matrix_aligns_with_target_index():
    """[REQUIRED] Feature matrix index == target ticker's trading-day index."""
    close = _make_close(100)
    ref_idx = close.index

    prices = {"GC=F": pd.DataFrame({"Close": close, "Open": close, "High": close, "Low": close, "Volume": 0})}
    feature_df = build_price_features(prices, target_ticker="GC=F")

    assert feature_df.index.equals(ref_idx), (
        f"Feature index ({len(feature_df)}) does not match target index ({len(ref_idx)})"
    )


def test_feature_matrix_no_duplicate_dates():
    """Feature matrix must have no duplicate date indices."""
    close = _make_close(120)
    prices = {"GC=F": pd.DataFrame({"Close": close, "Open": close, "High": close, "Low": close, "Volume": 0})}
    feature_df = build_price_features(prices, target_ticker="GC=F")

    dupes = feature_df.index[feature_df.index.duplicated()]
    assert len(dupes) == 0, f"Duplicate dates: {dupes.tolist()}"


# --------------------------------------------------------------------------- #
# Price feature correctness
# --------------------------------------------------------------------------- #

def test_ret_1d_matches_manual_log_return():
    """ret_1d[t] = log(close[t] / close[t-1]) exactly."""
    close = _make_close(50)
    feats = lagged_returns(close, lags=[1])
    manual = np.log(close / close.shift(1))
    pd.testing.assert_series_equal(feats["ret_1d"], manual, check_names=False)


def test_ret_5d_matches_manual():
    """ret_5d[t] = log(close[t] / close[t-5]) exactly."""
    close = _make_close(50)
    feats = lagged_returns(close, lags=[5])
    manual = np.log(close / close.shift(5))
    pd.testing.assert_series_equal(feats["ret_5d"], manual, check_names=False)


def test_ret_1d_first_row_is_nan():
    """First row of ret_1d is NaN (no prior close)."""
    close = _make_close(30)
    feats = lagged_returns(close, lags=[1])
    assert np.isnan(feats["ret_1d"].iloc[0])


def test_rolling_vol_requires_full_window():
    """Rolling vol with window=5 is NaN for first 4 rows, valid from row 5 on."""
    close = _make_close(30)
    feats = rolling_volatility(close, windows=[5])
    assert feats["vol_5d"].iloc[:4].isna().all()
    assert feats["vol_5d"].iloc[5:].notna().all()


def test_rolling_momentum_requires_full_window():
    """Rolling momentum with window=10 is NaN for first 9 rows."""
    close = _make_close(30)
    feats = rolling_momentum(close, windows=[10])
    assert feats["mom_10d"].iloc[:9].isna().all()
    # row 10 (index 9) should be valid: 10 returns available
    assert pd.notna(feats["mom_10d"].iloc[10])


def test_dist_ma_requires_full_window():
    """Distance from MA-20 is NaN for first 19 rows."""
    close = _make_close(50)
    feats = distance_from_ma(close, windows=[20])
    assert feats["dist_ma_20"].iloc[:19].isna().all()
    assert pd.notna(feats["dist_ma_20"].iloc[20])


def test_log_ratio_symmetric():
    """log_ratio(a, b) = -log_ratio(b, a)."""
    a = _make_close(50, seed=0)
    b = _make_close(50, seed=1)
    r_ab = log_ratio(a, b, "ab")
    r_ba = log_ratio(b, a, "ba")
    pd.testing.assert_series_equal(-r_ab, r_ba, check_names=False)


def test_log_ratio_zero_for_identical_series():
    """log_ratio(a, a) = 0 for all t."""
    a = _make_close(50)
    r = log_ratio(a, a.copy(), "same")
    assert (r.dropna() == 0.0).all()


def test_vol_is_annualised():
    """Rolling vol output is sqrt(252) times the raw daily std."""
    close = _make_close(60)
    r1 = log_returns(close)
    raw_std = r1.rolling(5, min_periods=5).std()
    feats = rolling_volatility(close, windows=[5], annualise=True)
    expected = raw_std * np.sqrt(252)
    pd.testing.assert_series_equal(feats["vol_5d"], expected, check_names=False)


def test_prefix_in_column_names():
    """Prefix is prepended to column names."""
    close = _make_close(50)
    feats = lagged_returns(close, lags=[1, 5], prefix="gc_")
    assert "gc_ret_1d" in feats.columns
    assert "gc_ret_5d" in feats.columns
    assert "ret_1d" not in feats.columns


def test_build_price_features_returns_dataframe():
    """build_price_features returns a DataFrame with correct index."""
    close = _make_close(100)
    prices = {"GC=F": pd.DataFrame({"Close": close, "Open": close, "High": close, "Low": close, "Volume": 0})}
    feats = build_price_features(prices, "GC=F")
    assert isinstance(feats, pd.DataFrame)
    assert len(feats) == len(close)


# --------------------------------------------------------------------------- #
# Macro feature correctness
# --------------------------------------------------------------------------- #

def test_macro_lag_enforced():
    """Feature at trading day t uses FRED value from day t-1 (not t).

    Build a FRED series where each calendar value equals its day-of-year.
    After aligning to a trading-day index and lagging by 1, the feature at
    each trading day t must equal the aligned value from the PREVIOUS
    trading day, not the current one.
    """
    n_biz = 50
    trading = _biz_index(n_biz, start="2015-06-01")

    # FRED series: value = integer index (0, 1, 2, ...)
    cal_idx = pd.date_range("2015-05-01", periods=200, freq="D")
    vals = pd.Series(np.arange(200.0), index=cal_idx, name="TEST")

    # Align to trading days first (no lag)
    aligned = align_to_trading_index(vals, trading)

    # Lag by 1
    lagged = lag_series(aligned, lag=1)

    # At each trading day t, lagged[t] should equal aligned[t-1]
    for i in range(1, len(lagged)):
        if pd.notna(lagged.iloc[i]) and pd.notna(aligned.iloc[i - 1]):
            assert lagged.iloc[i] == aligned.iloc[i - 1], (
                f"At t={i}: lagged={lagged.iloc[i]}, aligned[t-1]={aligned.iloc[i-1]}"
            )


def test_macro_lag_less_than_one_raises():
    """lag_series with lag < 1 must raise ValueError."""
    s = pd.Series([1.0, 2.0, 3.0], index=pd.bdate_range("2020-01-01", periods=3))
    with pytest.raises(ValueError, match="lag must be >= 1"):
        lag_series(s, lag=0)


def test_macro_daily_change_correct():
    """daily_change computes first difference."""
    s = pd.Series([1.0, 3.0, 2.0, 5.0], index=pd.bdate_range("2020-01-01", periods=4))
    chg = macro_daily_change(s)
    expected = pd.Series([np.nan, 2.0, -1.0, 3.0], index=s.index)
    pd.testing.assert_series_equal(chg, expected)


def test_build_macro_features_returns_dataframe():
    """build_macro_features returns a DataFrame aligned to trading_index."""
    n = 100
    trading = _biz_index(n)
    fred = _make_fred_series(n_calendar=150, name="DFII10")
    feats = build_macro_features({"DFII10": fred}, trading)

    assert isinstance(feats, pd.DataFrame)
    assert feats.index.equals(trading)


def test_macro_features_include_expected_columns():
    """build_macro_features produces all expected column names."""
    n = 100
    trading = _biz_index(n)
    macro = {
        "DFII10": _make_fred_series(name="DFII10"),
        "T10YIE": _make_fred_series(name="T10YIE"),
        "DGS10": _make_fred_series(name="DGS10"),
        "DTWEXBGS": _make_fred_series(name="DTWEXBGS"),
        "VIXCLS": _make_fred_series(name="VIXCLS"),
    }
    feats = build_macro_features(macro, trading)

    expected_cols = {
        "dfii10_level", "dfii10_chg_1d",
        "t10yie_level", "t10yie_chg_1d",
        "dgs10_chg_1d",
        "dxy_level", "dxy_chg_1d",
        "vix_level", "vix_chg_1d",
    }
    assert expected_cols.issubset(set(feats.columns)), (
        f"Missing columns: {expected_cols - set(feats.columns)}"
    )


def test_macro_missing_series_skipped_gracefully():
    """build_macro_features proceeds if a series is missing (with a warning)."""
    n = 80
    trading = _biz_index(n)
    # Provide only VIX — other series should be silently skipped
    feats = build_macro_features(
        {"VIXCLS": _make_fred_series(name="VIXCLS", n_calendar=150)},
        trading,
    )
    assert "vix_level" in feats.columns
    assert "dfii10_level" not in feats.columns


def test_align_to_trading_index_ffills_holidays():
    """align_to_trading_index forward-fills over holidays in the trading calendar.

    Scenario: FRED has values Mon–Fri, NaN on weekends. Trading index includes
    a Monday that is NaN in the FRED series (simulated holiday). The aligned
    series should carry the preceding Friday's value for that Monday.
    """
    # Calendar series: Mon–Fri = sequential values, Sat–Sun = NaN
    idx = pd.date_range("2020-01-01", periods=21, freq="D")
    vals = pd.Series(np.arange(21.0), index=idx, name="SEQ")
    vals[idx.dayofweek >= 5] = np.nan

    # Simulate a holiday: make Monday Jan 6 NaN in the FRED source
    holiday = pd.Timestamp("2020-01-06")
    friday_before = pd.Timestamp("2020-01-03")
    vals.loc[holiday] = np.nan  # FRED didn't publish on this day

    # Trading calendar still includes Jan 6 (exchange was open, FRED wasn't)
    trading = pd.bdate_range("2020-01-06", periods=5)
    aligned = align_to_trading_index(vals, trading, max_ffill_days=3)

    # Holiday day should be forward-filled from the preceding Friday
    assert pd.notna(aligned.loc[holiday]), "Holiday should be forward-filled, not NaN"
    assert aligned.loc[holiday] == pytest.approx(vals.loc[friday_before]), (
        f"Expected Friday's value {vals.loc[friday_before]}, got {aligned.loc[holiday]}"
    )

    # Regular Tuesday should have its own value
    tue = pd.Timestamp("2020-01-07")
    assert aligned.loc[tue] == pytest.approx(vals.loc[tue])


# --------------------------------------------------------------------------- #
# [REQUIRED] ETF late-close lag tests
# --------------------------------------------------------------------------- #

def _make_prices_dict(close_target: pd.Series, close_ctx: pd.Series, ctx_key: str) -> dict:
    """Build minimal prices dict for build_price_features."""
    def _df(c):
        return pd.DataFrame({"Close": c, "Open": c, "High": c, "Low": c, "Volume": 0})
    return {"GC=F": _df(close_target), ctx_key: _df(close_ctx)}


class TestETFLateCloseLag:
    """[REQUIRED] ETF context features must use the previous day's return.

    ETFs (GLD, SLV, GDX, SPY, TLT, UUP, ^VIX) close at 4pm ET, ~1.5 hours
    after gold futures settle (~2:30pm ET). The same-day ETF return embeds
    gold price movement that has not yet appeared in the futures close price.
    When passed as late_close_tickers, the 1-day return must be shifted by
    one extra trading day.
    """

    def test_late_close_ticker_gets_extra_shift(self):
        """gld_ret_1d at t must equal log(GLD[t-1] / GLD[t-2]), NOT log(GLD[t] / GLD[t-1])."""
        close_target = _make_close(60, seed=0)
        close_etf = _make_close(60, seed=1)
        prices = _make_prices_dict(close_target, close_etf, "GLD")

        feats = build_price_features(
            prices, "GC=F",
            context_tickers=["GLD"],
            late_close_tickers=["GLD"],
        )

        # Expected: log(close_etf[t-1] / close_etf[t-2]) — one extra day of lag
        r1 = log_returns(close_etf.reindex(close_target.index).ffill())
        expected = r1.shift(1)

        pd.testing.assert_series_equal(
            feats["gld_ret_1d"],
            expected,
            check_names=False,
            obj="gld_ret_1d with late_close_tickers",
        )

    def test_same_close_ticker_has_no_extra_shift(self):
        """Futures context ticker (same close time) must NOT get the extra shift."""
        close_target = _make_close(60, seed=0)
        close_sif = _make_close(60, seed=2)
        prices = _make_prices_dict(close_target, close_sif, "SI=F")

        feats = build_price_features(
            prices, "GC=F",
            context_tickers=["SI=F"],
            late_close_tickers=[],          # SI=F is NOT a late-close ticker
        )

        # Expected: log(close_sif[t] / close_sif[t-1]) — no extra shift
        r1 = log_returns(close_sif.reindex(close_target.index).ffill())

        pd.testing.assert_series_equal(
            feats["sif_ret_1d"],
            r1,
            check_names=False,
            obj="sif_ret_1d without extra lag",
        )

    def test_late_close_shift_and_compare(self):
        """Removing future rows must not alter historical late-close feature values.

        This is the shift-and-compare test for the extra-lag path.
        """
        n = 80
        close_target = _make_close(n, seed=0)
        close_etf = _make_close(n, seed=3)
        prices_full = _make_prices_dict(close_target, close_etf, "GLD")
        prices_trunc = {
            "GC=F": prices_full["GC=F"].iloc[:-5],
            "GLD": prices_full["GLD"].iloc[:-5],
        }

        feats_full = build_price_features(
            prices_full, "GC=F",
            context_tickers=["GLD"],
            late_close_tickers=["GLD"],
        )
        feats_trunc = build_price_features(
            prices_trunc, "GC=F",
            context_tickers=["GLD"],
            late_close_tickers=["GLD"],
        )

        shared = feats_full.index.intersection(feats_trunc.index)
        pd.testing.assert_series_equal(
            feats_full.loc[shared, "gld_ret_1d"],
            feats_trunc.loc[shared, "gld_ret_1d"],
            check_exact=True,
            obj="gld_ret_1d shift-and-compare",
        )

    def test_late_close_feature_differs_from_no_lag(self):
        """Verify that applying the extra lag actually changes the feature values.

        If this test fails, the lag is a no-op — which would mean the artifact
        fix was not applied correctly.
        """
        close_target = _make_close(60, seed=0)
        close_etf = _make_close(60, seed=4)
        prices = _make_prices_dict(close_target, close_etf, "GLD")

        feats_lag = build_price_features(
            prices, "GC=F",
            context_tickers=["GLD"],
            late_close_tickers=["GLD"],
        )
        feats_no_lag = build_price_features(
            prices, "GC=F",
            context_tickers=["GLD"],
            late_close_tickers=[],
        )

        # The two feature columns must differ (not identical)
        assert not feats_lag["gld_ret_1d"].equals(feats_no_lag["gld_ret_1d"]), (
            "late_close_tickers fix must produce different values from the unfixed version"
        )

    def test_mixed_context_some_late_some_not(self):
        """When mixing late-close and same-close tickers, each gets its correct lag."""
        n = 60
        close_target = _make_close(n, seed=0)
        close_etf = _make_close(n, seed=5)
        close_fut = _make_close(n, seed=6)

        def _df(c):
            return pd.DataFrame({"Close": c, "Open": c, "High": c, "Low": c, "Volume": 0})

        prices = {
            "GC=F": _df(close_target),
            "GLD": _df(close_etf),
            "SI=F": _df(close_fut),
        }

        feats = build_price_features(
            prices, "GC=F",
            context_tickers=["GLD", "SI=F"],
            late_close_tickers=["GLD"],
        )

        ref = close_target.index
        r1_etf = log_returns(close_etf.reindex(ref).ffill())
        r1_fut = log_returns(close_fut.reindex(ref).ffill())

        # GLD gets extra shift; SI=F does not
        pd.testing.assert_series_equal(
            feats["gld_ret_1d"], r1_etf.shift(1), check_names=False
        )
        pd.testing.assert_series_equal(
            feats["sif_ret_1d"], r1_fut, check_names=False
        )


def test_no_normalization_in_features():
    """Features must NOT be standardized: mean and std are unconstrained.

    If any feature were normalized (mean=0, std=1), it would mean test-period
    statistics were used — a leakage violation. We verify at least one feature
    has mean != 0 and std != 1.
    """
    close = _make_close(200)
    feats = _all_price_features(close).dropna()

    for col in feats.columns:
        vals = feats[col].dropna()
        if len(vals) < 10:
            continue
        mean_is_zero = abs(vals.mean()) < 1e-12
        std_is_one = abs(vals.std() - 1.0) < 1e-6
        assert not (mean_is_zero and std_is_one), (
            f"Column {col!r} appears to be z-scored (mean≈0, std≈1). "
            "Normalization must NOT happen in feature engineering."
        )
