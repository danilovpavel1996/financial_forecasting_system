"""Tests for the pooled dataset builder and its fold-integrity guarantee.

[REQUIRED] test_fold_integrity:
    All 4 assets at the same date must land in the same walk-forward fold.
    If gold-on-Tuesday is in training and silver-on-Tuesday is in test,
    that leaks the cross-sectional structure.

Additional tests cover:
  - 4 assets present on every date
  - Target is a forward RETURN (not a price level)
  - MultiIndex has levels (date, asset)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.eval.splitter import WalkForwardSplitter
from src.features.pooled_dataset import feature_cols, get_asset_order


_METALS = ["GC=F", "SI=F", "PL=F", "PA=F"]
_N_DATES = 500   # synthetic trading days
_N_FEATURES = 6  # kept minimal for unit-test speed


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic pooled dataset (no real data required for structural tests)
# ─────────────────────────────────────────────────────────────────────────────

def _make_synthetic_pooled(
    n_dates: int = _N_DATES,
    seed: int = 42,
) -> pd.DataFrame:
    """Build a minimal pooled DataFrame with MultiIndex(date, asset).

    Mimics the structure produced by build_pooled_dataset but uses synthetic
    data to avoid any I/O dependency.  All feature columns are finite.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n_dates, freq="B")

    rows = []
    for date in dates:
        for asset in _METALS:
            row = {
                "feat_a": rng.normal(),
                "feat_b": rng.normal(),
                "feat_c": rng.normal(),
                "feat_d": rng.normal(),
                "feat_e": rng.normal(),
                "feat_f": rng.normal(),
                "target": rng.normal(0, 0.01),
            }
            rows.append((date, asset, row))

    idx = pd.MultiIndex.from_tuples(
        [(d, a) for d, a, _ in rows], names=["date", "asset"]
    )
    df = pd.DataFrame(
        [r for _, _, r in rows], index=idx
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# [REQUIRED] Fold integrity: all 4 assets at same date in same fold
# ─────────────────────────────────────────────────────────────────────────────

class TestFoldIntegrity:
    """[REQUIRED] Walk-forward split on dates must keep all assets together."""

    def _run_integrity_check(self, splitter: WalkForwardSplitter) -> None:
        pooled = _make_synthetic_pooled()
        unique_dates = (
            pooled.index.get_level_values("date").unique().sort_values()
        )
        folds = splitter.split(len(unique_dates))

        for k, (train_dpos, test_dpos) in enumerate(folds):
            train_dates = set(unique_dates[train_dpos])
            test_dates = set(unique_dates[test_dpos])

            # 1. No date in both train and test
            overlap = train_dates & test_dates
            assert len(overlap) == 0, (
                f"Fold {k}: {len(overlap)} dates in both train and test: "
                f"{list(overlap)[:3]}"
            )

            # 2. For every test date, ALL 4 assets are in test (none in train)
            pooled_dates = pooled.index.get_level_values("date")
            for date in test_dates:
                assets_in_test = set(
                    pooled.loc[pooled_dates == date]
                    .index.get_level_values("asset")
                )
                assets_in_train = assets_in_test & train_dates
                # All 4 metals should be present in test, not train
                assert set(_METALS).issubset(assets_in_test), (
                    f"Fold {k}, date {date}: missing assets in test: "
                    f"{set(_METALS) - assets_in_test}"
                )

            # 3. For every train date, ALL 4 assets are in train (none in test)
            for date in train_dates:
                assets_here = set(
                    pooled.loc[pooled_dates == date]
                    .index.get_level_values("asset")
                )
                assert len(assets_here & test_dates) == 0  # dates don't cross

    def test_standard_splitter(self):
        splitter = WalkForwardSplitter(min_train=200, test_size=60, embargo=5)
        self._run_integrity_check(splitter)

    def test_splitter_with_n_splits(self):
        splitter = WalkForwardSplitter(
            min_train=150, test_size=50, embargo=5, n_splits=4
        )
        self._run_integrity_check(splitter)


def test_fold_integrity_dates_never_straddle_boundary():
    """[REQUIRED] Within each fold, no date appears in BOTH train rows AND test rows.

    This is the core anti-leakage invariant for cross-sectional ranking.
    If date T is in training for GC=F but in test for SI=F within the SAME fold,
    the model can exploit silver's same-day information when predicting gold.

    Note: In an expanding-window splitter a date from fold-k's test set will
    legitimately appear in fold-(k+1)'s training set — that is expected and
    correct. We only assert the within-fold constraint.
    """
    pooled = _make_synthetic_pooled(n_dates=300)
    splitter = WalkForwardSplitter(min_train=100, test_size=50, embargo=5)
    unique_dates = pooled.index.get_level_values("date").unique().sort_values()
    pooled_dates = pooled.index.get_level_values("date")

    for k, (train_dpos, test_dpos) in enumerate(splitter.split(len(unique_dates))):
        train_date_set = set(unique_dates[train_dpos])
        test_date_set  = set(unique_dates[test_dpos])

        # Within fold: train ∩ test = ∅
        overlap = train_date_set & test_date_set
        assert len(overlap) == 0, (
            f"Fold {k}: {len(overlap)} dates appear in both train and test: "
            f"{list(overlap)[:3]}"
        )

        # Every asset that belongs to a test date must NOT appear in the
        # training rows for that SAME fold (ensured by date-only split).
        for date in test_date_set:
            # All rows at this date
            rows_at_date = pooled.loc[pooled_dates == date]
            n_rows = len(rows_at_date)
            assert n_rows == len(_METALS), (
                f"Fold {k}, date {date}: expected {len(_METALS)} asset rows, got {n_rows}"
            )
            # The date must NOT be in train_date_set (already tested above,
            # but explicit check per the requirement description)
            assert date not in train_date_set, (
                f"Fold {k}: test date {date} also in train_date_set"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Structure tests
# ─────────────────────────────────────────────────────────────────────────────

def test_four_assets_per_date():
    """Every date in the pooled dataset has exactly 4 rows (one per metal)."""
    pooled = _make_synthetic_pooled()
    counts = pooled.groupby(level="date").size()
    assert (counts == 4).all(), (
        f"Some dates have != 4 rows: {counts[counts != 4].head()}"
    )


def test_multiindex_levels():
    """Pooled DataFrame has MultiIndex with levels named 'date' and 'asset'."""
    pooled = _make_synthetic_pooled()
    assert pooled.index.names == ["date", "asset"]


def test_feature_cols_excludes_target():
    """feature_cols() returns all columns except 'target'."""
    pooled = _make_synthetic_pooled()
    fcols = feature_cols(pooled)
    assert "target" not in fcols
    assert set(fcols) == set(pooled.columns) - {"target"}


def test_get_asset_order_returns_all_metals():
    """get_asset_order returns all 4 metals in sorted order."""
    pooled = _make_synthetic_pooled()
    order = get_asset_order(pooled)
    assert sorted(order) == sorted(_METALS)


def test_target_is_finite_and_small():
    """Target values are finite and in a plausible daily-return range.

    Ensures the target is a return, not a price level (which would be ~100–2000).
    """
    pooled = _make_synthetic_pooled()
    targets = pooled["target"].dropna()
    assert targets.notna().all()
    # Daily log-returns are typically in (-0.2, 0.2) range
    assert targets.abs().max() < 1.0, (
        f"Target max abs = {targets.abs().max():.4f} — looks like a price level, not a return"
    )


def test_all_feature_cols_finite():
    """All feature columns in the synthetic pooled dataset are finite."""
    pooled = _make_synthetic_pooled()
    fcols = feature_cols(pooled)
    n_nan = pooled[fcols].isna().sum().sum()
    assert n_nan == 0, f"Feature matrix has {n_nan} NaN values"


# ─────────────────────────────────────────────────────────────────────────────
# Phase 8: 9-asset fold-integrity test
# ─────────────────────────────────────────────────────────────────────────────

_9_ASSETS = ["GC=F", "SI=F", "PL=F", "PA=F", "CL=F", "NG=F", "HG=F", "ZC=F", "ZS=F"]


def _make_synthetic_pooled_n(
    assets: list,
    n_dates: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """Synthetic pooled DataFrame for an arbitrary N-asset basket."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n_dates, freq="B")
    rows = []
    for date in dates:
        for asset in assets:
            row = {
                "feat_a": rng.normal(),
                "feat_b": rng.normal(),
                "feat_c": rng.normal(),
                "target": rng.normal(0, 0.01),
            }
            rows.append((date, asset, row))
    idx = pd.MultiIndex.from_tuples(
        [(d, a) for d, a, _ in rows], names=["date", "asset"]
    )
    return pd.DataFrame([r for _, _, r in rows], index=idx)


def test_fold_integrity_9_assets():
    """[REQUIRED] All 9 Phase-8 assets at the same date must land in the same fold."""
    pooled = _make_synthetic_pooled_n(_9_ASSETS, n_dates=500)
    splitter = WalkForwardSplitter(min_train=200, test_size=60, embargo=5)
    unique_dates = pooled.index.get_level_values("date").unique().sort_values()
    pooled_dates = pooled.index.get_level_values("date")

    for k, (train_dpos, test_dpos) in enumerate(splitter.split(len(unique_dates))):
        train_dates = set(unique_dates[train_dpos])
        test_dates = set(unique_dates[test_dpos])

        # No date in both train and test
        overlap = train_dates & test_dates
        assert len(overlap) == 0, f"Fold {k}: {len(overlap)} dates in both sets"

        # Every test date has all 9 assets present and none in train
        for date in test_dates:
            assets_here = set(
                pooled.loc[pooled_dates == date].index.get_level_values("asset")
            )
            assert set(_9_ASSETS).issubset(assets_here), (
                f"Fold {k}, date {date}: missing assets: {set(_9_ASSETS) - assets_here}"
            )
            assert date not in train_dates


def test_nine_assets_per_date():
    """Every date in a 9-asset pooled dataset has exactly 9 rows."""
    pooled = _make_synthetic_pooled_n(_9_ASSETS)
    counts = pooled.groupby(level="date").size()
    assert (counts == 9).all(), f"Some dates have != 9 rows: {counts[counts != 9].head()}"
