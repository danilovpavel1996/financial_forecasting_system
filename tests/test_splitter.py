"""CRITICAL leakage tests for WalkForwardSplitter.

These tests are the backbone of the whole system. If any of them fail,
ALL downstream backtest results are invalid and must be treated as such.

Four required Phase-2 tests (marked with # [REQUIRED]):
  1. For every fold: max(train_idx) < min(test_idx) - embargo
  2. With embargo >= horizon, no training label's look-ahead window
     overlaps any test index
  3. Train always precedes test with no overlap in indices
  4. n_splits parameter caps the number of returned folds

Additional tests cover edge cases, rolling window mode, and from_config().
"""
from __future__ import annotations

import numpy as np
import pytest

from src.eval.splitter import WalkForwardSplitter, TRADING_DAYS_PER_YEAR
from src.config import load_config


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def make_splitter(
    min_train: int = 100,
    test_size: int = 50,
    embargo: int = 5,
    n_splits: int | None = None,
    expanding: bool = True,
    train_window: int | None = None,
) -> WalkForwardSplitter:
    return WalkForwardSplitter(
        min_train=min_train,
        test_size=test_size,
        embargo=embargo,
        n_splits=n_splits,
        expanding=expanding,
        train_window=train_window,
    )


# --------------------------------------------------------------------------- #
# [REQUIRED] Test 1 — embargo gap invariant
# --------------------------------------------------------------------------- #

def test_embargo_gap_every_fold():
    """[REQUIRED] For every fold: max(train_idx) < min(test_idx) - embargo."""
    splitter = make_splitter(min_train=200, test_size=50, embargo=5)
    folds = splitter.split(n=500)
    assert len(folds) > 0, "splitter produced no folds"

    for k, (train_idx, test_idx) in enumerate(folds):
        max_train = int(train_idx.max())
        min_test = int(test_idx.min())
        embargo = splitter.embargo
        assert max_train < min_test - embargo, (
            f"Fold {k}: embargo gap violated — "
            f"max(train)={max_train}, min(test)={min_test}, embargo={embargo}. "
            f"Expected max_train < min_test - embargo ({min_test - embargo})."
        )


def test_embargo_gap_multiple_embargo_values():
    """Embargo invariant holds for various embargo sizes."""
    n = 600
    for embargo in [1, 3, 5, 10, 21]:
        splitter = make_splitter(min_train=150, test_size=50, embargo=embargo)
        for k, (train_idx, test_idx) in enumerate(splitter.split(n)):
            assert train_idx.max() < test_idx.min() - embargo, (
                f"embargo={embargo}, fold {k}: invariant broken"
            )


def test_embargo_gap_exact_spacing():
    """Verify the exact relationship: max(train_idx) == min(test_idx) - embargo - 1."""
    splitter = make_splitter(min_train=100, test_size=50, embargo=5)
    for train_idx, test_idx in splitter.split(n=300):
        # The splitter sets train_end = test_start - embargo (exclusive)
        # so max(train_idx) = train_end - 1 = test_start - embargo - 1
        expected_max_train = test_idx.min() - splitter.embargo - 1
        assert int(train_idx.max()) == expected_max_train


# --------------------------------------------------------------------------- #
# [REQUIRED] Test 2 — label look-ahead window does not overlap test
# --------------------------------------------------------------------------- #

def test_label_lookahead_does_not_overlap_test():
    """[REQUIRED] With embargo >= horizon, no training label's look-ahead
    window overlaps any test index.

    A forward-return label at training index t looks ahead to t+1 .. t+horizon.
    We verify: max(train_idx) + horizon < min(test_idx) for all folds.
    """
    horizon = 5
    embargo = horizon  # minimum valid setting

    splitter = make_splitter(min_train=200, test_size=60, embargo=embargo)
    folds = splitter.split(n=600)
    assert len(folds) > 0

    for k, (train_idx, test_idx) in enumerate(folds):
        last_train = int(train_idx.max())
        first_test = int(test_idx.min())
        # The look-ahead window of the last label ends at last_train + horizon.
        # It must not reach the first test index.
        assert last_train + horizon < first_test, (
            f"Fold {k}: label look-ahead leaks into test — "
            f"last_train={last_train}, last_train+horizon={last_train + horizon}, "
            f"first_test={first_test}"
        )


def test_label_lookahead_all_training_indices():
    """Every training label's look-ahead window stays out of the test set."""
    horizon = 3
    embargo = horizon

    splitter = make_splitter(min_train=100, test_size=30, embargo=embargo)
    n = 250

    for train_idx, test_idx in splitter.split(n):
        test_set = set(test_idx.tolist())
        for t in train_idx:
            for h in range(1, horizon + 1):
                assert t + h not in test_set, (
                    f"Training label at {t} looks ahead to {t + h} which is in test set"
                )


def test_insufficient_embargo_would_leak():
    """Demonstrate that embargo < horizon causes look-ahead overlap.

    This is the negative test — it confirms our guard is actually necessary.
    With embargo=0 and horizon=1, the last training label looks at t+1
    which IS the first test index.
    """
    horizon = 1
    bad_embargo = 0  # intentionally too small

    splitter = make_splitter(min_train=100, test_size=50, embargo=bad_embargo)
    for train_idx, test_idx in splitter.split(n=300):
        last_train = int(train_idx.max())
        first_test = int(test_idx.min())
        # With embargo=0: max_train = test_start - 1 = first_test - 1
        # look-ahead end = first_test - 1 + horizon = first_test
        # So look-ahead end == first_test: LEAK
        lookahead_end = last_train + horizon
        assert lookahead_end >= first_test, (
            "Expected look-ahead leak with embargo=0 but didn't find one"
        )
        break  # one fold is enough to demonstrate


# --------------------------------------------------------------------------- #
# [REQUIRED] Test 3 — train always precedes test with no index overlap
# --------------------------------------------------------------------------- #

def test_train_precedes_test_no_overlap():
    """[REQUIRED] Train always precedes test; indices never overlap."""
    splitter = make_splitter(min_train=100, test_size=50, embargo=5)

    for k, (train_idx, test_idx) in enumerate(splitter.split(n=500)):
        # No shared indices
        overlap = set(train_idx.tolist()) & set(test_idx.tolist())
        assert not overlap, f"Fold {k}: train and test share indices {overlap}"

        # Train strictly before test (all train indices < all test indices)
        assert train_idx.max() < test_idx.min(), (
            f"Fold {k}: max(train)={train_idx.max()} >= min(test)={test_idx.min()}"
        )


def test_consecutive_folds_no_test_overlap():
    """Test periods across folds do not overlap."""
    splitter = make_splitter(min_train=100, test_size=50, embargo=5)
    folds = splitter.split(n=600)

    for i in range(len(folds) - 1):
        _, te_i = folds[i]
        _, te_next = folds[i + 1]
        overlap = set(te_i.tolist()) & set(te_next.tolist())
        assert not overlap, (
            f"Consecutive test folds {i} and {i+1} overlap: {overlap}"
        )
        # Folds are in chronological order
        assert te_i.max() < te_next.min(), (
            f"Fold {i} test ends after fold {i+1} test starts"
        )


def test_indices_in_bounds():
    """All indices are within [0, n)."""
    n = 400
    splitter = make_splitter(min_train=100, test_size=50, embargo=5)
    for train_idx, test_idx in splitter.split(n):
        assert train_idx.min() >= 0
        assert test_idx.max() < n


# --------------------------------------------------------------------------- #
# [REQUIRED] Test 4 — n_splits caps fold count (last N folds)
# --------------------------------------------------------------------------- #

def test_n_splits_caps_folds():
    """[REQUIRED] n_splits parameter returns at most n_splits folds."""
    n = 800
    for n_splits in [1, 2, 4, 6]:
        splitter = make_splitter(min_train=100, test_size=50, embargo=5, n_splits=n_splits)
        folds = splitter.split(n)
        assert len(folds) <= n_splits, (
            f"n_splits={n_splits} but got {len(folds)} folds"
        )


def test_n_splits_returns_last_folds():
    """n_splits selects the LAST folds (most recent test periods)."""
    n = 800
    splitter_all = make_splitter(min_train=100, test_size=50, embargo=5)
    splitter_last4 = make_splitter(min_train=100, test_size=50, embargo=5, n_splits=4)

    all_folds = splitter_all.split(n)
    last4_folds = splitter_last4.split(n)

    assert len(last4_folds) == 4

    # The last 4 folds from splitter_all must match splitter_last4's output
    expected = all_folds[-4:]
    for (tr_exp, te_exp), (tr_got, te_got) in zip(expected, last4_folds):
        np.testing.assert_array_equal(tr_exp, tr_got)
        np.testing.assert_array_equal(te_exp, te_got)


def test_n_splits_last_folds_still_satisfy_embargo():
    """Embargo invariant holds even after n_splits filtering."""
    splitter = make_splitter(min_train=100, test_size=50, embargo=7, n_splits=3)
    for train_idx, test_idx in splitter.split(n=600):
        assert train_idx.max() < test_idx.min() - splitter.embargo


# --------------------------------------------------------------------------- #
# Rolling window mode
# --------------------------------------------------------------------------- #

def test_rolling_window_train_size_bounded():
    """Rolling window mode: training set size never exceeds train_window."""
    train_window = 150
    splitter = make_splitter(
        min_train=train_window, test_size=50, embargo=5,
        expanding=False, train_window=train_window,
    )
    for train_idx, _ in splitter.split(n=500):
        assert len(train_idx) <= train_window, (
            f"Rolling window exceeded: got {len(train_idx)}, max {train_window}"
        )


def test_rolling_window_embargo_holds():
    """Embargo invariant holds in rolling window mode."""
    splitter = make_splitter(
        min_train=150, test_size=50, embargo=5,
        expanding=False, train_window=150,
    )
    for train_idx, test_idx in splitter.split(n=500):
        assert train_idx.max() < test_idx.min() - splitter.embargo


# --------------------------------------------------------------------------- #
# Expanding window properties
# --------------------------------------------------------------------------- #

def test_expanding_train_size_grows():
    """In expanding mode, each fold has a larger training set than the previous."""
    splitter = make_splitter(min_train=100, test_size=50, embargo=5, expanding=True)
    folds = splitter.split(n=500)
    for i in range(1, len(folds)):
        assert len(folds[i][0]) > len(folds[i - 1][0]), (
            f"Expanding window should grow: fold {i-1} train={len(folds[i-1][0])}, "
            f"fold {i} train={len(folds[i][0])}"
        )


def test_expanding_train_always_starts_at_zero():
    """In expanding mode, every fold's training set starts at index 0."""
    splitter = make_splitter(min_train=100, test_size=50, embargo=5, expanding=True)
    for train_idx, _ in splitter.split(n=400):
        assert int(train_idx.min()) == 0


# --------------------------------------------------------------------------- #
# Edge cases
# --------------------------------------------------------------------------- #

def test_no_folds_when_data_too_small():
    """If the dataset is smaller than min_train + embargo + test_size, no folds."""
    splitter = make_splitter(min_train=200, test_size=100, embargo=10)
    folds = splitter.split(n=200)  # not enough for even one test fold
    assert len(folds) == 0


def test_invalid_embargo_raises():
    with pytest.raises(ValueError, match="embargo"):
        WalkForwardSplitter(min_train=100, test_size=50, embargo=-1)


def test_rolling_without_window_raises():
    with pytest.raises(ValueError, match="train_window"):
        WalkForwardSplitter(min_train=100, test_size=50, expanding=False)


def test_n_folds_matches_split_length():
    splitter = make_splitter(min_train=100, test_size=50, embargo=5)
    n = 500
    assert splitter.n_folds(n) == len(splitter.split(n))


# --------------------------------------------------------------------------- #
# from_config integration
# --------------------------------------------------------------------------- #

def test_from_config_embargo_gte_horizon():
    """Splitter built from config satisfies embargo >= horizon (validated at load)."""
    cfg = load_config()
    splitter = WalkForwardSplitter.from_config(cfg)
    assert splitter.embargo >= cfg.horizon, (
        f"Splitter embargo ({splitter.embargo}) < config horizon ({cfg.horizon})"
    )


def test_from_config_produces_folds():
    """Splitter built from config produces folds on a realistic dataset size."""
    cfg = load_config()
    splitter = WalkForwardSplitter.from_config(cfg)
    # Simulate ~15 years of daily data
    n = 15 * TRADING_DAYS_PER_YEAR
    folds = splitter.split(n)
    assert len(folds) > 0, "Expected at least one fold for 15 years of data"
    assert len(folds) <= cfg.splitter.n_splits


def test_from_config_embargo_invariant_on_realistic_data():
    """Embargo invariant holds on a realistic dataset using the project config."""
    cfg = load_config()
    splitter = WalkForwardSplitter.from_config(cfg)
    n = 15 * TRADING_DAYS_PER_YEAR
    for train_idx, test_idx in splitter.split(n):
        assert train_idx.max() < test_idx.min() - splitter.embargo
