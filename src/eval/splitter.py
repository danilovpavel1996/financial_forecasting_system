"""Walk-forward splitter with embargo gap (anti-leakage).

Core invariant (enforced and tested):

    For every fold:  max(train_idx) + 1 + embargo <= min(test_idx)
    Equivalently:    max(train_idx) < min(test_idx) - embargo

Why the embargo matters
-----------------------
A forward-return label at time t uses prices at [t+1 .. t+horizon]. If we
train right up to day T and test starting day T+1, the label at training day
T "looks ahead" into the test set. The embargo drops a gap of at least
`horizon` samples between the end of training and the start of testing,
ensuring no training label's look-ahead window overlaps the test indices.

    With embargo >= horizon:
        max(train_idx) + horizon < min(test_idx)   ✓  (proven below)

    Proof: max(train_idx) = test_start - embargo - 1
           max(train_idx) + horizon = test_start - embargo - 1 + horizon
           For this to be < test_start:  horizon - embargo - 1 < 0
           i.e. horizon <= embargo   ✓

n_splits behaviour
------------------
If n_splits is set, only the LAST n_splits folds are returned. This ensures
we always use the most recent data — the earliest folds (with the least
training history) are dropped in favour of the later, better-trained ones.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterator

import numpy as np

if TYPE_CHECKING:
    from src.config import Config

TRADING_DAYS_PER_YEAR: int = 252


@dataclass
class WalkForwardSplitter:
    """Expanding-window walk-forward splits with an embargo gap.

    Parameters
    ----------
    min_train:  minimum number of training samples before the first test fold.
    test_size:  number of samples in each test fold.
    embargo:    gap (in samples) between train-end and test-start.
                Must be >= the forecast horizon to prevent leakage.
    n_splits:   maximum number of folds to return. If None, all folds are
                returned. When set, only the *last* n_splits folds are used.
    expanding:  True (default) = expanding train window (grow from t=0).
                False = rolling window of size `train_window`.
    train_window: size of rolling window (only used when expanding=False).
    """

    min_train: int
    test_size: int
    embargo: int = 5
    n_splits: int | None = None
    expanding: bool = True
    train_window: int | None = None

    def __post_init__(self) -> None:
        if self.embargo < 0:
            raise ValueError(f"embargo must be >= 0, got {self.embargo}")
        if self.min_train < 1:
            raise ValueError(f"min_train must be >= 1, got {self.min_train}")
        if self.test_size < 1:
            raise ValueError(f"test_size must be >= 1, got {self.test_size}")
        if not self.expanding and self.train_window is None:
            raise ValueError("rolling mode (expanding=False) requires train_window to be set")

    @classmethod
    def from_config(
        cls,
        cfg: "Config",
        trading_days_per_year: int = TRADING_DAYS_PER_YEAR,
    ) -> "WalkForwardSplitter":
        """Build a splitter from the project config.

        Converts train_years and test_years to approximate trading-day counts.
        """
        return cls(
            min_train=int(cfg.splitter.train_years * trading_days_per_year),
            test_size=int(cfg.splitter.test_years * trading_days_per_year),
            embargo=cfg.splitter.embargo,
            n_splits=cfg.splitter.n_splits,
            expanding=True,
        )

    # ---------------------------------------------------------------------- #
    # Core split logic
    # ---------------------------------------------------------------------- #

    def split(self, n: int) -> list[tuple[np.ndarray, np.ndarray]]:
        """Return a list of (train_indices, test_indices) integer arrays.

        All folds are generated first; if n_splits is set, only the last
        n_splits folds are returned (most recent evaluation periods).

        Parameters
        ----------
        n: total number of samples in the dataset.
        """
        all_folds = list(self._generate_all_folds(n))
        if self.n_splits is not None:
            all_folds = all_folds[-self.n_splits :]
        return all_folds

    def _generate_all_folds(self, n: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Generate every possible fold without the n_splits cap."""
        test_start = self.min_train + self.embargo

        while test_start + self.test_size <= n:
            train_end = test_start - self.embargo  # exclusive upper bound

            if self.expanding:
                train_start = 0
            else:
                assert self.train_window is not None
                train_start = max(0, train_end - self.train_window)

            train_idx = np.arange(train_start, train_end)
            test_idx = np.arange(test_start, min(test_start + self.test_size, n))

            yield train_idx, test_idx
            test_start += self.test_size

    def n_folds(self, n: int) -> int:
        """Count the number of folds that split(n) will produce."""
        return len(self.split(n))
