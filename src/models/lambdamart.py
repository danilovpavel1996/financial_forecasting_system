"""LightGBM LambdaMART wrapper for cross-sectional ranking.

LambdaMART trains directly on the ranking objective (NDCG) rather than
fitting return predictions with MSE. Each date is a "query" containing
all N assets; the model learns which asset orderings correspond to the
best forward returns.

Unlike LightGBMModel, this model requires group labels at fit time.
The `requires_groups = True` attribute signals RankingBacktester to pass
date codes alongside X and y.

Early-stopping design mirrors gbm.py: split on the TEMPORAL TAIL of
training dates (not a random subset). The split is at date boundaries
so no partial-date groups straddle the fit/val boundary.
"""
from __future__ import annotations

import logging

import lightgbm as lgb
import numpy as np

logger = logging.getLogger(__name__)

def _label_gain(n: int) -> list[int]:
    """NDCG gain values for labels 0..n-1 using the standard 2^i-1 formula."""
    return [max(0, (1 << i) - 1) for i in range(n)]


class LambdaMARTModel:
    """LightGBM with lambdarank objective for cross-sectional ranking.

    Unlike the standard LightGBM (MSE objective), this model is trained
    to produce the correct ORDERING of assets, not accurate point
    predictions of returns.

    Training requires group labels: the number of items (assets) per
    query (date). For our pooled dataset with N assets per date,
    every group has size N.

    Parameters
    ----------
    n_estimators:          maximum number of boosting rounds.
    learning_rate:         shrinkage per round.
    num_leaves:            maximum leaves per tree.
    min_child_samples:     minimum samples per leaf.
    subsample:             row fraction sampled per tree.
    colsample_bytree:      feature fraction sampled per tree.
    reg_alpha:             L1 leaf-weight regularization.
    reg_lambda:            L2 leaf-weight regularization.
    val_fraction:          fraction of training DATES reserved for ES validation.
    inner_embargo:         dates dropped between fit portion and ES window.
    early_stopping_rounds: stop if NDCG metric hasn't improved in this many rounds.
    random_state:          for reproducibility.
    """

    requires_groups = True  # tells RankingBacktester to pass date codes to fit()

    def __init__(
        self,
        n_estimators: int = 500,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        min_child_samples: int = 20,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.1,
        reg_lambda: float = 0.1,
        val_fraction: float = 0.15,
        inner_embargo: int = 5,
        early_stopping_rounds: int = 50,
        random_state: int = 42,
    ) -> None:
        self._n_estimators = n_estimators
        self._val_fraction = val_fraction
        self._inner_embargo = inner_embargo
        self._early_stopping_rounds = early_stopping_rounds
        self._params: dict = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_eval_at": [3, 5],
            "learning_rate": learning_rate,
            "num_leaves": num_leaves,
            "min_child_samples": min_child_samples,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "seed": random_state,
            "verbose": -1,
            "n_jobs": 1,
        }
        self._booster: lgb.Booster | None = None

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _compute_group_sizes(self, date_codes: np.ndarray) -> list[int]:
        """Return list of consecutive group sizes from a date-code array.

        Assumes date_codes is sorted (all rows for the same date are contiguous),
        which is guaranteed by the (date, asset) MultiIndex sort in pooled datasets.
        """
        if len(date_codes) == 0:
            return []
        sizes: list[int] = []
        current = date_codes[0]
        count = 1
        for dc in date_codes[1:]:
            if dc == current:
                count += 1
            else:
                sizes.append(count)
                current = dc
                count = 1
        sizes.append(count)
        return sizes

    def _returns_to_relevance(
        self, y: np.ndarray, date_codes: np.ndarray
    ) -> np.ndarray:
        """Convert realized returns to integer relevance labels within each date.

        Within each query (date), assets are ranked 0 = worst to N-1 = best.
        Ties are broken by the argsort of argsort (stable, lowest index wins).
        """
        relevance = np.zeros(len(y), dtype=np.int32)
        i = 0
        while i < len(date_codes):
            current = date_codes[i]
            j = i
            while j < len(date_codes) and date_codes[j] == current:
                j += 1
            y_group = y[i:j]
            # argsort of argsort: 0 = position of the smallest element, N-1 = largest
            ranks = np.argsort(np.argsort(y_group))
            relevance[i:j] = ranks.astype(np.int32)
            i = j
        return relevance

    def _split_train_val_dates(self, n_dates: int) -> tuple[int, int]:
        """Return (fit_end, val_start) date-level indices.

        Returns (-1, -1) when the dataset is too small for a meaningful split.
        """
        val_dates = max(10, int(n_dates * self._val_fraction))
        val_start = n_dates - val_dates
        fit_end = val_start - self._inner_embargo
        if fit_end < 20:
            return -1, -1
        return fit_end, val_start

    # ── Public API ────────────────────────────────────────────────────────────

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        date_codes: np.ndarray | None = None,
    ) -> "LambdaMARTModel":
        """Fit the LambdaMART model.

        Parameters
        ----------
        X:          feature matrix (n_rows × n_features).
        y:          realized forward returns (n_rows,).
        date_codes: array of date labels, one per row. Must be sorted so that
                    all rows for the same date are contiguous. Required.
        """
        if date_codes is None:
            raise ValueError(
                "LambdaMARTModel.fit() requires date_codes. "
                "RankingBacktester passes these when requires_groups=True."
            )

        X = np.asarray(X, float)
        y = np.asarray(y, float)
        date_codes = np.asarray(date_codes)

        groups = self._compute_group_sizes(date_codes)
        n_dates = len(groups)
        relevance = self._returns_to_relevance(y, date_codes)

        # label_gain must cover the full label range (0 .. max_group_size - 1)
        max_group = max(groups) if groups else 1
        params = dict(self._params, label_gain=_label_gain(max_group))

        fit_end, val_start = self._split_train_val_dates(n_dates)

        if fit_end < 0:
            logger.debug(
                "LambdaMART: n_dates=%d too small for ES split; training without early stopping",
                n_dates,
            )
            ds = lgb.Dataset(
                X, label=relevance, group=groups, free_raw_data=False
            )
            self._booster = lgb.train(
                params, ds, num_boost_round=self._n_estimators
            )
        else:
            cum = np.concatenate([[0], np.cumsum(groups)])
            fit_rows = int(cum[fit_end])
            val_rows_start = int(cum[val_start])

            ds_fit = lgb.Dataset(
                X[:fit_rows],
                label=relevance[:fit_rows],
                group=groups[:fit_end],
                free_raw_data=False,
            )
            ds_val = lgb.Dataset(
                X[val_rows_start:],
                label=relevance[val_rows_start:],
                group=groups[val_start:],
                reference=ds_fit,
                free_raw_data=False,
            )
            callbacks = [
                lgb.early_stopping(
                    stopping_rounds=self._early_stopping_rounds, verbose=False
                ),
                lgb.log_evaluation(period=-1),
            ]
            self._booster = lgb.train(
                params,
                ds_fit,
                num_boost_round=self._n_estimators,
                valid_sets=[ds_val],
                callbacks=callbacks,
            )
            logger.debug(
                "LambdaMART: %d trees (fit_dates=%d, val_dates=%d, inner_embargo=%d)",
                self._booster.num_trees(),
                fit_end,
                n_dates - val_start,
                self._inner_embargo,
            )

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return relevance scores (higher = model expects better ranking)."""
        if self._booster is None:
            raise RuntimeError("predict() called before fit()")
        return self._booster.predict(np.asarray(X, float))

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"n_estimators={self._n_estimators}, "
            f"lr={self._params['learning_rate']}, "
            f"num_leaves={self._params['num_leaves']})"
        )
