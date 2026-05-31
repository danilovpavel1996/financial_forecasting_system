"""LightGBM wrappers with time-respecting early stopping.

Early stopping design
---------------------
Standard early stopping uses a random validation holdout. That is wrong here:
a random subset can include rows that are LATER in time than the "training" rows
it is paired with, creating look-ahead bias in the stopping criterion.

We use the TAIL of the training data as the validation set, with an inner
embargo separating the fit portion from the validation portion:

    [0 ........... fit_end) [fit_end .... val_start) [val_start ..... n-1]
         actual fit data         inner embargo           ES validation

The inner embargo (≥ horizon, default 5 trading days) ensures no training label
at the tail of the fit portion can look ahead into the ES validation window.
The outer embargo from WalkForwardSplitter already isolates the entire training
block from the test fold.
"""
from __future__ import annotations

import logging

import lightgbm as lgb
import numpy as np

logger = logging.getLogger(__name__)


class LightGBMModel:
    """LightGBM point-forecast regression with time-respecting early stopping.

    Parameters
    ----------
    n_estimators:          maximum number of boosting rounds.
    learning_rate:         shrinkage per round.
    num_leaves:            maximum leaves per tree.
    min_child_samples:     minimum samples per leaf (primary overfitting guard).
    subsample:             row fraction sampled per tree.
    colsample_bytree:      feature fraction sampled per tree.
    reg_alpha:             L1 leaf-weight regularization.
    reg_lambda:            L2 leaf-weight regularization.
    val_fraction:          fraction of training data reserved for early stopping.
    inner_embargo:         rows dropped between fit portion and ES window.
    early_stopping_rounds: stop if validation metric hasn't improved in this many rounds.
    random_state:          for reproducibility.
    """

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
            "objective": "regression",
            "metric": "rmse",
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

    def _split_train_val(self, n: int) -> tuple[int, int]:
        """Return (fit_end, val_start) for the time-ordered split.

        Returns (-1, -1) when the dataset is too small to split meaningfully.
        """
        val_size = max(20, int(n * self._val_fraction))
        val_start = n - val_size
        fit_end = val_start - self._inner_embargo
        if fit_end < 20:
            return -1, -1
        return fit_end, val_start

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LightGBMModel":
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        n = len(X)

        fit_end, val_start = self._split_train_val(n)

        if fit_end < 0:
            logger.debug(
                "LightGBM: n=%d too small for ES split; training without early stopping", n
            )
            ds_train = lgb.Dataset(X, label=y)
            self._booster = lgb.train(
                self._params,
                ds_train,
                num_boost_round=self._n_estimators,
            )
        else:
            ds_fit = lgb.Dataset(X[:fit_end], label=y[:fit_end])
            ds_val = lgb.Dataset(
                X[val_start:], label=y[val_start:], reference=ds_fit
            )
            callbacks = [
                lgb.early_stopping(
                    stopping_rounds=self._early_stopping_rounds,
                    verbose=False,
                ),
                lgb.log_evaluation(period=-1),
            ]
            self._booster = lgb.train(
                self._params,
                ds_fit,
                num_boost_round=self._n_estimators,
                valid_sets=[ds_val],
                callbacks=callbacks,
            )
            logger.debug(
                "LightGBM: %d trees fitted (fit rows=%d, val rows=%d, inner_embargo=%d)",
                self._booster.num_trees(), fit_end, n - val_start, self._inner_embargo,
            )

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
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


class LightGBMQuantileModel(LightGBMModel):
    """LightGBM quantile regression.

    Predicts the `quantile_alpha`-quantile of the return distribution rather
    than the conditional mean. A high predicted upper quantile signals elevated
    upside risk and is used as a long signal; a low value signals short.
    Rank-IC is the natural metric: it measures whether the quantile ranking
    agrees with the realized-return ranking.

    Parameters
    ----------
    quantile_alpha: the quantile level to predict (default 0.9 = upper tail).
    **kwargs:       forwarded to LightGBMModel.
    """

    def __init__(self, quantile_alpha: float = 0.9, **kwargs) -> None:
        super().__init__(**kwargs)
        self._quantile_alpha = quantile_alpha
        self._params["objective"] = "quantile"
        self._params["metric"] = "quantile"
        self._params["alpha"] = quantile_alpha
