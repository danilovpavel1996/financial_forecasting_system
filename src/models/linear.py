"""ElasticNet regression with train-only StandardScaler.

The StandardScaler is fit INSIDE the walk-forward fold on training data only.
This is enforced by wrapping both steps in a single sklearn Pipeline so the
harness never has an opportunity to leak test-set statistics into the scaler.

If cross-validation is ever added inside a fold, the Pipeline will
automatically re-fit the scaler for each inner fold, preserving the invariant.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class ElasticNetModel:
    """ElasticNet regression with train-only StandardScaler in a single Pipeline.

    Parameters
    ----------
    alpha:        overall regularization strength.
    l1_ratio:     L1 vs L2 mix: 0.0 = Ridge, 1.0 = Lasso.
    max_iter:     maximum iterations for the coordinate-descent solver.
    random_state: for reproducibility.
    """

    def __init__(
        self,
        alpha: float = 0.01,
        l1_ratio: float = 0.5,
        max_iter: int = 2000,
        random_state: int = 42,
    ) -> None:
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self._pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", ElasticNet(
                alpha=alpha,
                l1_ratio=l1_ratio,
                max_iter=max_iter,
                random_state=random_state,
                fit_intercept=True,
            )),
        ])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ElasticNetModel":
        self._pipeline.fit(np.asarray(X, float), np.asarray(y, float))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._pipeline.predict(np.asarray(X, float))

    @property
    def coef_(self) -> np.ndarray:
        """Coefficients back-transformed to the original (unscaled) feature space."""
        scaler: StandardScaler = self._pipeline.named_steps["scaler"]
        model: ElasticNet = self._pipeline.named_steps["model"]
        return model.coef_ / scaler.scale_

    @property
    def scaler_(self) -> StandardScaler:
        return self._pipeline.named_steps["scaler"]

    def __repr__(self) -> str:
        return f"ElasticNetModel(alpha={self.alpha}, l1_ratio={self.l1_ratio})"
