"""Baseline models — the benchmarks every real model must beat.

RandomWalk and Drift are the null hypotheses. A model that cannot consistently
beat both of these on out-of-sample data has no demonstrated edge.
"""
from __future__ import annotations

import numpy as np


class RandomWalk:
    """Predict zero return every day.

    This is the hardest baseline to beat at a daily horizon. Markets are close
    to a random walk in the short term. If your model can't beat this, stop.
    """

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomWalk":
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.zeros(len(X))


class Drift:
    """Predict the mean return seen in training (the unconditional drift).

    Captures the positive equity/commodity drift over long horizons. Easy to
    beat in up-markets, a fair bar in flat/down markets.
    """

    def __init__(self) -> None:
        self.mu_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Drift":
        self.mu_ = float(np.nanmean(y))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), self.mu_)


class Momentum:
    """Predict scaled sign of the last feature column (a momentum proxy).

    Expects the last column of X to be a lagged return or momentum signal.
    Scales by training std so the prediction magnitude is plausible.
    Pure heuristic — no fitting beyond capturing the signal's scale.
    """

    def __init__(self) -> None:
        self.scale_: float = 1.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Momentum":
        last_col = X[:, -1]
        s = float(np.nanstd(last_col))
        self.scale_ = s if s > 0 else 1.0
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.sign(X[:, -1]) * self.scale_
