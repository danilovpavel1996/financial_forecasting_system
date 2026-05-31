"""Model protocol — every model in this repo must satisfy this interface."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Model(Protocol):
    """sklearn-style fit/predict contract.

    fit and predict both operate on numpy arrays so models stay decoupled from
    pandas index semantics. The harness handles all alignment externally.
    """

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Model": ...

    def predict(self, X: np.ndarray) -> np.ndarray: ...
