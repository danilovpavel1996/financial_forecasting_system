"""Tests for LambdaMARTModel.

Required:
  1. Model fits without error on synthetic data.
  2. Predictions are finite and not all identical (model differentiates assets).
  3. requires_groups flag is True.
  4. fit() without date_codes raises ValueError.
  5. _returns_to_relevance assigns 0 to min-return asset, N-1 to max-return asset.
  6. _compute_group_sizes returns correct consecutive sizes.
  7. Model integrates with RankingBacktester via the requires_groups path.
  8. Backtester produces a valid RankingResult when LambdaMART is used.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.eval.rank_backtester import RankingBacktester, RankingResult
from src.eval.splitter import WalkForwardSplitter
from src.models.lambdamart import LambdaMARTModel

_METALS = ["GC=F", "SI=F", "PL=F", "PA=F"]
_N_DATES = 300
_N_FEATS = 8


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_synthetic(n_dates: int = _N_DATES, n_assets: int = 4, seed: int = 0):
    """Return (X, y, date_codes) for a pooled-style dataset."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n_dates, freq="B")
    X_rows, y_rows, dc_rows = [], [], []
    for d in dates:
        for _ in range(n_assets):
            X_rows.append(rng.normal(size=_N_FEATS))
            y_rows.append(rng.normal(0, 0.01))
            dc_rows.append(d)
    return (
        np.array(X_rows, dtype=float),
        np.array(y_rows, dtype=float),
        np.array(dc_rows),
    )


def _make_pooled(n_dates: int = _N_DATES, seed: int = 0) -> pd.DataFrame:
    """Build a minimal synthetic pooled DataFrame for backtester integration tests."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n_dates, freq="B")
    rows, idx = [], []
    for date in dates:
        for asset in _METALS:
            row = {f"feat_{i}": rng.normal() for i in range(_N_FEATS)}
            row["target"] = rng.normal(0, 0.01)
            rows.append(row)
            idx.append((date, asset))
    mi = pd.MultiIndex.from_tuples(idx, names=["date", "asset"])
    return pd.DataFrame(rows, index=mi)


# ─────────────────────────────────────────────────────────────────────────────
# Unit tests
# ─────────────────────────────────────────────────────────────────────────────

def test_requires_groups_flag():
    """[REQUIRED] requires_groups attribute must be True."""
    m = LambdaMARTModel()
    assert m.requires_groups is True


def test_fit_without_date_codes_raises():
    """[REQUIRED] fit() without date_codes must raise ValueError."""
    X, y, _ = _make_synthetic(n_dates=50)
    m = LambdaMARTModel()
    with pytest.raises(ValueError, match="date_codes"):
        m.fit(X, y)


def test_compute_group_sizes_uniform():
    """_compute_group_sizes returns uniform sizes for uniform date codes."""
    m = LambdaMARTModel()
    dc = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    sizes = m._compute_group_sizes(dc)
    assert sizes == [3, 3, 3]


def test_compute_group_sizes_variable():
    """_compute_group_sizes handles non-uniform group sizes."""
    m = LambdaMARTModel()
    dc = np.array([0, 0, 1, 2, 2, 2])
    sizes = m._compute_group_sizes(dc)
    assert sizes == [2, 1, 3]


def test_returns_to_relevance_two_assets():
    """Worst return gets label 0, best gets label 1."""
    m = LambdaMARTModel()
    # 2 dates, 2 assets each
    dc = np.array([0, 0, 1, 1])
    y  = np.array([0.01, -0.01, 0.02, 0.03])
    rel = m._returns_to_relevance(y, dc)
    # date 0: y[0]=0.01 > y[1]=-0.01  → labels [1, 0]
    assert rel[0] == 1 and rel[1] == 0
    # date 1: y[2]=0.02 < y[3]=0.03   → labels [0, 1]
    assert rel[2] == 0 and rel[3] == 1


def test_returns_to_relevance_n_assets():
    """For N assets per date, labels cover exactly 0..N-1."""
    m = LambdaMARTModel()
    n_assets = 4
    dc = np.zeros(n_assets, dtype=int)
    y  = np.array([0.01, -0.05, 0.03, 0.00])
    rel = m._returns_to_relevance(y, dc)
    assert sorted(rel.tolist()) == list(range(n_assets))


def test_fit_and_predict_basic():
    """[REQUIRED] Model fits without error and returns finite predictions."""
    X, y, dc = _make_synthetic(n_dates=150)
    m = LambdaMARTModel(n_estimators=50)
    m.fit(X, y, dc)

    # Predict on a single date (4 assets)
    X_test = X[:4]
    preds = m.predict(X_test)
    assert preds.shape == (4,)
    assert np.all(np.isfinite(preds)), "All predictions must be finite"


def test_predictions_are_diverse():
    """[REQUIRED] Predictions on distinct feature rows should not all be identical."""
    X, y, dc = _make_synthetic(n_dates=150)
    m = LambdaMARTModel(n_estimators=50)
    m.fit(X, y, dc)
    preds = m.predict(X[:4])
    assert np.std(preds) > 1e-10, "LambdaMART should produce diverse scores"


def test_predict_before_fit_raises():
    """predict() before fit() must raise RuntimeError."""
    m = LambdaMARTModel()
    with pytest.raises(RuntimeError):
        m.predict(np.zeros((4, _N_FEATS)))


# ─────────────────────────────────────────────────────────────────────────────
# Integration: RankingBacktester passes date codes
# ─────────────────────────────────────────────────────────────────────────────

def test_lambdamart_runs_in_backtester():
    """[REQUIRED] LambdaMART integrates with RankingBacktester and returns RankingResult."""
    splitter = WalkForwardSplitter(min_train=150, test_size=60, embargo=5)
    bt = RankingBacktester(
        splitter=splitter,
        cost_bps=5.0,
        horizon=5,
        assets=_METALS,
    )
    pooled = _make_pooled(n_dates=_N_DATES)
    result = bt.run(pooled, lambda: LambdaMARTModel(n_estimators=50), model_name="LambdaMART")

    assert isinstance(result, RankingResult)
    assert result.model_name == "LambdaMART"
    assert result.n_folds > 0
    assert len(result.cs_ric_ts) > 0
    assert len(result.ls_pnl_ts) > 0


def test_lambdamart_predictions_finite_in_backtester():
    """Predictions collected in the backtester are all finite."""
    splitter = WalkForwardSplitter(min_train=150, test_size=60, embargo=5)
    bt = RankingBacktester(
        splitter=splitter,
        cost_bps=5.0,
        horizon=5,
        assets=_METALS,
    )
    pooled = _make_pooled(n_dates=_N_DATES)
    result = bt.run(pooled, lambda: LambdaMARTModel(n_estimators=50), model_name="LambdaMART")

    assert np.all(np.isfinite(result.pred_df.values)), "All OOS predictions must be finite"


def test_lambdamart_cs_ric_stability_in_range():
    """CS-RIC stability from LambdaMART must be in [0, 1] when finite."""
    splitter = WalkForwardSplitter(min_train=150, test_size=60, embargo=5)
    bt = RankingBacktester(
        splitter=splitter,
        cost_bps=5.0,
        horizon=5,
        assets=_METALS,
    )
    pooled = _make_pooled(n_dates=_N_DATES)
    result = bt.run(pooled, lambda: LambdaMARTModel(n_estimators=50), model_name="LambdaMART")

    if np.isfinite(result.cs_ric_stability):
        assert 0.0 <= result.cs_ric_stability <= 1.0
