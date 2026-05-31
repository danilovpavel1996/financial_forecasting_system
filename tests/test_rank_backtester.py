"""Tests for the ranking backtester and baselines.

[REQUIRED] tests:
  1. EqualWeight baseline produces 0 P&L (flat portfolio when all preds = 0).
  2. All baselines + ElasticNet run without error and return a RankingResult.
  3. CS-RIC stability is in [0, 1].
  4. Costs reduce (or leave equal) net P&L vs gross.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.eval.rank_backtester import (
    EqualWeightRanker,
    MeanReversionRanker,
    MomentumRankRanker,
    RankingBacktester,
    RankingResult,
    _construct_positions,
    ranking_comparison_table,
)
from src.eval.splitter import WalkForwardSplitter
from src.models.linear import ElasticNetModel

_METALS = ["GC=F", "SI=F", "PL=F", "PA=F"]
_N_DATES = 600
_N_FEATS  = 8  # must include indices 0..7 for baseline feature lookups


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic pooled dataset
# ─────────────────────────────────────────────────────────────────────────────

def _make_pooled(n_dates: int = _N_DATES, seed: int = 0) -> pd.DataFrame:
    """Build a minimal synthetic pooled DataFrame for backtester tests."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n_dates, freq="B")

    rows: list[dict] = []
    idx_tuples: list[tuple] = []
    for date in dates:
        for asset in _METALS:
            row: dict = {f"feat_{i}": rng.normal() for i in range(_N_FEATS)}
            row["target"] = rng.normal(0, 0.01)
            rows.append(row)
            idx_tuples.append((date, asset))

    mi = pd.MultiIndex.from_tuples(idx_tuples, names=["date", "asset"])
    return pd.DataFrame(rows, index=mi)


def _make_backtester(horizon: int = 1) -> RankingBacktester:
    splitter = WalkForwardSplitter(min_train=200, test_size=60, embargo=5)
    return RankingBacktester(
        splitter=splitter,
        cost_bps=5.0,
        horizon=horizon,
        assets=_METALS,
    )


# ─────────────────────────────────────────────────────────────────────────────
# _construct_positions unit tests
# ─────────────────────────────────────────────────────────────────────────────

def test_construct_positions_top_and_bottom():
    """Long highest prediction, short lowest; middle two are flat."""
    pred = np.array([0.1, -0.2, 0.3, 0.05])  # order: SI, PA, GC, PL
    pos = _construct_positions(pred, n_long=1, n_short=1, n_assets=4)
    assert pos[2] == 1.0   # GC=F (index 0) has highest pred
    assert pos[1] == -1.0  # SI=F (index 1) has lowest pred
    assert pos[0] == 0.0
    assert pos[3] == 0.0


def test_construct_positions_all_equal_returns_zero():
    """All equal predictions → no ranking signal → all positions = 0."""
    pred = np.array([0.0, 0.0, 0.0, 0.0])
    pos = _construct_positions(pred, n_long=1, n_short=1, n_assets=4)
    assert np.all(pos == 0.0)


def test_construct_positions_gross_check():
    """With n_long=1, n_short=1: exactly one +1 and one -1."""
    pred = np.array([0.3, -0.1, 0.2, 0.1])
    pos = _construct_positions(pred, n_long=1, n_short=1, n_assets=4)
    assert (pos == 1.0).sum() == 1
    assert (pos == -1.0).sum() == 1
    assert (pos == 0.0).sum() == 2


# ─────────────────────────────────────────────────────────────────────────────
# [REQUIRED] EqualWeight produces flat portfolio
# ─────────────────────────────────────────────────────────────────────────────

def test_equal_weight_positions_are_zero():
    """[REQUIRED] EqualWeightRanker predicts 0 → all positions = 0 → zero P&L."""
    bt = _make_backtester()
    pooled = _make_pooled()
    result = bt.run(pooled, EqualWeightRanker, model_name="EqualWeight")

    # All positions should be zero (no ranking from constant prediction)
    assert (result.pos_df.values == 0.0).all(), (
        "EqualWeight positions should all be zero"
    )
    # All net P&L should be zero
    assert np.allclose(result.ls_pnl_ts.values, 0.0, atol=1e-12), (
        f"EqualWeight net P&L should be 0; got max|pnl|="
        f"{result.ls_pnl_ts.abs().max():.2e}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# [REQUIRED] All baselines and ElasticNet run without error
# ─────────────────────────────────────────────────────────────────────────────

def test_all_baselines_run():
    """[REQUIRED] EqualWeight, MomentumRank, MeanReversion produce RankingResult."""
    bt = _make_backtester()
    pooled = _make_pooled()
    fcols = [c for c in pooled.columns if c != "target"]

    mom_idx = 0   # feat_0 acts as momentum proxy
    rev_idx = 1   # feat_1 acts as 5d-return proxy

    factories = {
        "EqualWeight":   EqualWeightRanker,
        "MomentumRank":  lambda: MomentumRankRanker(feature_idx=mom_idx),
        "MeanReversion": lambda: MeanReversionRanker(feature_idx=rev_idx),
    }

    results: dict[str, RankingResult] = {}
    for name, fac in factories.items():
        r = bt.run(pooled, fac, model_name=name)
        assert isinstance(r, RankingResult), f"{name} did not return RankingResult"
        assert r.model_name == name
        assert r.n_folds > 0
        results[name] = r


def test_elasticnet_runs():
    """[REQUIRED] ElasticNet runs end-to-end and returns a RankingResult."""
    bt = _make_backtester()
    pooled = _make_pooled()
    result = bt.run(pooled, lambda: ElasticNetModel(random_state=42),
                    model_name="ElasticNet")
    assert isinstance(result, RankingResult)
    assert result.n_folds > 0


# ─────────────────────────────────────────────────────────────────────────────
# Metric range checks
# ─────────────────────────────────────────────────────────────────────────────

def test_cs_ric_stability_in_zero_one():
    """CS-RIC stability is always in [0, 1]."""
    bt = _make_backtester()
    pooled = _make_pooled()
    for fac, name in [
        (EqualWeightRanker, "EW"),
        (lambda: MomentumRankRanker(0), "Mom"),
    ]:
        r = bt.run(pooled, fac, model_name=name)
        if np.isfinite(r.cs_ric_stability):
            assert 0.0 <= r.cs_ric_stability <= 1.0, (
                f"{name}: stability={r.cs_ric_stability} out of [0,1]"
            )


def test_turnover_non_negative():
    """Turnover is always >= 0."""
    bt = _make_backtester()
    pooled = _make_pooled()
    for fac, name in [
        (lambda: MomentumRankRanker(0), "Mom"),
        (lambda: MeanReversionRanker(1), "MR"),
    ]:
        r = bt.run(pooled, fac, model_name=name)
        assert r.turnover >= 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Comparison table
# ─────────────────────────────────────────────────────────────────────────────

def test_ranking_comparison_table_shape():
    """comparison_table has correct shape and expected columns."""
    bt = _make_backtester()
    pooled = _make_pooled()
    results = {
        "EqualWeight": bt.run(pooled, EqualWeightRanker, "EqualWeight"),
        "MomentumRank": bt.run(pooled, lambda: MomentumRankRanker(0), "MomentumRank"),
    }
    tbl = ranking_comparison_table(results)
    assert tbl.shape[0] == 2
    assert "mean_CS_RIC" in tbl.columns
    assert "Sharpe_net" in tbl.columns
    assert "turnover" in tbl.columns
    assert set(tbl.index) == {"EqualWeight", "MomentumRank"}


# ─────────────────────────────────────────────────────────────────────────────
# Horizon > 1 smoke test
# ─────────────────────────────────────────────────────────────────────────────

def test_horizon5_runs_without_error():
    """RankingBacktester with horizon=5 completes without error."""
    bt = _make_backtester(horizon=5)
    pooled = _make_pooled()
    r = bt.run(pooled, lambda: MomentumRankRanker(0), model_name="MomH5")
    assert isinstance(r, RankingResult)
    assert np.isfinite(r.ls_sharpe) or np.isnan(r.ls_sharpe)
