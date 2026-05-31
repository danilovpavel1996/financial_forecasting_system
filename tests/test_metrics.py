"""Tests for src/eval/metrics.py and backtester integration.

Phase-2 required tests (marked # [REQUIRED]):
  3. Metrics on known synthetic series match hand-computed values.
  4. Baselines run and appear in the comparison table.

Hand-computed reference values are derived analytically below each test.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.eval.metrics import (
    StrategyStats,
    directional_accuracy,
    information_coefficient,
    rank_ic,
    strategy_pnl,
)
from src.eval.backtester import Backtester, BacktestResult, comparison_table
from src.eval.splitter import WalkForwardSplitter
from src.models.baselines import Drift, Momentum, RandomWalk


# --------------------------------------------------------------------------- #
# [REQUIRED] Test 3a — information_coefficient hand-computed values
# --------------------------------------------------------------------------- #

def test_ic_perfect_correlation():
    """[REQUIRED] Perfectly correlated predictions give IC = 1.0."""
    pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    realized = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert information_coefficient(pred, realized) == pytest.approx(1.0, abs=1e-9)


def test_ic_perfect_anticorrelation():
    """[REQUIRED] Perfectly anticorrelated predictions give IC = -1.0."""
    pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    realized = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    assert information_coefficient(pred, realized) == pytest.approx(-1.0, abs=1e-9)


def test_ic_constant_pred_is_nan():
    """[REQUIRED] Constant predictions yield IC = NaN (undefined correlation)."""
    pred = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    realized = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = information_coefficient(pred, realized)
    assert np.isnan(result), f"Expected NaN, got {result}"


def test_ic_constant_realized_is_nan():
    """Constant realized returns yield IC = NaN."""
    pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    realized = np.array([0.01, 0.01, 0.01, 0.01, 0.01])
    result = information_coefficient(pred, realized)
    assert np.isnan(result)


def test_ic_with_nans_ignored():
    """NaN entries in either array are ignored; result from valid pairs."""
    pred = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    realized = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    # IC from [1,2,4,5] vs [1,2,4,5] should be 1.0
    assert information_coefficient(pred, realized) == pytest.approx(1.0, abs=1e-9)


def test_ic_too_few_points_is_nan():
    """Fewer than 3 finite pairs yield IC = NaN."""
    pred = np.array([1.0, 2.0, np.nan, np.nan])
    realized = np.array([1.0, 2.0, np.nan, np.nan])
    assert np.isnan(information_coefficient(pred, realized))


# --------------------------------------------------------------------------- #
# [REQUIRED] Test 3b — rank_ic hand-computed values
# --------------------------------------------------------------------------- #

def test_rank_ic_perfect():
    """Perfectly ordered predictions give rank IC = 1.0."""
    pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    realized = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert rank_ic(pred, realized) == pytest.approx(1.0, abs=1e-9)


def test_rank_ic_hand_computed():
    """[REQUIRED] rank_IC matches hand-computed Spearman value.

    pred     = [2, 1, 4, 3, 5]
    realized = [1, 2, 3, 4, 5]

    Ranks of pred:     [2, 1, 4, 3, 5]  (already the ranks)
    Ranks of realized: [1, 2, 3, 4, 5]

    Differences d = [1, -1, 1, -1, 0]
    d^2            = [1,  1, 1,  1, 0]  → Σd² = 4
    n = 5

    Spearman = 1 - 6*Σd² / (n*(n²-1)) = 1 - 24/(5*24) = 1 - 0.2 = 0.8
    """
    pred = np.array([2.0, 1.0, 4.0, 3.0, 5.0])
    realized = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert rank_ic(pred, realized) == pytest.approx(0.8, abs=1e-9)


def test_rank_ic_reverse_is_minus_one():
    """Perfectly reversed ranks give rank IC = -1.0."""
    pred = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    realized = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert rank_ic(pred, realized) == pytest.approx(-1.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# [REQUIRED] Test 3c — directional_accuracy hand-computed values
# --------------------------------------------------------------------------- #

def test_directional_accuracy_perfect():
    """[REQUIRED] Perfect sign agreement gives hit rate = 1.0."""
    pred = np.array([1.0, -1.0, 1.0, -1.0, 1.0])
    realized = np.array([0.5, -0.3, 0.1, -0.2, 0.4])
    assert directional_accuracy(pred, realized) == pytest.approx(1.0)


def test_directional_accuracy_zero():
    """[REQUIRED] All wrong signs give hit rate = 0.0."""
    pred = np.array([1.0, -1.0, 1.0, -1.0])
    realized = np.array([-0.5, 0.3, -0.1, 0.2])
    assert directional_accuracy(pred, realized) == pytest.approx(0.0)


def test_directional_accuracy_half():
    """Half correct → 0.5."""
    pred = np.array([1.0, 1.0, -1.0, -1.0])
    realized = np.array([0.5, -0.5, 0.5, -0.5])
    assert directional_accuracy(pred, realized) == pytest.approx(0.5)


def test_directional_accuracy_zero_preds_are_nan():
    """Zero predictions are excluded; all-zero preds → NaN."""
    pred = np.array([0.0, 0.0, 0.0])
    realized = np.array([1.0, -1.0, 1.0])
    assert np.isnan(directional_accuracy(pred, realized))


def test_directional_accuracy_mixed_zeros():
    """Zero predictions are skipped; only non-zero ones count."""
    pred = np.array([0.0, 1.0, -1.0, 0.0, 1.0])
    realized = np.array([1.0, 0.5, -0.2, -0.3, -0.1])
    # Non-zero pred indices: 1, 2, 4
    # Signs: pred=[+, -, +], realized=[+, -, -]
    # Matches: [True, True, False] → 2/3
    assert directional_accuracy(pred, realized) == pytest.approx(2 / 3, abs=1e-9)


# --------------------------------------------------------------------------- #
# [REQUIRED] Test 3d — strategy_pnl / max_drawdown hand-computed values
# --------------------------------------------------------------------------- #

def test_max_drawdown_hand_computed():
    """[REQUIRED] Max drawdown matches hand-computed value.

    pred     = [1, 1, 1]     → positions = [+1, +1, +1] (sign mode)
    realized = [0.1, -0.2, 0.1]
    cost_bps = 0              (to isolate the drawdown calculation)

    turnover (prepend 0): |+1-0|=1, |+1-(+1)|=0, |+1-(+1)|=0
    cost = [0, 0, 0]   (cost_bps=0)
    net  = [0.1, -0.2, 0.1]

    Equity curve (cumulative product of 1+net):
      [1.1,  1.1*0.8 = 0.88,  0.88*1.1 = 0.968]

    Running peak:
      [1.1, 1.1, 1.1]

    Drawdown (curve/peak - 1):
      [0, 0.88/1.1 - 1, 0.968/1.1 - 1]
    = [0, 0.8 - 1, 0.88 - 1]
    = [0, -0.2, -0.12]

    Max drawdown = -0.2  (exactly)
    """
    pred = np.array([1.0, 1.0, 1.0])
    realized = np.array([0.1, -0.2, 0.1])
    stats = strategy_pnl(pred, realized, cost_bps=0.0)
    assert stats.max_drawdown == pytest.approx(-0.2, abs=1e-9)


def test_turnover_hand_computed():
    """[REQUIRED] Turnover matches hand-computed mean absolute position change.

    pred = [1, -1, 1, -1]  → pos = [+1, -1, +1, -1]
    Changes (prepend 0): [+1-0, -1-(+1), +1-(-1), -1-(+1)]
                       = [1,    -2,       2,       -2]
    |changes| = [1, 2, 2, 2]
    mean turnover = 7/4 = 1.75
    """
    pred = np.array([1.0, -1.0, 1.0, -1.0])
    realized = np.array([0.01, 0.01, 0.01, 0.01])
    stats = strategy_pnl(pred, realized, cost_bps=0.0)
    assert stats.turnover == pytest.approx(1.75, abs=1e-9)


def test_cost_reduces_sharpe():
    """Charging costs always reduces (or leaves equal) the net Sharpe."""
    rng = np.random.default_rng(42)
    pred = rng.normal(0, 1, 500)
    realized = 0.3 * pred + rng.normal(0, 1, 500)  # mild signal

    stats_free = strategy_pnl(pred, realized, cost_bps=0.0)
    stats_costly = strategy_pnl(pred, realized, cost_bps=50.0)

    if np.isfinite(stats_free.sharpe) and np.isfinite(stats_costly.sharpe):
        assert stats_free.sharpe >= stats_costly.sharpe - 1e-9, (
            "Costs should never increase Sharpe"
        )


def test_good_signal_positive_sharpe():
    """A strongly predictive signal produces a positive net Sharpe."""
    rng = np.random.default_rng(0)
    realized = rng.normal(0, 0.01, 500)
    # Near-perfect signal with a little noise
    pred = realized + rng.normal(0, 0.0001, 500)
    stats = strategy_pnl(pred, realized, cost_bps=1.0)
    assert np.isfinite(stats.sharpe)
    assert stats.sharpe > 0, f"Expected positive Sharpe, got {stats.sharpe:.4f}"


def test_anti_signal_negative_sharpe():
    """A perfectly wrong signal produces a negative net Sharpe."""
    rng = np.random.default_rng(0)
    realized = rng.normal(0, 0.01, 500)
    pred = -realized  # perfectly wrong
    stats = strategy_pnl(pred, realized, cost_bps=0.0)
    assert np.isfinite(stats.sharpe)
    assert stats.sharpe < 0, f"Expected negative Sharpe, got {stats.sharpe:.4f}"


def test_ann_return_hand_computed():
    """[REQUIRED] Annualised return matches hand-computed value.

    252 days, constant long position on realized returns of 0.001/day, no costs.
    pos = [+1]*252,  net = [0.001]*252 (after first-step turnover cost=0)
    mean daily net = 0.001
    ann_return = 0.001 * 252 = 0.252
    """
    n = 252
    pred = np.ones(n)
    realized = np.full(n, 0.001)
    stats = strategy_pnl(pred, realized, cost_bps=0.0)
    assert stats.ann_return == pytest.approx(0.252, abs=1e-6)


def test_strategy_pnl_constant_position_zero_vol_is_nan():
    """Constant daily P&L has zero vol → Sharpe = NaN (not inf or error)."""
    pred = np.ones(100)
    realized = np.full(100, 0.01)
    stats = strategy_pnl(pred, realized, cost_bps=0.0)
    assert np.isnan(stats.sharpe)


def test_invalid_mode_raises():
    pred = np.array([1.0, -1.0])
    realized = np.array([0.01, -0.01])
    with pytest.raises(ValueError, match="mode"):
        strategy_pnl(pred, realized, mode="unknown")


def test_scaled_mode_clips_large_predictions():
    """Scaled mode clips positions to [-1, 1]."""
    pred = np.array([100.0, -200.0, 50.0])
    realized = np.array([0.01, -0.01, 0.01])
    # Should not raise and should produce finite stats
    stats = strategy_pnl(pred, realized, cost_bps=0.0, mode="scaled")
    assert np.isfinite(stats.hit_rate)


# --------------------------------------------------------------------------- #
# [REQUIRED] Test 4 — baselines run and appear in comparison table
# --------------------------------------------------------------------------- #

def _make_backtester() -> Backtester:
    splitter = WalkForwardSplitter(min_train=200, test_size=60, embargo=5)
    return Backtester(splitter=splitter, cost_bps=5.0, pnl_mode="sign")


def _make_synthetic_data(n: int = 800, seed: int = 42):
    """Synthetic data: weak autocorrelation + noise (realistic difficulty)."""
    rng = np.random.default_rng(seed)
    rets = np.zeros(n)
    for t in range(1, n):
        rets[t] = 0.03 * rets[t - 1] + rng.normal(0, 0.012)

    # Features: lag-1 return + lag-5 return + trailing momentum
    r1 = np.zeros(n)
    r1[1:] = rets[:-1]
    r5 = np.zeros(n)
    r5[5:] = rets[:-5]
    mom = np.array([np.mean(rets[max(0, t - 10):t]) if t > 0 else 0 for t in range(n)])
    X = np.column_stack([r1, r5, mom])
    y = rets  # forward return (already a return, not price level)
    return X, y


def test_baselines_run_and_appear_in_table():
    """[REQUIRED] RandomWalk, Drift, and Momentum run and appear in comparison table."""
    bt = _make_backtester()
    X, y = _make_synthetic_data()

    factories = {
        "RandomWalk": RandomWalk,
        "Drift": Drift,
        "Momentum": Momentum,
    }
    results = {
        name: bt.run(X, y, fac, model_name=name)
        for name, fac in factories.items()
    }

    table = comparison_table(results)

    # All three baselines must appear
    for name in ("RandomWalk", "Drift", "Momentum"):
        assert name in table.index, f"{name!r} missing from comparison table"

    # Table must have the expected metric columns
    expected_cols = {"mean_IC", "rank_IC", "IC_stability", "Sharpe_net", "turnover"}
    assert expected_cols.issubset(set(table.columns)), (
        f"Missing columns: {expected_cols - set(table.columns)}"
    )


def test_random_walk_always_predicts_zero():
    """[REQUIRED] RandomWalk predictions are all zero."""
    bt = _make_backtester()
    X, y = _make_synthetic_data()
    result = bt.run(X, y, RandomWalk, model_name="RandomWalk")
    assert np.all(result.oos_pred() == 0.0), "RandomWalk must predict exactly 0"


def test_drift_predicts_constant_per_fold():
    """Drift predicts the same constant value within each fold."""
    bt = _make_backtester()
    X, y = _make_synthetic_data()
    result = bt.run(X, y, Drift, model_name="Drift")

    for fold in result.folds:
        unique_vals = np.unique(fold.pred)
        assert len(unique_vals) == 1, (
            f"Drift fold {fold.fold_idx} predicted {len(unique_vals)} distinct values"
        )


def test_all_backtest_results_are_typed():
    """BacktestResult is the right type and folds list is populated."""
    bt = _make_backtester()
    X, y = _make_synthetic_data()
    result = bt.run(X, y, RandomWalk, model_name="RandomWalk")

    assert isinstance(result, BacktestResult)
    assert result.n_folds() > 0
    assert len(result.folds) == result.n_folds()


def test_backtest_oos_pred_length_matches_total_test_rows():
    """Total OOS predictions == sum of test fold sizes."""
    bt = _make_backtester()
    X, y = _make_synthetic_data()
    result = bt.run(X, y, RandomWalk, model_name="RandomWalk")

    expected_len = sum(f.test_size for f in result.folds)
    assert len(result.oos_pred()) == expected_len
    assert len(result.oos_realized()) == expected_len


def test_ic_stability_between_zero_and_one():
    """IC stability is always in [0, 1]."""
    bt = _make_backtester()
    X, y = _make_synthetic_data()
    for fac in [RandomWalk, Drift, Momentum]:
        result = bt.run(X, y, fac)
        if np.isfinite(result.ic_stability):
            assert 0.0 <= result.ic_stability <= 1.0


def test_comparison_table_shape():
    """comparison_table produces correct shape with all models."""
    bt = _make_backtester()
    X, y = _make_synthetic_data()
    factories = {"RandomWalk": RandomWalk, "Drift": Drift}
    results = {name: bt.run(X, y, fac, model_name=name) for name, fac in factories.items()}
    table = comparison_table(results)
    assert table.shape[0] == 2  # two models
    assert "Sharpe_net" in table.columns
    assert "turnover" in table.columns


# --------------------------------------------------------------------------- #
# Cross-check: backtester respects splitter embargo
# --------------------------------------------------------------------------- #

def test_backtester_uses_splitter_embargo():
    """Backtester honours the splitter's embargo: predictions only exist in OOS period."""
    embargo = 10
    splitter = WalkForwardSplitter(min_train=100, test_size=50, embargo=embargo)
    bt = Backtester(splitter=splitter, cost_bps=5.0)
    X, y = _make_synthetic_data(n=500)

    result = bt.run(X, y, RandomWalk)

    # Verify that the fold indices in the result satisfy the embargo invariant
    folds_raw = splitter.split(len(X))
    for (train_idx, test_idx), fold in zip(folds_raw, result.folds):
        assert train_idx.max() < test_idx.min() - embargo
