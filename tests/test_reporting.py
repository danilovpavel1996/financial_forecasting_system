"""Tests for src/reporting.py."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.eval.backtester import BacktestResult, FoldResult
from src.eval.metrics import StrategyStats
from src.reporting import (
    build_verdict,
    config_hash,
    daily_net_pnl,
    equity_curve,
    save_equity_figure,
    save_rolling_ic_figure,
    write_report,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_fold(fold_idx: int, n: int = 50, ic: float = 0.0, seed: int = 0) -> FoldResult:
    rng = np.random.default_rng(seed)
    realized = rng.standard_normal(n)
    pred = ic * realized + np.sqrt(1 - ic ** 2) * rng.standard_normal(n)
    from scipy.stats import pearsonr
    actual_ic = float(pearsonr(pred, realized)[0])
    return FoldResult(
        fold_idx=fold_idx,
        train_size=200,
        test_size=n,
        ic=actual_ic,
        rank_ic=actual_ic,
        directional_accuracy=0.5,
        pred=pred,
        realized=realized,
    )


def _make_result(
    name: str,
    n_folds: int = 4,
    ic_per_fold: float = 0.0,
    pooled_ic: float = 0.0,
    sharpe: float = 0.5,
    ic_stability: float = 0.5,
    n_per_fold: int = 50,
) -> BacktestResult:
    folds = [_make_fold(i, n=n_per_fold, ic=ic_per_fold, seed=i) for i in range(n_folds)]
    return BacktestResult(
        model_name=name,
        folds=folds,
        pooled_ic=pooled_ic,
        pooled_rank_ic=pooled_ic,
        mean_ic=ic_per_fold,
        mean_rank_ic=ic_per_fold,
        mean_directional_accuracy=0.5,
        ic_stability=ic_stability,
        strategy=StrategyStats(
            sharpe=sharpe,
            ann_return=sharpe * 0.1,
            ann_vol=0.1,
            max_drawdown=-0.15,
            turnover=0.5,
            hit_rate=0.52,
        ),
    )


def _oos_index(n_folds: int = 4, n_per_fold: int = 50) -> pd.DatetimeIndex:
    return pd.bdate_range("2020-01-01", periods=n_folds * n_per_fold)


# ─────────────────────────────────────────────────────────────────────────────
# config_hash
# ─────────────────────────────────────────────────────────────────────────────

class TestConfigHash:
    def test_returns_8_hex_chars(self):
        cfg = load_config()
        h = config_hash(cfg)
        assert len(h) == 8
        assert all(c in "0123456789abcdef" for c in h)

    def test_stable_across_calls(self):
        cfg = load_config()
        assert config_hash(cfg) == config_hash(cfg)

    def test_same_config_same_hash(self):
        cfg1 = load_config()
        cfg2 = load_config()
        assert config_hash(cfg1) == config_hash(cfg2)

    def test_hash_changes_with_horizon(self):
        cfg = load_config()
        original = config_hash(cfg)
        cfg.horizon = cfg.horizon + 1
        # Temporarily allow invalid embargo to test hash only
        cfg.splitter.embargo = cfg.horizon  # keep valid
        assert config_hash(cfg) != original

    def test_hash_changes_with_dates(self):
        cfg = load_config()
        original = config_hash(cfg)
        cfg.dates = {**cfg.dates, "start": "2011-01-01"}
        assert config_hash(cfg) != original

    def test_hash_changes_with_cost(self):
        cfg = load_config()
        original = config_hash(cfg)
        cfg.cost_bps = cfg.cost_bps + 1.0
        assert config_hash(cfg) != original


# ─────────────────────────────────────────────────────────────────────────────
# PnL helpers
# ─────────────────────────────────────────────────────────────────────────────

class TestPnLHelpers:
    def test_equity_curve_starts_near_one(self):
        pred = np.ones(100)
        real = np.random.default_rng(0).standard_normal(100) * 0.01
        curve = equity_curve(pred, real, cost_bps=0)
        assert abs(curve[0] - (1.0 + real[0])) < 1e-10

    def test_zero_position_gives_flat_curve(self):
        pred = np.zeros(100)  # sign(0) = 0, no position
        real = np.random.default_rng(0).standard_normal(100)
        curve = equity_curve(pred, real, cost_bps=5)
        # Net PnL is pure cost on first position change: 0 → 0 → cost is 0
        assert np.allclose(curve, 1.0)

    def test_perfect_long_positive_returns(self):
        pred = np.ones(50)
        real = np.full(50, 0.01)
        curve = equity_curve(pred, real, cost_bps=0)
        assert curve[-1] > 1.0

    def test_daily_pnl_shape(self):
        pred = np.random.randn(80)
        real = np.random.randn(80)
        pnl = daily_net_pnl(pred, real, cost_bps=5)
        assert pnl.shape == (80,)

    def test_cost_reduces_pnl(self):
        pred = np.sign(np.random.randn(100))
        real = np.random.randn(100) * 0.01
        pnl_free = daily_net_pnl(pred, real, cost_bps=0)
        pnl_cost = daily_net_pnl(pred, real, cost_bps=10)
        assert pnl_free.sum() >= pnl_cost.sum()


# ─────────────────────────────────────────────────────────────────────────────
# Verdict
# ─────────────────────────────────────────────────────────────────────────────

class TestVerdict:
    def _results_no_edge(self):
        return {
            "RandomWalk": _make_result("RandomWalk", sharpe=np.nan, pooled_ic=np.nan),
            "Drift": _make_result("Drift", sharpe=0.56, pooled_ic=-0.003),
            "MyModel": _make_result("MyModel", sharpe=0.4, pooled_ic=0.01, ic_stability=0.4),
        }

    def _results_with_edge(self):
        return {
            "RandomWalk": _make_result("RandomWalk", sharpe=np.nan, pooled_ic=np.nan),
            "Drift": _make_result("Drift", sharpe=0.56, pooled_ic=-0.003),
            "GoodModel": _make_result("GoodModel", sharpe=1.2, pooled_ic=0.04, ic_stability=0.75),
        }

    def _results_with_leakage(self):
        return {
            "Drift": _make_result("Drift", sharpe=0.56, pooled_ic=-0.003),
            "BadModel": _make_result("BadModel", sharpe=3.5, pooled_ic=0.18, ic_stability=1.0),
        }

    def test_no_edge_says_no_edge(self):
        cfg = load_config()
        v = build_verdict(self._results_no_edge(), cfg, "GC=F")
        assert "No model" in v or "no exploitable edge" in v.lower() or "no edge" in v.lower()

    def test_no_edge_mentions_drift_sharpe(self):
        cfg = load_config()
        v = build_verdict(self._results_no_edge(), cfg, "GC=F")
        assert "0.56" in v

    def test_edge_mentions_winning_model(self):
        cfg = load_config()
        v = build_verdict(self._results_with_edge(), cfg, "GC=F")
        assert "GoodModel" in v

    def test_leakage_flag_in_verdict(self):
        cfg = load_config()
        v = build_verdict(self._results_with_leakage(), cfg, "GC=F")
        assert "LEAKAGE" in v or "leakage" in v.lower()
        assert "BadModel" in v

    def test_verdict_is_nonempty_string(self):
        cfg = load_config()
        for res_fn in (self._results_no_edge, self._results_with_edge, self._results_with_leakage):
            v = build_verdict(res_fn(), cfg, "GC=F")
            assert isinstance(v, str) and len(v) > 50

    def test_verdict_mentions_ticker(self):
        cfg = load_config()
        v = build_verdict(self._results_no_edge(), cfg, "GC=F")
        assert "GC=F" in v

    def test_verdict_mentions_period(self):
        cfg = load_config()
        v = build_verdict(self._results_no_edge(), cfg, "GC=F")
        assert cfg.dates["start"] in v or str(cfg.dates["start"])[:4] in v


# ─────────────────────────────────────────────────────────────────────────────
# Figure creation
# ─────────────────────────────────────────────────────────────────────────────

class TestFigures:
    def _sample_results(self):
        return {
            "Drift": _make_result("Drift", sharpe=0.56),
            "Model": _make_result("Model", ic_per_fold=0.05, sharpe=0.8),
        }

    def test_equity_figure_created(self):
        cfg = load_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "equity.png"
            save_equity_figure(
                self._sample_results(), _oos_index(), cfg, "GC=F", path
            )
            assert path.exists()
            assert path.stat().st_size > 5000

    def test_rolling_ic_figure_created(self):
        cfg = load_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rolling.png"
            save_rolling_ic_figure(
                self._sample_results(), _oos_index(), cfg, "GC=F", path, window=10
            )
            assert path.exists()
            assert path.stat().st_size > 5000

    def test_rolling_ic_skips_constant_predictions(self, tmp_path):
        """RandomWalk (all-zero predictions) must not crash the figure routine."""
        cfg = load_config()
        results = {
            "RandomWalk": _make_result("RandomWalk", sharpe=np.nan, pooled_ic=np.nan),
            "Model": _make_result("Model", ic_per_fold=0.05, sharpe=0.8),
        }
        # Force RandomWalk predictions to be exactly zero
        for fold in results["RandomWalk"].folds:
            fold.pred[:] = 0.0

        path = tmp_path / "rolling.png"
        save_rolling_ic_figure(results, _oos_index(), cfg, "GC=F", path, window=10)
        # File may or may not exist (skipped if all models constant); no crash
        # Model has non-constant predictions so file should exist
        assert path.exists()


# ─────────────────────────────────────────────────────────────────────────────
# write_report (integration)
# ─────────────────────────────────────────────────────────────────────────────

class TestWriteReport:
    def _run(self, tmp_path):
        cfg = load_config()
        results = {
            "Drift": _make_result("Drift", sharpe=0.56, pooled_ic=-0.003),
            "Model": _make_result("Model", ic_per_fold=0.04, sharpe=0.8, ic_stability=0.75),
        }
        idx = _oos_index()
        return write_report(
            results=results,
            oos_index=idx,
            cfg=cfg,
            ticker="GC=F",
            output_dir=tmp_path / "reports",
            figures_dir=tmp_path / "figures",
            run_date="2026-01-01",
            rolling_ic_window=10,
        ), cfg

    def test_report_file_created(self, tmp_path):
        path, _ = self._run(tmp_path)
        assert path.exists()
        assert path.suffix == ".md"

    def test_report_filename_contains_hash(self, tmp_path):
        path, cfg = self._run(tmp_path)
        h = config_hash(cfg)
        assert h in path.name

    def test_report_filename_contains_date(self, tmp_path):
        path, _ = self._run(tmp_path)
        assert "2026-01-01" in path.name

    def test_report_filename_contains_ticker(self, tmp_path):
        path, _ = self._run(tmp_path)
        assert "GC" in path.name

    def test_equity_figure_created(self, tmp_path):
        self._run(tmp_path)
        figs = list((tmp_path / "figures").glob("equity_*.png"))
        assert len(figs) == 1

    def test_rolling_ic_figure_created(self, tmp_path):
        self._run(tmp_path)
        figs = list((tmp_path / "figures").glob("rolling_ic_*.png"))
        assert len(figs) == 1

    def test_report_contains_config_hash(self, tmp_path):
        path, cfg = self._run(tmp_path)
        text = path.read_text()
        assert config_hash(cfg) in text

    def test_report_contains_verdict(self, tmp_path):
        path, _ = self._run(tmp_path)
        text = path.read_text()
        assert "## Verdict" in text

    def test_report_contains_comparison_table(self, tmp_path):
        path, _ = self._run(tmp_path)
        text = path.read_text()
        assert "Sharpe" in text
        assert "Drift" in text

    def test_report_contains_figure_links(self, tmp_path):
        path, _ = self._run(tmp_path)
        text = path.read_text()
        assert "equity_" in text
        assert "rolling_ic_" in text

    def test_report_contains_per_fold_section(self, tmp_path):
        path, _ = self._run(tmp_path)
        text = path.read_text()
        assert "## Per-fold detail" in text

    def test_two_runs_same_config_same_hash_in_filename(self, tmp_path):
        path1, cfg = self._run(tmp_path)
        # second run (different tmp subdir but same config)
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            path2 = write_report(
                results={"Drift": _make_result("Drift")},
                oos_index=_oos_index(),
                cfg=cfg,
                ticker="GC=F",
                output_dir=td / "reports",
                figures_dir=td / "figures",
                run_date="2026-06-01",
                rolling_ic_window=10,
            )
        # Same hash, different date
        assert config_hash(cfg) in path1.name
        assert config_hash(cfg) in path2.name
        assert "2026-01-01" in path1.name
        assert "2026-06-01" in path2.name
