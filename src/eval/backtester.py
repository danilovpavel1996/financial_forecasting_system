"""Backtester orchestrator.

Runs a model_factory callable across walk-forward folds, pools out-of-sample
predictions, and produces a BacktestResult. Call comparison_table() on a dict
of results to get a side-by-side comparison that always includes baselines.

Usage pattern
-------------
    splitter = WalkForwardSplitter.from_config(cfg)
    bt = Backtester(splitter=splitter, cost_bps=cfg.cost_bps)

    factories = {
        "RandomWalk": RandomWalk,
        "Drift":      Drift,
        "MyModel":    lambda: ElasticNet(alpha=0.01),
    }
    results = {name: bt.run(X, y, fac) for name, fac in factories.items()}
    print(comparison_table(results))

model_factory must be a zero-argument callable returning a fresh model each
fold so no state leaks across folds.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from src.eval.metrics import (
    StrategyStats,
    directional_accuracy,
    information_coefficient,
    rank_ic,
    strategy_pnl,
)
from src.eval.splitter import WalkForwardSplitter
from src.models.base import Model

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Result containers
# --------------------------------------------------------------------------- #

@dataclass
class FoldResult:
    """Metrics for a single out-of-sample fold."""
    fold_idx: int
    train_size: int
    test_size: int
    ic: float
    rank_ic: float
    directional_accuracy: float
    pred: np.ndarray = field(repr=False)
    realized: np.ndarray = field(repr=False)


@dataclass
class BacktestResult:
    """Aggregated results across all folds for one model."""
    model_name: str
    folds: list[FoldResult]

    # Pooled metrics (treat all OOS predictions as one series)
    pooled_ic: float
    pooled_rank_ic: float

    # Per-fold averages
    mean_ic: float
    mean_rank_ic: float
    mean_directional_accuracy: float

    # IC stability: fraction of folds with positive IC
    ic_stability: float

    # Net-of-cost strategy stats computed on the pooled OOS series
    strategy: StrategyStats

    def n_folds(self) -> int:
        return len(self.folds)

    def oos_pred(self) -> np.ndarray:
        """All out-of-sample predictions concatenated in time order."""
        return np.concatenate([f.pred for f in self.folds])

    def oos_realized(self) -> np.ndarray:
        """All out-of-sample realised returns concatenated in time order."""
        return np.concatenate([f.realized for f in self.folds])


# --------------------------------------------------------------------------- #
# Backtester
# --------------------------------------------------------------------------- #

@dataclass
class Backtester:
    """Run walk-forward backtests.

    Parameters
    ----------
    splitter:  a configured WalkForwardSplitter instance.
    cost_bps:  round-trip transaction cost in basis points.
    pnl_mode:  "sign" or "scaled" position construction.
    horizon:   forecast horizon in days; passed to strategy_pnl for
               non-overlapping Sharpe calculation when horizon > 1.
    """

    splitter: WalkForwardSplitter
    cost_bps: float = 5.0
    pnl_mode: str = "sign"
    horizon: int = 1

    def run(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_factory: Callable[[], Model],
        model_name: str = "model",
    ) -> BacktestResult:
        """Run one model across all folds and return a BacktestResult.

        Parameters
        ----------
        X:             feature matrix, shape (n_samples, n_features).
        y:             target array, shape (n_samples,).
        model_factory: zero-arg callable returning a fresh model each fold.
        model_name:    label used in the BacktestResult.
        """
        X = np.asarray(X, float)
        y = np.asarray(y, float)

        folds_data = self.splitter.split(len(X))
        if not folds_data:
            raise ValueError(
                f"WalkForwardSplitter produced 0 folds for n={len(X)} samples. "
                "Increase the dataset size or reduce min_train/test_size."
            )

        fold_results: list[FoldResult] = []

        for k, (tr, te) in enumerate(folds_data):
            Xtr, ytr = X[tr], y[tr]
            Xte, yte = X[te], y[te]

            # Drop NaN rows from training only — never from test
            ok = np.isfinite(ytr) & np.all(np.isfinite(Xtr), axis=1)
            n_dropped = int((~ok).sum())
            if n_dropped:
                logger.debug(
                    "Fold %d: dropped %d NaN training rows out of %d",
                    k, n_dropped, len(ytr),
                )

            if ok.sum() < 2:
                logger.warning(
                    "Fold %d: only %d valid training rows; skipping fold.", k, ok.sum()
                )
                continue

            model = model_factory()
            model.fit(Xtr[ok], ytr[ok])
            pred = model.predict(Xte)

            fold_results.append(
                FoldResult(
                    fold_idx=k,
                    train_size=int(ok.sum()),
                    test_size=len(te),
                    ic=information_coefficient(pred, yte),
                    rank_ic=rank_ic(pred, yte),
                    directional_accuracy=directional_accuracy(pred, yte),
                    pred=pred,
                    realized=yte,
                )
            )
            logger.debug(
                "Fold %d: train=%d, test=%d, IC=%.4f, rank_IC=%.4f",
                k, int(ok.sum()), len(te),
                fold_results[-1].ic,
                fold_results[-1].rank_ic,
            )

        if not fold_results:
            raise ValueError(f"All folds were skipped for model {model_name!r}.")

        all_pred = np.concatenate([f.pred for f in fold_results])
        all_real = np.concatenate([f.realized for f in fold_results])

        finite_ics = [f.ic for f in fold_results if np.isfinite(f.ic)]
        finite_rics = [f.rank_ic for f in fold_results if np.isfinite(f.rank_ic)]
        finite_das = [
            f.directional_accuracy for f in fold_results
            if np.isfinite(f.directional_accuracy)
        ]

        return BacktestResult(
            model_name=model_name,
            folds=fold_results,
            pooled_ic=information_coefficient(all_pred, all_real),
            pooled_rank_ic=rank_ic(all_pred, all_real),
            mean_ic=float(np.mean(finite_ics)) if finite_ics else np.nan,
            mean_rank_ic=float(np.mean(finite_rics)) if finite_rics else np.nan,
            mean_directional_accuracy=(
                float(np.mean(finite_das)) if finite_das else np.nan
            ),
            ic_stability=(
                float(np.mean([ic > 0 for ic in finite_ics])) if finite_ics else np.nan
            ),
            strategy=strategy_pnl(all_pred, all_real, self.cost_bps, self.pnl_mode, horizon=self.horizon),
        )


# --------------------------------------------------------------------------- #
# Comparison table
# --------------------------------------------------------------------------- #

def comparison_table(results: dict[str, BacktestResult]) -> pd.DataFrame:
    """Build a side-by-side comparison DataFrame.

    Baselines must be present in `results` — the caller is responsible for
    running them. This function does not add or hide any model.
    """
    rows = []
    for name, r in results.items():
        s: StrategyStats = r.strategy
        rows.append(
            {
                "model": name,
                "n_folds": r.n_folds(),
                "mean_IC": _fmt(r.mean_ic),
                "rank_IC": _fmt(r.mean_rank_ic),
                "IC_stability": _fmt(r.ic_stability, 2),
                "pooled_IC": _fmt(r.pooled_ic),
                "hit_rate": _fmt(s.hit_rate, 3),
                "Sharpe_net": _fmt(s.sharpe, 2),
                "ann_ret": _fmt(s.ann_return, 3),
                "max_dd": _fmt(s.max_drawdown, 3),
                "turnover": _fmt(s.turnover, 3),
            }
        )
    return pd.DataFrame(rows).set_index("model")


def _fmt(x: float, decimals: int = 4) -> float:
    """Round a float; return NaN unchanged."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    return round(float(x), decimals)
