"""Walk-forward ranking backtester for cross-sectional metal predictions.

Design constraints
------------------
* Walk-forward splits on DATES only.  All 4 metals on the same date are
  always in the same fold (train or test), never split across the boundary.
  This preserves the cross-sectional structure and prevents leakage.

* At each test date the model predicts a return for each of the 4 metals.
  The backtester ranks those predictions and constructs a long-short position
  (long top-n_long, short bottom-n_short).

* P&L is net of transaction costs charged on turnover (position changes).

* Sharpe uses the non-overlapping subsampling fix from metrics.py: when
  horizon > 1, observations are subsampled every horizon steps before
  computing Sharpe to avoid artificial vol reduction from overlapping returns.

Baselines
---------
* EqualWeightRanker   — predicts 0 for all assets → no ranking → flat portfolio.
* MomentumRankRanker  — ranks by trailing 21d momentum (feature column 'mom_21d').
* MeanReversionRanker — ranks by negative trailing 5d return (feature 'ret_5d').
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from src.eval.cross_metrics import (
    cs_ric_series,
    cs_ric_stability,
    ls_return_series,
    mean_cs_ric,
    mean_spread_capture,
    spread_capture_series,
    std_cs_ric,
)
from src.eval.splitter import WalkForwardSplitter
from src.eval.vol_targeting import PortfolioVolTargeter

logger = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR = 252


# ─────────────────────────────────────────────────────────────────────────────
# Simple ranking baselines
# ─────────────────────────────────────────────────────────────────────────────

class EqualWeightRanker:
    """Predict 0 for every asset → no differentiation → flat portfolio."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> "EqualWeightRanker":
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.zeros(len(X))


class MomentumRankRanker:
    """Rank assets by their trailing 21d momentum (highest → long)."""

    def __init__(self, feature_idx: int) -> None:
        self.feature_idx = feature_idx

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MomentumRankRanker":
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X[:, self.feature_idx].copy()


class MeanReversionRanker:
    """Rank assets by NEGATIVE trailing 5d return (most oversold → long)."""

    def __init__(self, feature_idx: int) -> None:
        self.feature_idx = feature_idx

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MeanReversionRanker":
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return -X[:, self.feature_idx].copy()


# ─────────────────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RankingResult:
    """Aggregated out-of-sample ranking backtest results."""
    model_name: str
    cs_ric_ts: pd.Series           # per-date CS-RIC
    ls_pnl_ts: pd.Series           # per-date net L/S P&L (after costs)
    pos_df: pd.DataFrame           # date × asset scaled positions
    pred_df: pd.DataFrame          # date × asset predictions
    real_df: pd.DataFrame          # date × asset realized returns
    n_folds: int
    horizon: int = 1               # forecast horizon for non-overlapping Sharpe
    vol_scale_ts: pd.Series | None = None   # per-date position scale factor

    # ── Derived scalars (computed once at build time) ─────────────────────
    mean_cs_ric: float = field(init=False)
    std_cs_ric: float = field(init=False)
    cs_ric_stability: float = field(init=False)
    ls_sharpe: float = field(init=False)
    ls_ann_return: float = field(init=False)
    ls_ann_vol: float = field(init=False)
    ls_max_dd: float = field(init=False)
    spread_capture: float = field(init=False)
    turnover: float = field(init=False)

    def __post_init__(self) -> None:
        self.mean_cs_ric = mean_cs_ric(self.cs_ric_ts)
        self.std_cs_ric  = std_cs_ric(self.cs_ric_ts)
        self.cs_ric_stability = cs_ric_stability(self.cs_ric_ts)

        sc = spread_capture_series(self.pos_df, self.real_df)
        self.spread_capture = mean_spread_capture(sc)

        self.turnover = _mean_turnover(self.pos_df)

        self.ls_sharpe, self.ls_ann_return, self.ls_ann_vol, self.ls_max_dd = \
            _ls_sharpe_stats_with_horizon(self.ls_pnl_ts, horizon=self.horizon)


# ─────────────────────────────────────────────────────────────────────────────
# Ranking backtester
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RankingBacktester:
    """Walk-forward cross-sectional ranking backtester.

    Parameters
    ----------
    splitter:      WalkForwardSplitter operating on the unique-date axis.
    cost_bps:      round-trip transaction cost in basis points per position change.
    horizon:       forecast horizon in trading days.  Used for non-overlapping
                   Sharpe calculation when horizon > 1.
    assets:        ordered list of asset labels (must match pooled_df's asset level).
    n_long:        number of assets to go long (highest predicted returns).
    n_short:       number of assets to go short (lowest predicted returns).
    vol_target:    annualized target portfolio vol (e.g. 0.10 = 10%).
                   None (default) disables vol targeting → fixed ±1 positions.
    max_leverage:  position scale clip upper bound (default 2.0).
    vol_lookback:  trailing days for realized vol estimate (default 21).
    """

    splitter: WalkForwardSplitter
    cost_bps: float
    horizon: int
    assets: list[str]
    n_long: int = 1
    n_short: int = 1
    vol_target: float | None = None
    max_leverage: float = 2.0
    vol_lookback: int = 21

    def run(
        self,
        pooled_df: pd.DataFrame,
        model_factory: Callable,
        model_name: str = "model",
    ) -> RankingResult:
        """Run one model across all folds and return a RankingResult.

        Parameters
        ----------
        pooled_df:     MultiIndex(date, asset) DataFrame with feature + target cols.
        model_factory: zero-arg callable returning a fresh model each fold.
        model_name:    label for the result.
        """
        feat_names = [c for c in pooled_df.columns if c != "target"]
        target_col = "target"

        unique_dates = (
            pooled_df.index.get_level_values("date").unique().sort_values()
        )
        folds = self.splitter.split(len(unique_dates))

        if not folds:
            raise ValueError(
                f"WalkForwardSplitter produced 0 folds for {len(unique_dates)} dates."
            )

        # Accumulators for OOS results
        all_dates: list[pd.Timestamp] = []
        all_pred: dict[str, list[float]] = {a: [] for a in self.assets}
        all_real: dict[str, list[float]] = {a: [] for a in self.assets}
        all_pos: dict[str, list[float]] = {a: [] for a in self.assets}
        all_net_pnl: list[float] = []
        all_vol_scale: list[float] = []
        n_folds_used = 0

        prev_pos = {a: 0.0 for a in self.assets}

        # Vol targeter (stateful; accumulates P&L across folds)
        vol_targeter = (
            PortfolioVolTargeter(
                target_vol=self.vol_target,
                max_leverage=self.max_leverage,
                lookback_days=self.vol_lookback,
            )
            if self.vol_target is not None
            else None
        )

        for k, (train_dpos, test_dpos) in enumerate(folds):
            train_dates = unique_dates[train_dpos]
            test_dates  = unique_dates[test_dpos]

            # ── Build training set ──────────────────────────────────────────
            train_mask = pooled_df.index.get_level_values("date").isin(train_dates)
            train_sub = pooled_df.loc[train_mask]

            X_train = train_sub[feat_names].values.astype(float)
            y_train = train_sub[target_col].values.astype(float)

            ok = np.isfinite(y_train) & np.all(np.isfinite(X_train), axis=1)
            if ok.sum() < 2:
                logger.warning("Fold %d: too few training rows; skipping.", k)
                continue

            model = model_factory()
            if getattr(model, "requires_groups", False):
                dates_train = train_sub.index.get_level_values("date").values
                model.fit(X_train[ok], y_train[ok], dates_train[ok])
            else:
                model.fit(X_train[ok], y_train[ok])
            n_folds_used += 1

            # ── Test: iterate through dates in chronological order ──────────
            for date in sorted(test_dates):
                try:
                    date_rows = pooled_df.xs(date, level="date")
                except KeyError:
                    continue

                # Ensure all assets present
                present = [a for a in self.assets if a in date_rows.index]
                if len(present) < len(self.assets):
                    logger.debug("Date %s: only %d assets, skipping.", date, len(present))
                    continue

                X_date = date_rows.loc[self.assets, feat_names].values.astype(float)
                y_date = date_rows.loc[self.assets, target_col].values.astype(float)

                if not np.all(np.isfinite(X_date)):
                    continue

                pred = model.predict(X_date)  # shape (n_assets,)

                # ── Construct L/S positions ─────────────────────────────────
                new_pos_arr = _construct_positions(
                    pred, self.n_long, self.n_short, len(self.assets)
                )
                new_pos = {a: new_pos_arr[i] for i, a in enumerate(self.assets)}

                # ── Vol targeting: scale positions before computing P&L ──────
                vol_scale = 1.0
                if vol_targeter is not None:
                    vol_scale = vol_targeter.scale_factor()
                    new_pos = {a: v * vol_scale for a, v in new_pos.items()}

                # ── P&L and cost ────────────────────────────────────────────
                gross = self.n_long + self.n_short
                gross_pnl = sum(new_pos[a] * y_date[i]
                                for i, a in enumerate(self.assets))
                norm_pnl = gross_pnl / gross if gross > 0 else 0.0

                turnover_t = sum(abs(new_pos[a] - prev_pos[a])
                                 for a in self.assets) / gross
                cost_t = turnover_t * (self.cost_bps / 1e4)
                net_pnl_t = norm_pnl - cost_t

                # ── Update vol targeter with realized (scaled) P&L ──────────
                if vol_targeter is not None:
                    vol_targeter.update(net_pnl_t)

                # ── Accumulate ──────────────────────────────────────────────
                all_dates.append(date)
                all_net_pnl.append(net_pnl_t)
                all_vol_scale.append(vol_scale)
                for i, a in enumerate(self.assets):
                    all_pred[a].append(float(pred[i]))
                    all_real[a].append(float(y_date[i]))
                    all_pos[a].append(float(new_pos[a]))

                prev_pos = new_pos

        if not all_dates:
            raise ValueError(f"No test observations produced for {model_name!r}.")

        # ── Assemble result DataFrames ──────────────────────────────────────
        idx = pd.DatetimeIndex(all_dates, name="date")
        pred_df = pd.DataFrame(all_pred, index=idx)
        real_df = pd.DataFrame(all_real, index=idx)
        pos_df  = pd.DataFrame(all_pos,  index=idx)
        pnl_ts  = pd.Series(all_net_pnl, index=idx, name="net_pnl")
        vol_scale_ts = pd.Series(all_vol_scale, index=idx, name="vol_scale") \
            if (all_vol_scale and self.vol_target is not None) else None

        cs_ric = cs_ric_series(pred_df, real_df)

        return RankingResult(
            model_name=model_name,
            cs_ric_ts=cs_ric,
            ls_pnl_ts=pnl_ts,
            pos_df=pos_df,
            pred_df=pred_df,
            real_df=real_df,
            n_folds=n_folds_used,
            horizon=self.horizon,
            vol_scale_ts=vol_scale_ts,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Comparison table
# ─────────────────────────────────────────────────────────────────────────────

def ranking_comparison_table(results: dict[str, RankingResult]) -> pd.DataFrame:
    """Build a side-by-side ranking comparison DataFrame."""
    rows = []
    for name, r in results.items():
        rows.append({
            "model": name,
            "n_folds": r.n_folds,
            "mean_CS_RIC": _fmt(r.mean_cs_ric, 4),
            "std_CS_RIC":  _fmt(r.std_cs_ric, 4),
            "CS_RIC_stab": _fmt(r.cs_ric_stability, 3),
            "Sharpe_net":  _fmt(r.ls_sharpe, 2),
            "ann_ret":     _fmt(r.ls_ann_return, 3),
            "max_dd":      _fmt(r.ls_max_dd, 3),
            "spr_capture": _fmt(r.spread_capture, 3),
            "turnover":    _fmt(r.turnover, 3),
        })
    return pd.DataFrame(rows).set_index("model")


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _construct_positions(
    pred: np.ndarray,
    n_long: int,
    n_short: int,
    n_assets: int,
) -> np.ndarray:
    """Return position array (+1/0/-1) from predictions.

    If all predictions are essentially identical (std < 1e-12), return
    all-zero positions (no ranking signal).
    """
    if np.std(pred) < 1e-12:
        return np.zeros(n_assets)

    pos = np.zeros(n_assets)
    order = np.argsort(pred)       # ascending: [worst, ..., best]
    pos[order[-n_long:]] = 1.0     # top n_long → long
    pos[order[:n_short]] = -1.0    # bottom n_short → short
    return pos


def _mean_turnover(pos_df: pd.DataFrame) -> float:
    """Mean absolute per-rebalancing turnover per unit of gross exposure."""
    pos = pos_df.values.astype(float)
    prev = np.vstack([np.zeros((1, pos.shape[1])), pos[:-1]])
    turnovers = np.abs(pos - prev).sum(axis=1)
    gross = pos_df.shape[1]  # heuristic: 4 assets
    return float(turnovers.mean() / gross)


def _ls_sharpe_stats_with_horizon(
    pnl: pd.Series,
    horizon: int,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> tuple[float, float, float, float]:
    """Sharpe stats with non-overlapping subsampling for horizon > 1."""
    net = pnl.dropna().values.astype(float)
    if len(net) == 0:
        return np.nan, np.nan, np.nan, np.nan

    eval_net = net[::horizon] if horizon > 1 else net
    ppy = periods_per_year / horizon

    finite = eval_net[np.isfinite(eval_net)]
    if len(finite) == 0:
        return np.nan, np.nan, np.nan, np.nan

    ann_ret = float(np.mean(finite) * ppy)
    ann_vol = float(np.std(finite) * np.sqrt(ppy))
    sharpe = ann_ret / ann_vol if ann_vol > 1e-12 else np.nan

    curve = np.cumprod(1.0 + finite)
    peak = np.maximum.accumulate(curve)
    max_dd = float(np.min(curve / peak - 1.0))

    return sharpe, ann_ret, ann_vol, max_dd


def _fmt(x: float, decimals: int = 4) -> float:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    return round(float(x), decimals)
