"""Evaluation metrics for forecast quality and strategy performance.

All scalar functions return float. NaN is returned (not raised) when inputs
are degenerate (too few points, constant predictions, etc.).

Strategy P&L is always computed NET of transaction costs on turnover. A model
with high IC that churns excessively will show a depressed net Sharpe — that
is the intended and correct behaviour.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy import stats


# --------------------------------------------------------------------------- #
# Forecast quality
# --------------------------------------------------------------------------- #

def _safe_corr(pred: np.ndarray, realized: np.ndarray, kind: str) -> float:
    """Correlation that returns NaN when either side is constant or too short."""
    pred = np.asarray(pred, float)
    realized = np.asarray(realized, float)
    mask = np.isfinite(pred) & np.isfinite(realized)
    n = mask.sum()
    if n < 3:
        return np.nan
    p, r = pred[mask], realized[mask]
    if np.std(p) == 0 or np.std(r) == 0:
        return np.nan
    fn = stats.pearsonr if kind == "pearson" else stats.spearmanr
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            result = float(fn(p, r)[0])
        except Exception:
            return np.nan
    return result if np.isfinite(result) else np.nan


def information_coefficient(pred: np.ndarray, realized: np.ndarray) -> float:
    """Pearson correlation between predicted and realized returns (IC).

    The primary forecast quality metric. Values near 0 are typical at a daily
    horizon; IC > 0.05 sustained is considered meaningful in practice.
    """
    return _safe_corr(pred, realized, "pearson")


def rank_ic(pred: np.ndarray, realized: np.ndarray) -> float:
    """Spearman (rank) correlation between predicted and realized returns.

    More robust than Pearson IC because it ignores the magnitude of outliers.
    This is what practitioners actually watch for signal quality.
    """
    return _safe_corr(pred, realized, "spearman")


def directional_accuracy(pred: np.ndarray, realized: np.ndarray) -> float:
    """Fraction of non-zero predictions where sign(pred) == sign(realized).

    0.5 is a coin flip. Returns NaN when no non-zero predictions exist.
    """
    pred = np.asarray(pred, float)
    realized = np.asarray(realized, float)
    mask = np.isfinite(pred) & np.isfinite(realized) & (pred != 0)
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.sign(pred[mask]) == np.sign(realized[mask])))


# --------------------------------------------------------------------------- #
# Strategy performance
# --------------------------------------------------------------------------- #

@dataclass
class StrategyStats:
    """Net-of-cost strategy statistics."""
    sharpe: float
    ann_return: float
    ann_vol: float
    max_drawdown: float
    turnover: float   # mean absolute daily position change
    hit_rate: float   # directional accuracy


def strategy_pnl(
    pred: np.ndarray,
    realized: np.ndarray,
    cost_bps: float = 5.0,
    mode: str = "sign",
    periods_per_year: int = 252,
    horizon: int = 1,
) -> StrategyStats:
    """Convert predictions to positions and compute net-of-cost performance.

    Parameters
    ----------
    pred:       forecast array.
    realized:   actual forward-return array.
    cost_bps:   round-trip transaction cost in basis points, charged on
                |Δposition| (turnover). This is where daily strategies die:
                a 55%-hit-rate signal that flips every day pays its edge
                straight to the broker.
    mode:       "sign"   → +1 / 0 / -1 binary position.
                "scaled" → position proportional to pred, clipped to [-1, 1].
    periods_per_year: annualisation factor (252 for daily).
    horizon:    forecast horizon in days. When > 1, consecutive realized values
                share (horizon-1) days of return data, inflating Sharpe by
                ~sqrt(horizon) if all observations are used. We subsample every
                horizon-th observation so only non-overlapping periods enter the
                Sharpe/vol calculation. Turnover is still computed on the full
                daily series.
    """
    pred = np.asarray(pred, float)
    realized = np.asarray(realized, float)

    if mode == "sign":
        pos = np.sign(pred)
    elif mode == "scaled":
        s = np.nanstd(pred)
        pos = np.clip(pred / s, -1.0, 1.0) if s > 0 else np.zeros_like(pred)
    else:
        raise ValueError(f"mode must be 'sign' or 'scaled', got {mode!r}")

    pos = np.nan_to_num(pos, nan=0.0)

    # Turnover: absolute change in position each day (first entry vs. 0)
    daily_turnover = np.abs(np.diff(pos, prepend=0.0))
    cost = daily_turnover * (cost_bps / 1e4)

    net = pos * realized - cost

    # Subsample to non-overlapping horizon-day blocks; annualise accordingly.
    net_eval = net[::horizon] if horizon > 1 else net
    ppy = periods_per_year / horizon

    finite_mask = np.isfinite(net_eval)
    net_clean = net_eval[finite_mask]

    mean_turnover = float(daily_turnover.mean())

    if len(net_clean) == 0:
        return StrategyStats(
            sharpe=np.nan,
            ann_return=np.nan,
            ann_vol=np.nan,
            max_drawdown=np.nan,
            turnover=mean_turnover,
            hit_rate=directional_accuracy(pred, realized),
        )

    ann_return = float(np.mean(net_clean) * ppy)
    ann_vol = float(np.std(net_clean) * np.sqrt(ppy))

    # Use a small epsilon for the zero-vol guard: np.std on truly identical
    # float64 values can return ~1e-18 (floating-point residual) rather than
    # exactly 0.  Any vol below 1e-12 is degenerate for daily returns.
    sharpe = ann_return / ann_vol if ann_vol > 1e-12 else np.nan

    # Max drawdown from equity curve
    curve = np.cumprod(1.0 + net_clean)
    peak = np.maximum.accumulate(curve)
    max_dd = float(np.min(curve / peak - 1.0))

    return StrategyStats(
        sharpe=sharpe,
        ann_return=ann_return,
        ann_vol=ann_vol,
        max_drawdown=max_dd,
        turnover=mean_turnover,
        hit_rate=directional_accuracy(pred, realized),
    )
