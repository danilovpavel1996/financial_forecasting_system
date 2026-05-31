"""Cross-sectional ranking metrics.

Primary metric: Cross-Sectional Rank IC (CS-RIC)
  For each rebalancing date, compute the Spearman rank correlation between
  the model's 4 predictions and the 4 realized returns.  Average over time.

With only 4 assets per cross-section, individual-date CS-RIC is very noisy
(the maximum Spearman value with n=4 is ±1.0 but the 5% critical value for
significance is ≈ ±1.0 with two-sided p<0.05).  Significance comes from
averaging over hundreds of dates.

Additional metrics:
  - CS-RIC stability: fraction of dates with positive CS-RIC.
  - Long-short return: realized return of long-top1 / short-bottom1 per date.
  - Spread capture: L/S return as a fraction of best-minus-worst spread.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats


# ─────────────────────────────────────────────────────────────────────────────
# Core per-date CS-RIC
# ─────────────────────────────────────────────────────────────────────────────

def cs_ric_series(
    pred_df: pd.DataFrame,
    real_df: pd.DataFrame,
) -> pd.Series:
    """Per-date cross-sectional rank IC (Spearman corr of preds vs realized).

    Parameters
    ----------
    pred_df:  DataFrame, index=date, columns=assets, values=predicted returns.
    real_df:  DataFrame, index=date, columns=assets, values=realized returns.

    Returns
    -------
    pd.Series indexed by date, each value in [-1, 1] or NaN when degenerate.
    """
    shared_idx = pred_df.index.intersection(real_df.index)
    pred_df = pred_df.loc[shared_idx]
    real_df = real_df.loc[shared_idx]

    results: list[float] = []
    for date in shared_idx:
        p = pred_df.loc[date].values.astype(float)
        r = real_df.loc[date].values.astype(float)
        results.append(_spearman_safe(p, r))

    return pd.Series(results, index=shared_idx, name="cs_ric")


def _spearman_safe(p: np.ndarray, r: np.ndarray) -> float:
    """Spearman correlation; returns NaN for degenerate inputs."""
    mask = np.isfinite(p) & np.isfinite(r)
    n = mask.sum()
    if n < 2:
        return np.nan
    p_, r_ = p[mask], r[mask]
    if np.std(p_) < 1e-12 or np.std(r_) < 1e-12:
        return np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rho = float(stats.spearmanr(p_, r_)[0])
    return rho if np.isfinite(rho) else np.nan


# ─────────────────────────────────────────────────────────────────────────────
# Aggregated CS-RIC statistics
# ─────────────────────────────────────────────────────────────────────────────

def mean_cs_ric(cs_ric: pd.Series) -> float:
    """Mean CS-RIC over all dates with a finite value."""
    finite = cs_ric.dropna()
    return float(finite.mean()) if len(finite) > 0 else np.nan


def std_cs_ric(cs_ric: pd.Series) -> float:
    """Standard deviation of per-date CS-RIC."""
    finite = cs_ric.dropna()
    return float(finite.std()) if len(finite) > 1 else np.nan


def cs_ric_stability(cs_ric: pd.Series) -> float:
    """Fraction of dates with positive CS-RIC (analogous to IC-stability)."""
    finite = cs_ric.dropna()
    if len(finite) == 0:
        return np.nan
    return float((finite > 0).mean())


# ─────────────────────────────────────────────────────────────────────────────
# Long-short strategy metrics
# ─────────────────────────────────────────────────────────────────────────────

def ls_return_series(
    pos_df: pd.DataFrame,
    real_df: pd.DataFrame,
) -> pd.Series:
    """Gross L/S return per date (before transaction costs).

    pos_df:   DataFrame, index=date, columns=assets, values=positions (+1/0/-1).
    real_df:  DataFrame, index=date, columns=assets, values=realized returns.

    Returns
    -------
    pd.Series of gross P&L per date.  Normalised by gross exposure so the
    result is a per-unit-invested return (gross_exposure = |pos|.sum()).
    """
    shared = pos_df.index.intersection(real_df.index)
    pos = pos_df.loc[shared]
    real = real_df.loc[shared]

    gross_pnl = (pos * real).sum(axis=1)
    gross_exp = pos.abs().sum(axis=1).replace(0, np.nan)
    return (gross_pnl / gross_exp).rename("ls_gross_return")


def spread_capture_series(
    pos_df: pd.DataFrame,
    real_df: pd.DataFrame,
) -> pd.Series:
    """Fraction of the best-minus-worst spread captured by the L/S strategy.

    spread_capture[t] = ls_gross_return[t] / (max_realized[t] - min_realized[t])

    Returns NaN when the spread is zero (all assets had the same return).
    """
    gross_ret = ls_return_series(pos_df, real_df)
    shared = pos_df.index.intersection(real_df.index)
    real = real_df.loc[shared]

    spread = real.max(axis=1) - real.min(axis=1)
    spread = spread.replace(0, np.nan)

    return (gross_ret / spread).rename("spread_capture")


def mean_spread_capture(sc: pd.Series) -> float:
    finite = sc.dropna()
    return float(finite.mean()) if len(finite) > 0 else np.nan
