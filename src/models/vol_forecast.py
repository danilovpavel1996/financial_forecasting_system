"""Volatility forecasters: EWMA and GARCH(1,1).

EWMA (RiskMetrics, lambda=0.94):
    sigma²_t = lambda * sigma²_{t-1} + (1 - lambda) * r²_{t-1}

GARCH(1,1):
    sigma²_t = omega + alpha * r²_{t-1} + beta * sigma²_{t-1}

Both produce 1-day-ahead conditional volatility forecasts.
The forecast at date t uses ONLY returns up to date t-1 (no look-ahead).
Outputs are annualized: sigma_t_annualized = sqrt(sigma²_t * 252).

GARCH fitting is walk-forward: fit on training slice, produce OOS forecasts
for the test slice. Never use future returns for fitting.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR = 252


# ── EWMA ─────────────────────────────────────────────────────────────────────

class EWMAVolForecast:
    """Exponentially Weighted Moving Average volatility forecast.

    No fitting required — purely recursive using only past squared returns.
    The forecast sigma_t uses returns r_0, …, r_{t-1} only.

    Parameters
    ----------
    lam : float
        Decay parameter (0 < lam < 1). Default 0.94 (RiskMetrics standard).
    """

    def __init__(self, lam: float = 0.94) -> None:
        if not (0 < lam < 1):
            raise ValueError(f"lam must be in (0, 1), got {lam}")
        self.lam = lam

    def fit_predict(self, returns: pd.Series) -> pd.Series:
        """Return annualized 1-day-ahead EWMA vol for each date.

        forecast[t] uses returns[0] … returns[t-1] only.
        forecast[0] is NaN (no prior data for day 0).
        """
        r = returns.values.astype(float)
        n = len(r)
        sigma2 = np.full(n, np.nan)

        # Initialize with first squared return; forecast[0] = NaN
        if n < 2:
            return pd.Series(sigma2, index=returns.index, name="ewma_vol")

        sigma2[1] = r[0] ** 2
        for t in range(2, n):
            sigma2[t] = self.lam * sigma2[t - 1] + (1.0 - self.lam) * r[t - 1] ** 2

        ann_vol = np.sqrt(sigma2 * TRADING_DAYS_PER_YEAR)
        return pd.Series(ann_vol, index=returns.index, name="ewma_vol")


# ── GARCH(1,1) ───────────────────────────────────────────────────────────────

class GARCHVolForecast:
    """Walk-forward GARCH(1,1) 1-day-ahead volatility forecaster.

    Fitting is walk-forward: train on returns[:train_end], produce 1-step-
    ahead forecasts for returns[train_end:]. Never leaks future returns into
    the parameter estimates.

    Parameters
    ----------
    min_train_days : int
        Minimum number of daily returns needed to fit GARCH. Default 250.
    refit_every : int
        Refit the model every N days. Default 63 (quarterly).
    """

    def __init__(self, min_train_days: int = 250, refit_every: int = 63) -> None:
        self.min_train_days = min_train_days
        self.refit_every = refit_every

    def fit_predict(self, returns: pd.Series) -> pd.Series:
        """Return annualized 1-day-ahead GARCH vol for each date.

        forecast[t] is produced from a model fit on returns[:t] (no look-ahead).
        Returns NaN for dates during the initial warm-up period.
        """
        try:
            from arch import arch_model  # type: ignore
        except ImportError:
            raise ImportError("arch library required for GARCH: pip install arch")

        r = returns.values.astype(float) * 100  # scale for arch stability
        n = len(r)
        sigma_ann = np.full(n, np.nan)

        last_fit_end = -1
        omega_: Optional[float] = None
        alpha_: Optional[float] = None
        beta_:  Optional[float] = None
        sigma2_prev: float = np.nanvar(r[:self.min_train_days]) if n >= self.min_train_days else 1.0

        for t in range(self.min_train_days, n):
            needs_refit = (last_fit_end < 0) or ((t - last_fit_end) >= self.refit_every)

            if needs_refit:
                train_r = r[:t]
                try:
                    am = arch_model(train_r, vol="Garch", p=1, q=1, dist="normal")
                    res = am.fit(disp="off", show_warning=False)
                    omega_ = float(res.params["omega"])
                    alpha_ = float(res.params["alpha[1]"])
                    beta_  = float(res.params["beta[1]"])
                    sigma2_prev = float(res.conditional_volatility[-1] ** 2)
                    last_fit_end = t
                except Exception as exc:
                    logger.debug("GARCH fit failed at t=%d: %s", t, exc)
                    if omega_ is None:
                        continue

            if omega_ is not None:
                # 1-step-ahead forecast
                sigma2_next = omega_ + alpha_ * r[t - 1] ** 2 + beta_ * sigma2_prev
                sigma2_prev = sigma2_next
                # Unscale (returns were scaled by 100) and annualize
                sigma_ann[t] = np.sqrt(sigma2_next * TRADING_DAYS_PER_YEAR) / 100.0

        return pd.Series(sigma_ann, index=returns.index, name="garch_vol")


# ── Realized vol (evaluation benchmark) ─────────────────────────────────────

def realized_vol(returns: pd.Series, window: int = 21) -> pd.Series:
    """Trailing annualized realized volatility (rolling std).

    Uses returns[t-window:t] to produce the estimate at t, so it is
    backward-looking only (no look-ahead).
    """
    ann = returns.rolling(window, min_periods=max(5, window // 2)).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    return ann.rename("realized_vol")


def vol_forecast_corr(
    forecast: pd.Series,
    returns: pd.Series,
    proxy_window: int = 21,
) -> float:
    """Pearson correlation between vol forecast and trailing realized vol.

    Uses realized_vol() as the proxy for realized volatility.
    Both series are aligned on their common non-NaN index.
    """
    rv = realized_vol(returns, window=proxy_window)
    shared = forecast.index.intersection(rv.index)
    f = forecast.reindex(shared).dropna()
    r = rv.reindex(f.index).dropna()
    aligned_f = f.reindex(r.index).dropna()
    aligned_r = r.reindex(aligned_f.index)

    if len(aligned_f) < 30:
        return np.nan
    return float(np.corrcoef(aligned_f.values, aligned_r.values)[0, 1])
