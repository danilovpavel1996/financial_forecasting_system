"""Portfolio-level volatility targeting for the long-short ranking strategy.

Approach: scale positions at each date by (target_vol / realized_portfolio_vol).

The portfolio vol estimate uses the trailing `lookback_days` of net P&L from
the L/S strategy, annualized. This is a realized vol estimate, not a forecast —
it looks backward only, so there is no look-ahead.

Scale factor is clipped to [0, max_leverage] (default [0, 2.0]).
During warm-up (< lookback_days observations), the scale factor is 1.0
(no position adjustment).

Feedback: we estimate vol from the scaled P&L stream. If we levered up and
realized high vol last month, we scale back down this month. This is the
intended risk-management behavior.
"""
from __future__ import annotations

import numpy as np

TRADING_DAYS_PER_YEAR = 252


class PortfolioVolTargeter:
    """Stateful vol-targeting scaler for the ranking backtester.

    Call ``update(net_pnl)`` after each day's P&L is realized, then call
    ``scale_factor()`` at the next date to get the position scale.

    Parameters
    ----------
    target_vol   : float — annualized target portfolio vol (e.g. 0.10 = 10%).
    max_leverage : float — maximum position multiplier (default 2.0).
    lookback_days: int   — trailing window for realized vol estimate.
    """

    def __init__(
        self,
        target_vol: float = 0.10,
        max_leverage: float = 2.0,
        lookback_days: int = 21,
    ) -> None:
        if not (0 < target_vol < 1):
            raise ValueError(f"target_vol must be in (0, 1), got {target_vol}")
        if max_leverage <= 0:
            raise ValueError(f"max_leverage must be > 0, got {max_leverage}")
        if lookback_days < 2:
            raise ValueError(f"lookback_days must be >= 2, got {lookback_days}")

        self.target_vol = target_vol
        self.max_leverage = max_leverage
        self.lookback_days = lookback_days
        self._pnl_history: list[float] = []

    def update(self, net_pnl: float) -> None:
        """Record the realized net P&L for this date."""
        self._pnl_history.append(float(net_pnl))

    def scale_factor(self) -> float:
        """Return the position scale factor for the NEXT date.

        During warm-up (< lookback_days obs): return 1.0 (no scaling).
        Otherwise: target_vol / trailing_realized_vol, clipped to [0, max_leverage].
        """
        n = len(self._pnl_history)
        if n < self.lookback_days:
            return 1.0

        recent = np.array(self._pnl_history[-self.lookback_days:], dtype=float)
        realized_vol = float(np.std(recent) * np.sqrt(TRADING_DAYS_PER_YEAR))

        if realized_vol < 1e-8:
            # Flat P&L → vol near zero → scale to max leverage
            return float(self.max_leverage)

        raw_scale = self.target_vol / realized_vol
        return float(np.clip(raw_scale, 0.0, self.max_leverage))

    def reset(self) -> None:
        """Clear accumulated P&L history (call between backtest runs)."""
        self._pnl_history = []
