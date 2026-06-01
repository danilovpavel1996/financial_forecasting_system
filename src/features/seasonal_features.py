"""Seasonal features: cyclical calendar encoding.

Features:
  month_sin = sin(2π × month / 12)
  month_cos = cos(2π × month / 12)

These are identical for all assets on the same date (like macro features).
Using sine/cosine encoding instead of one-hot dummies:
  - 2 features vs 11 dummies → less overfitting
  - Captures cyclical continuity (December ≈ January in angular distance)
  - Works correctly with tree models (LightGBM) and linear models

No data source needed — computed entirely from the date index.
No look-ahead possible — calendar month at t is fully known at t.

Seasonal patterns documented in commodity markets:
  - Natural gas: demand peaks in winter (heating) and summer (cooling)
  - Corn/soybeans: supply pressure at planting (Apr–May) and harvest (Sep–Oct)
  - Gold: stronger in Q1 and Q4 (jewellery demand, central bank buying)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_seasonal_features(trading_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Build cyclical month encoding for a trading-day index.

    Parameters
    ----------
    trading_index: DatetimeIndex of trading days.

    Returns
    -------
    pd.DataFrame with columns month_sin and month_cos, indexed by trading_index.
    All values are finite (no NaN); range [-1, 1].
    """
    months = trading_index.month.astype(float)
    angle = 2.0 * np.pi * months / 12.0
    return pd.DataFrame(
        {
            "month_sin": np.sin(angle),
            "month_cos": np.cos(angle),
        },
        index=trading_index,
    )
