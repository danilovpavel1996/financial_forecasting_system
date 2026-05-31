"""Forward log return target construction.

The target at time t is:

    r[t] = log(close[t + horizon] / close[t])

This is a FORWARD return — it uses prices AFTER t. Because of that, the last
`horizon` rows are NaN (no future close available), and the training harness
must never let those NaN rows straddle the train/test boundary (that is what
the embargo is for).

No price-level targets are built here. Predicting price levels yields a
spurious R² ≈ 0.99 from autocorrelation and has zero forecasting value.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def forward_log_return(close: pd.Series, horizon: int = 1) -> pd.Series:
    """Forward log return over `horizon` trading days.

    r[t] = log(close[t + horizon] / close[t])

    The last `horizon` rows are NaN because close[t + horizon] is not yet
    observed. These NaN rows must be dropped before passing the target to
    the backtester.

    Parameters
    ----------
    close:    daily Close price series with a DatetimeIndex.
    horizon:  forecast horizon in trading days (default 1).

    Returns
    -------
    pd.Series with the same index as `close`, named "fwd_ret_{horizon}d".
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    if not isinstance(close.index, pd.DatetimeIndex):
        raise TypeError("close must have a DatetimeIndex")
    if (close <= 0).any():
        raise ValueError("close contains non-positive prices; log return is undefined")

    result = np.log(close.shift(-horizon) / close)
    result.name = f"fwd_ret_{horizon}d"
    return result
