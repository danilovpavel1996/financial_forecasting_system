"""Live data fetcher — extends historical cache to TODAY.

The key difference from the backtest fetcher (scripts/fetch_data.py):
- end date = today (not the fixed 2024-12-31 backtest end)
- Cached to data/live/{date}/ so same-day re-runs are fast,
  tomorrow's run gets fresh data.

No new fetch logic is added here — this module calls the same
src/data/ fetchers used by the backtest, just with today's date.
"""
from __future__ import annotations

import datetime
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.config import Config
from src.data import universe
from src.data.cot import fetch_all_cot
from src.data.macro import fetch_all_series
from src.data.prices import fetch_all_tickers

logger = logging.getLogger(__name__)


@dataclass
class LiveData:
    prices: dict[str, pd.DataFrame]
    macro_raw: dict[str, pd.Series]
    cot_raw: dict[str, pd.DataFrame]
    as_of: datetime.date
    start: str   # historical start date for features
    end: str     # today's date


def fetch_live_data(
    cfg: Config,
    force_refresh: bool = False,
    include_cot: bool = False,
) -> LiveData:
    """Fetch all data up to today, caching within the same calendar day.

    Parameters
    ----------
    cfg:           project config (for universe and paths).
    force_refresh: bypass the same-day cache.
    include_cot:   whether to fetch COT data (slow; only needed if use_cot=True).

    Returns
    -------
    LiveData with prices/macro/COT fetched up to today.
    """
    today = datetime.date.today()
    # yfinance end is exclusive → add 1 day so today's close is included
    end_prices = (today + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    end_date   = today.strftime("%Y-%m-%d")
    start_date = cfg.dates["start"]

    # Live cache is separate from the fixed-range backtest cache
    live_cache = cfg.paths.data_raw.parent / "live" / end_date
    live_cache.mkdir(parents=True, exist_ok=True)

    all_tickers = universe.price_tickers(cfg)
    logger.info("Fetching live prices up to %s for %d tickers …", end_date, len(all_tickers))
    prices = fetch_all_tickers(
        all_tickers, start_date, end_prices, live_cache, force_refresh=force_refresh,
    )

    macro_raw: dict[str, pd.Series] = {}
    api_key = os.environ.get("FRED_API_KEY", "").strip()
    if api_key:
        logger.info("Fetching live macro up to %s …", end_date)
        try:
            macro_raw = fetch_all_series(
                universe.fred_series(cfg),
                start_date, end_date,
                live_cache,
                api_key=api_key,
                force_refresh=force_refresh,
            )
        except Exception as exc:
            logger.warning("Macro fetch failed (%s) — proceeding without macro features.", exc)
    else:
        logger.warning("FRED_API_KEY not set — macro features will be NaN in live signal.")

    cot_raw: dict[str, pd.DataFrame] = {}
    if include_cot:
        cftc_cfg = cfg.cftc
        if cftc_cfg and cftc_cfg.get("codes"):
            logger.info("Fetching live COT up to %s …", end_date)
            try:
                cot_raw = fetch_all_cot(
                    codes=cftc_cfg["codes"],
                    start=start_date,
                    end=end_date,
                    cache_dir=live_cache,
                    report_type=cftc_cfg.get("report_type", "disaggregated_fut"),
                    force_refresh=force_refresh,
                )
            except Exception as exc:
                logger.warning("COT fetch failed (%s) — proceeding without COT.", exc)

    return LiveData(
        prices=prices,
        macro_raw=macro_raw,
        cot_raw=cot_raw,
        as_of=today,
        start=start_date,
        end=end_date,
    )
