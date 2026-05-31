"""yfinance price loader with parquet cache.

Cache key: data/raw/prices/{ticker}_{start}_{end}.parquet
A second run with the same (ticker, start, end) reads from disk — no network.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

OHLCV_COLS = ["Open", "High", "Low", "Close", "Volume"]


def _cache_path(cache_dir: Path, ticker: str, start: str, end: str) -> Path:
    safe = ticker.replace("=", "_eq_").replace("^", "_hat_").replace("-", "_")
    return cache_dir / "prices" / f"{safe}_{start}_{end}.parquet"


def fetch_ticker(
    ticker: str,
    start: str,
    end: str,
    cache_dir: Path,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Return daily OHLCV DataFrame for one ticker, using cache when available.

    Returns a DataFrame with a tz-naive DatetimeIndex (date only) and columns
    [Open, High, Low, Close, Volume]. Raises RuntimeError if no data comes back.
    """
    path = _cache_path(cache_dir, ticker, start, end)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists() and not force_refresh:
        logger.info("Cache hit: %s → %s", ticker, path)
        df = pd.read_parquet(path)
        _log_summary(ticker, df)
        return df

    logger.info("Fetching from yfinance: %s (%s → %s)", ticker, start, end)
    raw = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        multi_level_index=False,
    )

    if raw is None or raw.empty:
        raise RuntimeError(f"yfinance returned no data for {ticker!r}")

    # Normalise to tz-naive date index and keep OHLCV only
    df = _normalise(raw, ticker)
    _validate(ticker, df)
    df.to_parquet(path)
    logger.info("Cached: %s → %s (%d rows)", ticker, path, len(df))
    _log_summary(ticker, df)
    return df


def fetch_all_tickers(
    tickers: list[str],
    start: str,
    end: str,
    cache_dir: Path,
    force_refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    """Fetch every ticker; return {ticker: DataFrame}. Never silently skips."""
    result: dict[str, pd.DataFrame] = {}
    errors: list[str] = []
    for ticker in tickers:
        try:
            result[ticker] = fetch_ticker(ticker, start, end, cache_dir, force_refresh)
        except Exception as exc:
            logger.error("FAILED %s: %s", ticker, exc)
            errors.append(f"{ticker}: {exc}")
    if errors:
        raise RuntimeError("Price fetch failures:\n" + "\n".join(errors))
    return result


def _normalise(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Strip timezone, keep OHLCV, rename to canonical columns."""
    df = raw.copy()

    # yfinance sometimes returns MultiIndex columns when a single ticker is passed
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    present = [c for c in OHLCV_COLS if c in df.columns]
    missing = [c for c in OHLCV_COLS if c not in df.columns]
    if missing:
        logger.warning("%s: missing columns %s — will be NaN", ticker, missing)

    df = df[present].copy()
    for col in missing:
        df[col] = float("nan")
    df = df[OHLCV_COLS]

    # Drop timezone from index
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df.index.name = "Date"
    df = df.sort_index()

    # Assert no duplicate dates
    dupes = df.index[df.index.duplicated()]
    if len(dupes):
        raise ValueError(f"{ticker}: duplicate dates in price data: {dupes.tolist()[:5]}")

    return df


def _validate(ticker: str, df: pd.DataFrame) -> None:
    """Log NaN gaps without silently swallowing them."""
    for col in OHLCV_COLS:
        n_nan = df[col].isna().sum()
        if n_nan:
            logger.warning("%s.%s: %d NaN values", ticker, col, n_nan)

    # Negative or zero close prices are a data-quality red flag
    if (df["Close"] <= 0).any():
        n_bad = (df["Close"] <= 0).sum()
        logger.warning("%s: %d non-positive Close prices", ticker, n_bad)


def _log_summary(ticker: str, df: pd.DataFrame) -> None:
    logger.info(
        "%s: %d rows, %s → %s",
        ticker,
        len(df),
        df.index.min().date(),
        df.index.max().date(),
    )
