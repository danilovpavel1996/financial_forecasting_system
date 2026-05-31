"""FRED macro series loader with parquet cache.

Cache key: data/raw/macro/{series}_{start}_{end}.parquet
A second run with the same (series, start, end) reads from disk — no network.

Leakage note: FRED series are released with a publication lag and revised.
Raw data is stored as-is here. Feature engineering (macro_features.py) MUST
lag every series by at least 1 trading day before use as a feature.

TODO (point-in-time): use ALFRED vintage data for rigorous publication-lag
handling. The current approach (1-day lag in feature engineering) is a
reasonable approximation but not fully point-in-time correct.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd
from fredapi import Fred

logger = logging.getLogger(__name__)


def _cache_path(cache_dir: Path, series: str, start: str, end: str) -> Path:
    return cache_dir / "macro" / f"{series}_{start}_{end}.parquet"


def _get_api_key() -> str:
    """Read FRED API key from environment. Raises if missing."""
    key = os.environ.get("FRED_API_KEY", "").strip()
    if not key:
        raise EnvironmentError(
            "FRED_API_KEY not set. Add it to your .env file (see .env.example)."
        )
    return key


def fetch_series(
    series_id: str,
    start: str,
    end: str,
    cache_dir: Path,
    api_key: str | None = None,
    force_refresh: bool = False,
) -> pd.Series:
    """Return a daily pd.Series for one FRED series, using cache when available.

    Index is a tz-naive DatetimeIndex. The Series name is series_id.
    Missing observation dates are NOT forward-filled here — that alignment step
    happens at the feature-matrix stage so it can be done correctly on the
    training-day index only.
    """
    path = _cache_path(cache_dir, series_id, start, end)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists() and not force_refresh:
        logger.info("Cache hit: %s → %s", series_id, path)
        s = pd.read_parquet(path).squeeze()
        s.name = series_id
        _log_summary(series_id, s)
        return s

    if api_key is None:
        api_key = _get_api_key()

    logger.info("Fetching from FRED: %s (%s → %s)", series_id, start, end)
    fred = Fred(api_key=api_key)
    s = fred.get_series(series_id, observation_start=start, observation_end=end)

    if s is None or s.empty:
        raise RuntimeError(f"FRED returned no data for {series_id!r}")

    s = _normalise(series_id, s)
    _validate(series_id, s)

    # Store as single-column DataFrame for parquet compatibility
    s.to_frame().to_parquet(path)
    logger.info("Cached: %s → %s (%d rows)", series_id, path, len(s))
    _log_summary(series_id, s)
    return s


def fetch_all_series(
    series_ids: list[str],
    start: str,
    end: str,
    cache_dir: Path,
    api_key: str | None = None,
    force_refresh: bool = False,
) -> dict[str, pd.Series]:
    """Fetch every FRED series; return {series_id: Series}. Never silently skips."""
    if api_key is None:
        api_key = _get_api_key()

    result: dict[str, pd.Series] = {}
    errors: list[str] = []
    for sid in series_ids:
        try:
            result[sid] = fetch_series(sid, start, end, cache_dir, api_key, force_refresh)
        except Exception as exc:
            logger.error("FAILED %s: %s", sid, exc)
            errors.append(f"{sid}: {exc}")
    if errors:
        raise RuntimeError("FRED fetch failures:\n" + "\n".join(errors))
    return result


def _normalise(series_id: str, s: pd.Series) -> pd.Series:
    s = s.copy()
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    s.index.name = "Date"
    s.name = series_id
    s = s.sort_index()

    dupes = s.index[s.index.duplicated()]
    if len(dupes):
        raise ValueError(f"{series_id}: duplicate dates: {dupes.tolist()[:5]}")

    return s


def _validate(series_id: str, s: pd.Series) -> None:
    n_nan = s.isna().sum()
    if n_nan:
        # FRED series can have legitimate interior NaNs (data not yet released)
        # Log them so the user is aware; feature engineering will decide how to handle.
        logger.warning("%s: %d NaN values in raw series", series_id, n_nan)

    # Log any gaps > 7 calendar days (might indicate a real data hole)
    diffs = s.dropna().index.to_series().diff().dt.days.dropna()
    long_gaps = diffs[diffs > 7]
    if not long_gaps.empty:
        logger.warning(
            "%s: %d gaps > 7 days. Longest: %d days ending %s",
            series_id,
            len(long_gaps),
            int(long_gaps.max()),
            long_gaps.idxmax().date(),
        )


def _log_summary(series_id: str, s: pd.Series) -> None:
    valid = s.dropna()
    logger.info(
        "%s: %d raw obs, %d non-NaN, %s → %s",
        series_id,
        len(s),
        len(valid),
        s.index.min().date() if len(s) else "N/A",
        s.index.max().date() if len(s) else "N/A",
    )
