"""CFTC Commitment of Traders (COT) data loader — Disaggregated Futures-Only.

Cache key: data/raw/cot/cot_disaggregated_{start_year}_{end_year}.parquet

The report covers positions as-of each Tuesday; the CFTC releases the data
on Friday at 3:30pm ET.  Each row is tagged with release_date = report_date + 3
calendar days (Tuesday → Friday).  Feature engineering lags by 1 additional
trading day so the feature at date t uses only reports released strictly before t.

Column name variation across years (discovered empirically, logged per year):
  - 2010–2013: 'Report_Date_as_MM_DD_YYYY'  (cot_reports parses to ISO string)
  - 2014+:     'Report_Date_as_YYYY-MM-DD'

All key numeric columns (M_Money_Positions_Long_All, M_Money_Positions_Short_All,
Prod_Merc_Positions_Long_All, Prod_Merc_Positions_Short_All, Open_Interest_All)
are stable across all years.

No API key required — the cot_reports library downloads official CFTC zip files.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ── Column name constants (verified against actual data) ─────────────────────

# Stable numeric columns across all years (verified 2010–2024)
_COL_MM_LONG  = "M_Money_Positions_Long_All"
_COL_MM_SHORT = "M_Money_Positions_Short_All"
_COL_PM_LONG  = "Prod_Merc_Positions_Long_All"
_COL_PM_SHORT = "Prod_Merc_Positions_Short_All"
_COL_OI       = "Open_Interest_All"
_COL_CODE     = "CFTC_Contract_Market_Code"
_COL_NAME     = "Market_and_Exchange_Names"

# Date column changed in 2014 (cot_reports already parses both as ISO strings)
_DATE_COL_NEW = "Report_Date_as_YYYY-MM-DD"   # 2014+
_DATE_COL_OLD = "Report_Date_as_MM_DD_YYYY"   # 2010–2013

# Raw output column names (our canonical naming)
RAW_COLS = ["mm_long", "mm_short", "prod_long", "prod_short", "oi"]


# ── Keyword fragments for name-based fallback matching ───────────────────────
_NAME_KEYWORDS: Dict[str, str] = {
    "GC=F": "GOLD",
    "SI=F": "SILVER",
    "PL=F": "PLATINUM",
    "PA=F": "PALLADIUM",
    "CL=F": "CRUDE OIL",
    "NG=F": "NAT GAS",
    "HG=F": "COPPER",
    "ZC=F": "CORN",
    "ZS=F": "SOYBEAN",
}


# ── Public API ───────────────────────────────────────────────────────────────

def fetch_all_cot(
    codes: Dict[str, str],
    start: str,
    end: str,
    cache_dir: Path,
    report_type: str = "disaggregated_fut",
    force_refresh: bool = False,
) -> Dict[str, pd.DataFrame]:
    """Fetch Disaggregated COT data for all tickers; cache by year range.

    Parameters
    ----------
    codes:        {ticker: cftc_code}, e.g. {"GC=F": "088691"}.
    start:        start date string "YYYY-MM-DD".
    end:          end date string "YYYY-MM-DD".
    cache_dir:    project data_raw Path.
    report_type:  cot_reports report type identifier.
    force_refresh: ignore parquet cache and re-download.

    Returns
    -------
    dict mapping ticker → pd.DataFrame with:
      - DatetimeIndex = release_date (Friday, tz-naive)
      - columns: mm_long, mm_short, prod_long, prod_short, oi  (float)

    Missing tickers are logged and omitted from the result (not an error).
    """
    start_year = pd.Timestamp(start).year
    end_year   = pd.Timestamp(end).year

    cache_path = _cache_path(cache_dir, report_type, start_year, end_year)
    raw = _load_or_fetch(cache_path, report_type, start_year, end_year, force_refresh)

    result: Dict[str, pd.DataFrame] = {}
    for ticker, code in codes.items():
        df = _extract_ticker(raw, ticker, code, start, end)
        if df is not None and not df.empty:
            result[ticker] = df
        else:
            logger.warning("COT: no data found for ticker %s (code=%s)", ticker, code)

    _log_coverage(result, start, end)
    return result


# ── Internals ────────────────────────────────────────────────────────────────

def _cache_path(cache_dir: Path, report_type: str, start_year: int, end_year: int) -> Path:
    p = cache_dir / "cot"
    p.mkdir(parents=True, exist_ok=True)
    return p / f"cot_{report_type}_{start_year}_{end_year}.parquet"


def _load_or_fetch(
    cache_path: Path,
    report_type: str,
    start_year: int,
    end_year: int,
    force_refresh: bool,
) -> pd.DataFrame:
    """Return combined slim DataFrame (essential columns only) for all years."""
    if cache_path.exists() and not force_refresh:
        logger.info("COT cache hit: %s", cache_path)
        return pd.read_parquet(cache_path)

    import cot_reports as cot  # deferred to avoid hard import at module level

    frames: list[pd.DataFrame] = []
    for year in range(start_year, end_year + 1):
        try:
            df = cot.cot_year(year=year, cot_report_type=report_type)
            date_col = _resolve_date_col(df, year)
            logger.info(
                "COT year=%d: %d rows, date_col=%r, numeric_cols=%s",
                year, len(df), date_col,
                [c for c in [_COL_MM_LONG, _COL_MM_SHORT, _COL_PM_LONG, _COL_PM_SHORT, _COL_OI]
                 if c in df.columns],
            )
            # Keep only the 8 columns we need — avoids mixed-type parquet issues
            slim = _slim_columns(df, date_col)
            frames.append(slim)
        except Exception as exc:
            logger.warning("COT year=%d: download failed (%s) — skipping", year, exc)

    if not frames:
        raise RuntimeError(f"COT: all years {start_year}–{end_year} failed to download.")

    combined = pd.concat(frames, ignore_index=True)
    logger.info("COT combined: %d slim rows across %d years", len(combined), len(frames))

    combined.to_parquet(cache_path)
    logger.info("COT cached: %s", cache_path)
    return combined


def _slim_columns(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Keep only the 8 essential columns and cast numerics; avoids mixed-type parquet errors."""
    out = pd.DataFrame({
        "_report_date": df[date_col].astype(str),
        _COL_CODE:      df[_COL_CODE].astype(str),
        _COL_NAME:      df[_COL_NAME].astype(str),
        _COL_MM_LONG:   pd.to_numeric(df[_COL_MM_LONG],  errors="coerce"),
        _COL_MM_SHORT:  pd.to_numeric(df[_COL_MM_SHORT], errors="coerce"),
        _COL_PM_LONG:   pd.to_numeric(df[_COL_PM_LONG],  errors="coerce"),
        _COL_PM_SHORT:  pd.to_numeric(df[_COL_PM_SHORT], errors="coerce"),
        _COL_OI:        pd.to_numeric(df[_COL_OI],       errors="coerce"),
    })
    return out


def _resolve_date_col(df: pd.DataFrame, year: int) -> str:
    """Return the correct date column name for the given DataFrame/year."""
    if _DATE_COL_NEW in df.columns:
        return _DATE_COL_NEW
    if _DATE_COL_OLD in df.columns:
        return _DATE_COL_OLD
    # Fallback: scan for any column with 'date' (case-insensitive) except YYMMDD
    for col in df.columns:
        low = col.lower()
        if "date" in low and "yymmdd" not in low and "as_of" not in low:
            logger.warning(
                "COT year=%d: using fallback date column %r (neither standard name found)",
                year, col,
            )
            return col
    raise ValueError(
        f"COT year={year}: no recognised date column found. "
        f"Columns: {df.columns.tolist()[:10]}"
    )


def _extract_ticker(
    raw: pd.DataFrame,
    ticker: str,
    code: str,
    start: str,
    end: str,
) -> Optional[pd.DataFrame]:
    """Filter raw combined DataFrame to one ticker; return release_date-indexed DataFrame."""
    code_stripped = code.strip()

    # Primary: match by CFTC code
    mask = raw[_COL_CODE].astype(str).str.strip() == code_stripped
    sub = raw[mask].copy()

    if sub.empty:
        # Fallback: name-based substring match
        keyword = _NAME_KEYWORDS.get(ticker, "")
        if keyword:
            name_mask = raw[_COL_NAME].str.upper().str.contains(keyword, na=False)
            sub = raw[name_mask].copy()
            if not sub.empty:
                found_name = sub[_COL_NAME].iloc[0]
                logger.info(
                    "COT %s: code %s not found; matched by name keyword %r → %r",
                    ticker, code, keyword, found_name,
                )
            else:
                return None
        else:
            return None
    else:
        found_name = sub[_COL_NAME].iloc[0] if _COL_NAME in sub.columns else "?"
        logger.info("COT %s: code %s → %r", ticker, code, found_name)

    # Parse report date (Tuesday as-of date) — stored as "_report_date" in slim format
    report_dates = pd.to_datetime(sub["_report_date"])

    # Release date = report_date + 3 calendar days (Tuesday → Friday)
    release_dates = report_dates + pd.Timedelta(days=3)

    sub = sub.copy()
    sub.index = release_dates.values
    sub.index = pd.DatetimeIndex(sub.index).normalize()
    sub.index.name = "release_date"

    # Select and rename numeric columns
    try:
        out = pd.DataFrame(
            {
                "mm_long":   pd.to_numeric(sub[_COL_MM_LONG],  errors="coerce"),
                "mm_short":  pd.to_numeric(sub[_COL_MM_SHORT], errors="coerce"),
                "prod_long": pd.to_numeric(sub[_COL_PM_LONG],  errors="coerce"),
                "prod_short":pd.to_numeric(sub[_COL_PM_SHORT], errors="coerce"),
                "oi":        pd.to_numeric(sub[_COL_OI],       errors="coerce"),
            },
            index=sub.index,
        )
    except KeyError as exc:
        logger.error("COT %s: missing expected column: %s", ticker, exc)
        return None

    # Drop duplicates (keep latest if a date appears twice from overlapping year files)
    out = out[~out.index.duplicated(keep="last")].sort_index()

    # Filter to configured date range (use report release dates)
    start_ts = pd.Timestamp(start)
    end_ts   = pd.Timestamp(end)
    out = out.loc[(out.index >= start_ts) & (out.index <= end_ts)]

    n_nan = out.isnull().any(axis=1).sum()
    logger.info(
        "COT %s: %d weekly obs, %s → %s, %d NaN rows",
        ticker,
        len(out),
        out.index.min().date() if len(out) else "N/A",
        out.index.max().date() if len(out) else "N/A",
        n_nan,
    )
    return out


def _log_coverage(result: Dict[str, pd.DataFrame], start: str, end: str) -> None:
    total_years = pd.Timestamp(end).year - pd.Timestamp(start).year + 1
    expected_weeks = total_years * 52

    logger.info("=== COT coverage summary ===")
    for ticker, df in sorted(result.items()):
        pct = 100 * len(df) / expected_weeks if expected_weeks else 0
        logger.info(
            "  %s: %d obs (%.0f%% of %d expected), %s → %s",
            ticker, len(df), pct, expected_weeks,
            df.index.min().date() if len(df) else "N/A",
            df.index.max().date() if len(df) else "N/A",
        )
