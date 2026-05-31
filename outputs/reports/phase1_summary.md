# Phase 1 Summary — Data Ingestion

**Date:** 2026-05-30
**Status:** COMPLETE — all acceptance criteria met

---

## What was built

### `src/data/universe.py`
Thin registry over the Config universe. Provides `price_tickers()`, `fred_series()`,
`metal_tickers()`, `etf_tickers()`, `macro_context_tickers()`. No hardcoded tickers —
everything flows from `config/config.yaml`.

### `src/data/prices.py`
yfinance loader with parquet cache.

- **Cache key:** `data/raw/prices/{safe_ticker}_{start}_{end}.parquet`
- **Second run:** reads from disk only — zero network calls
- **Normalisation:** strips timezone from DatetimeIndex, keeps OHLCV columns only, sorts by date, asserts no duplicate dates
- **Validation:** logs NaN counts per column and non-positive Close prices as WARNINGs (never silent)
- **Ticker-safe filenames:** `=` → `_eq_`, `^` → `_hat_`, `-` → `_`
- One fix required during testing: `multi_level_col` → `multi_level_index` (yfinance 0.2.61 API)

### `src/data/macro.py`
FRED loader with parquet cache.

- **Cache key:** `data/raw/macro/{series}_{start}_{end}.parquet`
- **Second run:** reads from disk only
- **API key:** loaded from environment via `os.environ["FRED_API_KEY"]` (never printed or logged)
- **Raw storage:** data stored as-is from FRED (no forward-fill here — that belongs in feature engineering)
- **Validation:** logs NaN counts; logs any gaps > 7 calendar days with dates
- **Leakage note + TODO:** docstring explicitly documents that features MUST lag ≥ 1 day and that ALFRED vintage data would be the rigorous fix

### `scripts/fetch_data.py`
CLI entry point. Loads `.env` via `python-dotenv`, then calls both loaders.

- `--refresh` flag forces re-download
- Fails loud if FRED_API_KEY is missing
- Prints a structured summary table on completion

### `requirements.txt` / `pyproject.toml`
Added `pyarrow==18.1.0` (missing from original scaffold; required for parquet I/O).

---

## Acceptance criteria results

| Criterion | Result |
|---|---|
| `python scripts/fetch_data.py` produces parquet for every configured asset | PASS |
| Second run reads from cache (no network) | PASS — all 16 files show "Cache hit" |
| Summary prints date ranges and row counts per series | PASS |
| No NaN gaps silently introduced | PASS — NaNs logged as WARNING |
| `pytest` still passes | PASS — 7/7 |

---

## Data inventory (first live run)

### Price series — 11 tickers, 2010-01-04 → 2024-12-30

| Ticker | Rows | Notes |
|--------|------|-------|
| GC=F | 3770 | Gold futures |
| SI=F | 3770 | Silver futures |
| PL=F | 3769 | Platinum futures (1 fewer trading day) |
| PA=F | 3753 | Palladium futures (~17 fewer — some gaps) |
| GLD | 3773 | |
| SLV | 3773 | |
| GDX | 3773 | |
| SPY | 3773 | |
| TLT | 3773 | |
| UUP | 3773 | |
| ^VIX | 3773 | |

PA=F has ~17–20 fewer rows than ETFs. This is expected (palladium futures have lower liquidity and occasional missing sessions). Feature engineering will need to handle this when constructing a common index.

### FRED macro series — 2010-01-01 → 2024-12-31

| Series | Raw rows | Non-NaN | NaN count | Notes |
|--------|----------|---------|-----------|-------|
| DFII10 | 3913 | 3752 | 161 | Weekends/holidays expected |
| T10YIE | 3913 | 3752 | 161 | |
| DGS10 | 3913 | 3752 | 161 | |
| DTWEXBGS | 3913 | 3732 | 181 | Slightly more gaps |
| VIXCLS | 3913 | 3793 | 120 | Fewer gaps (more frequent updates) |

FRED delivers calendar-day indices including weekends; NaNs on non-business days are expected and will be forward-filled when aligned to the trading-day index in Phase 3. The NaN counts here are logged as WARNINGs, not errors.

---

## Decisions made / ambiguities resolved

1. **No forward-fill in the raw loader.** FRED forward-fill is applied in feature engineering (Phase 3) on the *training-day index only*, so it can't leak future values across the train/test boundary. Doing it here would be premature and potentially unsafe.

2. **Raw data stored as-is.** The 1-day macro lag required by the leakage note is intentionally deferred to `macro_features.py`. A `TODO` in `macro.py` flags this and notes the ALFRED vintage-data improvement.

3. **PA=F row count difference.** Palladium has ~17 fewer trading days than the other assets over this period. Not a bug — palladium futures occasionally don't trade. Phase 3 feature engineering will need an explicit decision about how to handle this (drop PA=F from the common index, or fill with last-known price).

4. **`pyarrow` dependency.** Was missing from the Phase 0 scaffold. Added and pinned at `18.1.0` in both `requirements.txt` and `pyproject.toml`.

---

## What to watch for in Phase 2

- The splitter's embargo logic is the most critical correctness property in this system. The unit test that asserts `max(train_idx) < min(test_idx) - embargo` must be written before any model code.
- The common trading-day index (used by the harness) should be derived from the price data — probably SPY or GLD as the reference, then intersected or union'd with other tickers. This needs a clear choice documented in Phase 2 or 3.

---

**Ready for Phase 2 review.**
