# Phase 3 Summary — Feature Engineering

**Date:** 2026-05-30
**Status:** COMPLETE — all acceptance criteria met; 91/91 tests pass

---

## What was built

### `src/features/price_features.py`

Low-level building blocks (all operate on `pd.Series` of Close prices):

| Function | Description | Columns produced |
|----------|-------------|-----------------|
| `log_returns(close)` | Daily log return: `log(close[t]/close[t-1])` | 1 series |
| `lagged_returns(close, lags)` | Multi-horizon returns | `ret_1d`, `ret_5d`, `ret_10d`, `ret_21d` |
| `rolling_volatility(close, windows)` | Trailing std × √252 | `vol_5d`, `vol_21d` |
| `rolling_momentum(close, windows)` | Trailing mean daily return | `mom_10d`, `mom_21d` |
| `distance_from_ma(close, windows)` | `(close - MA_k) / MA_k` | `dist_ma_20`, `dist_ma_60` |
| `log_ratio(close_a, close_b, name)` | `log(a/b)` cross-asset ratio | `gold_silver_ratio` |

High-level builder: `build_price_features(prices, target_ticker, context_tickers, ratio_pair)` — assembles all of the above plus 1-day return for each context ticker.

### `src/features/macro_features.py`

| Function | Description |
|----------|-------------|
| `align_to_trading_index(raw, trading_index)` | Forward-fills calendar-day FRED series to trading-day index (max 5-day gap) |
| `lag_series(series, lag=1)` | Shifts by `lag` trading days; raises if lag < 1 |
| `daily_change(series)` | First difference |
| `build_macro_features(macro, trading_index)` | Assembles all macro features with lag applied |

Macro features produced (all lagged 1 trading day):
- `dfii10_level`, `dfii10_chg_1d` — 10Y TIPS real yield
- `t10yie_level`, `t10yie_chg_1d` — 10Y breakeven inflation
- `dgs10_chg_1d` — 10Y nominal yield change
- `dxy_level`, `dxy_chg_1d` — trade-weighted USD index
- `vix_level`, `vix_chg_1d` — VIX

### `src/features/__init__.py` — `build_feature_matrix(cfg, target_ticker)`

Orchestrates the full build: loads cached raw data → price features → macro features → join on common index → assert no duplicates, assert index alignment → cache to `data/processed/`.

---

## Acceptance criteria results

| Criterion | Status |
|-----------|--------|
| shift-and-compare leakage test — price features (drop 1/5/10/21 rows) | **PASS** |
| shift-and-compare leakage test — macro features (drop 1/5/10 rows) | **PASS** |
| Feature matrix index == target ticker's trading-day index (1:1) | **PASS** |
| `data/processed/` holds cached feature matrix | **PASS** |
| 91/91 tests pass (all phases) | **PASS** |

---

## Real-data feature matrix (GC=F target, 2010-01-04 → 2024-12-30)

```
Shape: 3770 rows × 30 columns
Rows with ALL features valid: 3711
Rows with any NaN (warm-up): 59
```

### NaN profile — warm-up expected, no mid-series gaps

| Feature | NaN count | First valid |
|---------|-----------|-------------|
| `ret_1d`, `sif_ret_1d`, ..., all 1-day returns | 1 | 2010-01-05 |
| `ret_5d`, `vol_5d` | 5 | 2010-01-11 |
| `ret_10d`, `mom_10d` | 10 | 2010-01-19 |
| `ret_21d`, `vol_21d`, `mom_21d` | 21 | 2010-02-03 |
| `dist_ma_20` | 19 | 2010-02-01 |
| `dist_ma_60` | 59 | 2010-03-30 ← longest warm-up |
| `gold_silver_ratio` | 0 | 2010-01-04 |
| Macro level features (lagged 1d) | 1 | 2010-01-05 |
| Macro change features (lagged 2d) | 2 | 2010-01-06 |

59 NaN rows total — determined by the 60-day MA warm-up period. All 3711 post-warm-up rows are fully valid.

---

## Decisions made / design choices

1. **No normalization anywhere.** This is enforced by design and mechanically checked by `test_no_normalization_in_features`. Z-scoring belongs inside the model pipeline (Phase 5), fit on training data only.

2. **Rolling windows use `min_periods=window`.** Partial windows are not used — a feature at day t only appears when the full window of past data is available. This avoids noisy estimates from the warm-up period.

3. **GC=F as the reference index.** The feature matrix uses GC=F's 3770-row trading-day index. ETFs (GLD, SPY, etc.) have 3 extra trading days that GC=F is missing — those days are simply absent from the feature matrix (they're not gold futures trading days). This is correct: we're forecasting GC=F returns.

4. **Macro forward-fill limit = 5 calendar days.** FRED series typically have values on every business day; gaps > 5 days likely indicate a genuine data hole. Any longer gap is left as NaN and logged. A 5-day limit safely covers 3-day holiday weekends.

5. **Lag enforced at construction with ValueError guard.** `lag_series(s, lag=0)` raises immediately. This prevents accidental same-day use of FRED values (the most likely leakage scenario for macro data).

6. **Leakage test uses `check_exact=True`.** Rolling operations on identical inputs produce bitwise-identical results when no future rows are present. Any difference is real look-ahead bias, not float noise.

7. **Context ticker columns:** 1-day returns only for context assets (SPY, TLT, UUP, GLD, SLV, GDX, PL=F, PA=F, SI=F, ^VIX). These are forward-filled to GC=F's index for the 3 missing days. Richer cross-asset features (momentum, vol) can be added in later phases without touching the leakage tests.

---

## What to watch for in Phase 4

- **Normalization inside the backtester loop.** When wiring `pipeline.py`, a `StandardScaler` must be fit ONLY on `X_train` and applied to `X_test`. A `sklearn.Pipeline` object with `StandardScaler → model` handles this correctly — pass the pipeline as the `model_factory`.
- **NaN rows from warm-up**: The backtester's `ok` mask already drops NaN training rows. But the 59 warm-up rows should be dropped from the full feature/target DataFrame before passing to the backtester, so early folds have cleaner training data.
- **Target alignment**: `forward_log_return` will produce NaN for the last `horizon` rows. These must be dropped together with the feature matrix before splitting.

---

**Ready for Phase 4 review.**
