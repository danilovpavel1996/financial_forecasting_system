# Phase 4 Summary — Targets, Baselines, First Honest Backtest

**Date:** 2026-05-30
**Status:** COMPLETE — all acceptance criteria met; 107/107 tests pass

---

## What was built

### `src/targets.py`

`forward_log_return(close, horizon=1)` — computes `log(close[t+horizon] / close[t])`.
Last `horizon` rows are NaN. Guards: `horizon >= 1`, DatetimeIndex required, all prices
must be positive.

### `src/pipeline.py`

Two entry points:

- **`load_aligned_data(cfg, target_ticker)`** — loads cached feature matrix, builds the
  forward return target, reindexes both to the same DatetimeIndex, drops warm-up NaN rows
  (from rolling windows) and trailing NaN rows (from the target horizon). Returns
  `(features_df, target_series)` fully finite.

- **`run_pipeline(cfg, target_ticker)`** — instantiates `WalkForwardSplitter.from_config`,
  `Backtester`, and runs RandomWalk / Drift / Momentum factories through the walk-forward
  loop. Returns `(results_dict, oos_index)` where `oos_index` is the DatetimeIndex for
  the concatenated OOS predictions.

### `scripts/run_backtest.py`

CLI: `python scripts/run_backtest.py [--refresh] [--ticker GC=F]`

Produces:
1. Baseline comparison table (stdout)
2. Equity curve PNG → `outputs/figures/`
3. Per-fold IC table + interpretation → `outputs/reports/`
4. Leakage warnings if pooled IC > 0.05 or Sharpe > 2.5 (none triggered)

### `tests/test_targets.py`

16 new tests covering: manual return calculation (horizon 1 and 2), zero-return for flat
prices, NaN count == horizon, single-row → all NaN, horizon/index/price validation guards,
name convention.

---

## Real-data results — GC=F, 2010-01-04 → 2024-12-30

**Aligned rows after NaN-trimming:** 3710 (59 warm-up dropped + 1 trailing target NaN)
**Splitter:** 8 folds, 3yr train (expanding), 1yr test, embargo=5d
**Cost:** 5 bps round-trip

### Baseline comparison

| model      | n_folds | mean_IC | pooled_IC | IC_stability | hit_rate | Sharpe_net | ann_ret | max_dd  | turnover |
|:-----------|--------:|--------:|----------:|-------------:|---------:|-----------:|--------:|--------:|---------:|
| RandomWalk |       8 |     NaN |       NaN |          NaN |      NaN |        NaN |   0.000 |   0.000 |     0.00 |
| Drift      |       8 |     NaN |   -0.0032 |          NaN |    0.531 |       0.56 |   0.081 |  -0.230 |     0.00 |
| Momentum   |       8 | -0.0073 |   -0.0068 |         0.38 |    0.507 |      -1.02 |  -0.147 |  -0.758 |     1.01 |

### Interpretation

Results are as expected for an efficient market at a 1-day horizon:

- **RandomWalk** — IC/Sharpe/hit_rate all NaN because `sign(0) = 0` means no positions
  are ever taken. Net P&L and turnover are exactly zero. This is correct.

- **Drift** — predicts a constant per fold (the training mean return), so per-fold IC is
  NaN (correlation with a constant is undefined). Pooled IC is -0.0032 because different
  folds have slightly different drift estimates. Net Sharpe = 0.56 reflects gold's genuine
  positive trend over 2010–2024, not an edge — a buy-and-hold would do the same.

- **Momentum** — `sign(ret_1d)` × scale predicts the direction of the last 1-day return.
  Pooled IC = -0.0068 suggests mild mean-reversion at 1-day. Sharpe = -1.02 is net of
  costs; high turnover (1.01) combined with no real edge means transaction costs dominate.
  3 of 8 folds (fold 0 IC=0.108, fold 6 IC=0.049) showed positive IC — noise, not signal.

**No leakage flags triggered.** All metrics are well within the expected range for
near-random walk behaviour.

---

## Decisions made

1. **`load_aligned_data` drops both ends in one mask.** The warm-up filter
   (`feature_df.notna().all(axis=1)`) drops the first 59 rows; the target NaN filter
   drops the last 1 row (horizon=1). Both are applied simultaneously so the outputs
   share the same index.

2. **OOS date index reconstructed from splitter.** `run_pipeline` calls
   `splitter.split(len(y))` a second time to build the `oos_index`. The splitter is
   deterministic (same call = same output), so this is safe and avoids storing index
   arrays inside `BacktestResult`.

3. **Leakage guard in script.** The script logs a WARNING if any model's pooled IC
   exceeds 0.05 or Sharpe exceeds 2.5. It does not abort — the researcher sees it and
   investigates.

4. **Equity curve uses `_equity_curve` helper (not `strategy_pnl`).** The `strategy_pnl`
   function returns aggregate stats, not a daily series. The helper replicates the position
   and cost logic identically so the plotted curve matches the reported Sharpe.

---

## Test counts

| File | Tests | Status |
|------|------:|-------|
| tests/test_config.py | 7 | PASS |
| tests/test_splitter.py | 23 | PASS |
| tests/test_metrics.py | 31 | PASS |
| tests/test_features.py | 30 | PASS |
| tests/test_targets.py | 16 | PASS |
| **Total** | **107** | **PASS** |

---

## What to watch for in Phase 5

- **LightGBM pipeline with StandardScaler inside sklearn.Pipeline.** The scaler must be
  fit on `X_train` only (the Pipeline handles this automatically when passed as
  `model_factory`). Do not normalize `X` before passing to the backtester.

- **Hyperparameter search leakage.** If grid search is used, it must run inside each
  fold's training set (e.g., `GridSearchCV(cv=TimeSeriesSplit(...))`), never across folds.

- **IC_stability as the primary quality bar.** A model with mean IC near zero but
  high IC_stability (> 0.6) may be useful; a model with high mean IC from 1 good fold
  is just noise. The per-fold table is the honest view.

- **Momentum fold 0 spike (IC=0.108).** Worth checking whether 2013 had some unusual
  autocorrelation in gold futures. If a real model also spikes in fold 0, that's the first
  place to look for leakage.

---

**Ready for Phase 5 review.**
