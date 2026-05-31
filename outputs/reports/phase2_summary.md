# Phase 2 Summary — Evaluation Harness

**Date:** 2026-05-30
**Status:** COMPLETE — all 4 required tests pass; 61/61 total

---

## What was built

### `src/models/base.py`
`Model` protocol (`fit`/`predict`) with `@runtime_checkable`. Implemented now (ahead of Phase 5) because the backtester requires it.

### `src/models/baselines.py`
Three baselines, all implementing the `Model` protocol:

| Baseline | Logic |
|----------|-------|
| `RandomWalk` | Always predicts 0. Hardest null hypothesis. |
| `Drift` | Predicts `nanmean(y_train)` (unconditional drift). |
| `Momentum` | Predicts `sign(X[:,-1]) * std(X_train[:,-1])` — assumes last column is a momentum proxy. |

### `src/eval/splitter.py` — `WalkForwardSplitter`

Core algorithm from the reference implementation, plus:
- **`n_splits` parameter**: takes the *last* N folds (most recent test periods), not the first. With 15 years and n_splits=8, this discards the earliest evaluation folds (smallest training sets) and keeps the later, better-trained ones.
- **`from_config()` classmethod**: converts `train_years`/`test_years` → approximate trading days (252/year) and wires in embargo and n_splits from config.
- **`n_folds(n)`** helper: counts folds without consuming the generator.
- **Rolling window mode**: `expanding=False` + `train_window` for a fixed rolling window.
- **Validation on construction**: raises `ValueError` for invalid embargo, missing train_window (rolling mode), etc.

### `src/eval/metrics.py`

| Function / Type | Description |
|-----------------|-------------|
| `information_coefficient(pred, realized)` | Pearson IC; NaN if < 3 valid points or either side is constant |
| `rank_ic(pred, realized)` | Spearman IC; same NaN guards |
| `directional_accuracy(pred, realized)` | Sign hit-rate; NaN if no non-zero predictions |
| `StrategyStats` | Dataclass: `sharpe`, `ann_return`, `ann_vol`, `max_drawdown`, `turnover`, `hit_rate` |
| `strategy_pnl(pred, realized, cost_bps, mode)` | Net-of-cost P&L; mode = "sign" or "scaled" |

One numerical fix: `np.std` on truly identical float64 values returns ~1e-18 (floating-point residual) rather than exactly 0.0. Zero-vol guard changed to `ann_vol > 1e-12` to correctly return `Sharpe=NaN` for degenerate constant-return inputs.

### `src/eval/backtester.py`

- **`FoldResult`**: per-fold IC, rank_IC, directional accuracy, raw predictions, realised returns.
- **`BacktestResult`**: aggregated over all folds — pooled IC, per-fold means, IC stability, `StrategyStats`. Helpers `oos_pred()` and `oos_realized()` concatenate the OOS series.
- **`Backtester.run(X, y, model_factory, model_name)`**: NaN rows dropped from *training only* (never from test). Logs fold-level diagnostics at DEBUG level.
- **`comparison_table(results)`**: builds a side-by-side DataFrame; caller is responsible for including baselines (no hiding or auto-adding).

---

## Acceptance criteria — Phase 2 definition of done

| Required test | Status |
|---------------|--------|
| 1. `max(train_idx) < min(test_idx) - embargo` for every fold | **PASS** (`test_embargo_gap_every_fold`, `test_embargo_gap_multiple_embargo_values`, `test_embargo_gap_exact_spacing`) |
| 2. With `embargo >= horizon`, no training label's look-ahead window overlaps test | **PASS** (`test_label_lookahead_does_not_overlap_test`, `test_label_lookahead_all_training_indices`) |
| 3. Metrics on known synthetic series match hand-computed values | **PASS** (IC=1.0/-1.0, rank_IC=0.8, DA=0/0.5/1.0, max_dd=-0.2, turnover=1.75, ann_return=0.252) |
| 4. Baselines run and appear in comparison table | **PASS** (`test_baselines_run_and_appear_in_table`) |

**Total: 61 tests, 0 failures.**

---

## Test inventory

### `tests/test_splitter.py` — 23 tests
- Embargo gap invariant (exact spacing, multiple embargo values, every fold)
- Label look-ahead: does not overlap test set (concise version + exhaustive brute-force version)
- Negative test: demonstrates that embargo=0 with horizon=1 WOULD leak
- Train precedes test, no index overlap, no cross-fold test overlap
- n_splits caps folds; selects LAST folds
- Rolling window train-size bound
- Expanding window grows, starts at 0
- Edge cases: too-small dataset, invalid args
- from_config: embargo ≥ horizon, produces folds, invariant holds on realistic dataset

### `tests/test_metrics.py` — 31 tests
- IC: perfect (1.0), anticorrelated (-1.0), constant→NaN, NaN-masked, too-few-points
- Rank IC: perfect, hand-computed (0.8), reversed (-1.0)
- Directional accuracy: 1.0, 0.0, 0.5, all-zero→NaN, mixed-zeros counted correctly
- Max drawdown: hand-computed (-0.2 exactly)
- Turnover: hand-computed (1.75 exactly)
- Annualised return: hand-computed (0.252 exactly)
- Cost always reduces Sharpe; good signal → positive Sharpe; inverse signal → negative Sharpe
- Constant-return → Sharpe=NaN (floating-point guard)
- Invalid mode raises ValueError; scaled mode clips correctly
- Backtester: baselines in table, RandomWalk predicts 0, Drift is per-fold constant, typed results, OOS length consistency, IC stability in [0,1], table shape, embargo respected end-to-end

---

## Decisions made / design choices

1. **`n_splits` takes last N folds.** The seed has no cap; it runs until data runs out. The config has `n_splits: 8`. Taking the *last* 8 folds is preferable to the *first* 8 because the later folds have more training history and the evaluation covers more recent market conditions.

2. **Baselines implemented in Phase 2.** The DoD requires "baselines run and appear in the comparison table." This requires working implementations, not stubs. They were moved up from Phase 5 where they're also listed; no code duplication introduced.

3. **`strategy_pnl` uses vol > 1e-12 threshold for Sharpe.** `np.std` on identical float64 returns a floating-point residual of ~1e-18 rather than exactly 0.0, causing a spurious huge-but-finite Sharpe instead of the correct NaN. The 1e-12 threshold is safely below any real daily-return vol (which starts ~1e-4) and handles the degenerate case correctly.

4. **NaN dropped from train only.** When the feature matrix has NaN rows (from rolling-window warm-up), only training rows are cleaned. Test rows are predicted as-is — a model that can't handle NaN test inputs is its own problem to solve at the model level.

5. **No look-ahead in the backtester's NaN handling.** The `ok` mask is computed from `y_train` and `X_train` only, never from any test data. This is consistent with the no-look-ahead principle.

---

## What to watch for in Phase 3

- **Z-score normalisation must be fit on train only.** The scaler goes inside the backtester loop (fit on `X_train`, applied to `X_test`), not globally. If any normalisation is applied before the split, it leaks test statistics into training.
- **Macro forward-fill must use only training-day values.** When aligning FRED series to a trading-day index, the forward-fill must be truncated at `train_end` — do not forward-fill across the embargo boundary.
- **PA=F has ~17 fewer rows than other tickers.** The feature matrix construction needs an explicit decision: use the intersection of all trading days, or use SPY/GLD as the reference index and forward-fill gaps for illiquid assets.

---

**Ready for Phase 3 review.**
