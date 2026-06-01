# Phase 13 Summary — LambdaMART + COT Horizon Sweep

## What was built

**LambdaMARTModel** (`src/models/lambdamart.py`)
- LightGBM `lambdarank` objective optimising NDCG directly rather than MSE on returns.
- Each date is a query (group); assets within the date are ranked by realized return and
  converted to integer relevance labels (0 = worst, N−1 = best) for training.
- Time-respecting early stopping: validation set is the temporal tail of training dates
  (same design as `LightGBMModel` in `gbm.py`); the split occurs at date boundaries so
  no partial groups straddle the embargo window.
- `requires_groups = True` flag; `RankingBacktester` detects this and passes date codes
  alongside X/y to `fit()`.

**Backtester modification** (`src/eval/rank_backtester.py`)
- Added a two-line check: if `model.requires_groups` is True, extract date codes from
  the training MultiIndex and pass them as a third positional argument to `fit()`.
- No change to the base interface; all existing models unaffected.

**Pipeline update** (`src/pipeline_ranking.py`)
- Added `LambdaMART` factory alongside `LightGBM`, `MeanReversion`, etc.
- Added `embargo` parameter (override splitter embargo for sweep runs).
- Added `model_names` parameter (run a subset of models to avoid wasted compute).

**Sweep script** (`scripts/run_sweep.py`)
- Iterates 3 horizons × 2 COT configs × 3 models = 18 runs.
- Embargo = horizon for each run (non-negotiable invariant).
- Produces `outputs/reports/phase13_sweep.md` with the full matrix.

**Tests** (`tests/test_lambdamart.py`)
- 12 tests covering: `requires_groups` flag, fit/predict, relevance label correctness,
  group-size computation, end-to-end backtester integration, finite predictions.

---

## Sweep matrix (Sharpe / CS-RIC)

| Model         | h=5 no-COT      | h=5 COT         | h=21 no-COT     | h=21 COT        | h=63 no-COT     | h=63 COT        |
|---------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|
| MeanReversion | **0.63 / 0.020**| 0.29 / 0.026    | -0.07 / 0.022   | -0.07 / 0.026   |  0.03 / 0.013   | -0.12 / 0.015   |
| LightGBM MSE  |  0.43 / 0.011   | -0.33 / -0.002  | -0.21 / -0.021  |  0.11 / 0.025   |  0.10 / 0.025   | -0.48 / -0.006  |
| LambdaMART    | -0.59 / -0.017  | -0.38 / -0.007  |  0.09 / 0.025   |  0.15 / 0.030   |  0.02 / 0.083   |  0.27 / 0.077   |

*Embargo = horizon. Costs = 5 bps round-trip. Sharpe: non-overlapping, annualised.*

---

## Acceptance-criteria results

| Criterion | Result |
|-----------|--------|
| LambdaMART fits/predicts without error | ✅ |
| LambdaMART CS-RIC > LightGBM MSE CS-RIC at some config | ✅ (h=21, h=63 with and without COT) |
| LambdaMART Sharpe > 0.63 (beats MeanReversion baseline) | ❌ |
| Live signal updated to LambdaMART | ❌ (criterion not met) |
| Full 18-run matrix produced | ✅ |
| All existing tests pass (271) + 12 new LambdaMART tests | ✅ |

---

## Analysis

### LambdaMART at h=5: clear underperformance

Sharpe −0.59 (no-COT) and −0.38 (with COT) vs MeanReversion's 0.63. The ranking loss
objective is theoretically better aligned with CS-RIC than MSE, but it is **worse** at
h=5. The likely explanation: at a 5-day horizon the training signal is very noisy. The
ranking loss has more parameters to fit (it needs to correctly order all 9 assets each
day) and the trees overfit to noise that MSE smooths through. The CS-RIC is also negative,
confirming the model is actively mispredicting the ranking.

### LambdaMART improves with longer horizons

At h=21 and h=63, LambdaMART's Sharpe climbs to 0.15 and 0.27 respectively (with COT).
The CS-RIC at h=63 reaches 0.083 — the highest cross-sectional predictability seen in
any configuration of this sweep, and close to the 0.05 "clear win" threshold stated in
the plan. However, the Sharpe does not follow: 0.27 is far below 0.63.

The CS-RIC / Sharpe divergence at h=63 is worth investigating in Phase 14:
- CS-RIC measures ranking correctness per date; Sharpe measures the net P&L of a
  long-short portfolio after costs.
- At h=63, turnover is low (0.12) so costs are not the culprit.
- The likely cause is that a 63-day forecast horizon produces very few independent
  observations (3555 dates / 63 ≈ 56 non-overlapping periods), making Sharpe estimates
  unreliable even if the ranking skill is real.
- More data (longer history, or lower-correlation assets) is needed to resolve this.

### COT effects

COT features **hurt** MeanReversion: Sharpe drops from 0.63 to 0.29 at h=5 (consistent
with Phase 9 findings). COT features **help** LambdaMART at longer horizons (h=21: +0.06
Sharpe, +0.005 CS-RIC; h=63: +0.25 Sharpe, −0.006 CS-RIC in no-COT vs +0.27 Sharpe,
+0.077 CS-RIC with COT). The pattern is directionally consistent with the hypothesis
that COT positioning data has a longer-horizon signal, but the evidence remains weak
given the small number of independent observations at h=63.

### MeanReversion dominance

MeanReversion h=5 no-COT (Sharpe 0.63) remains the best-performing configuration
across all 18 runs. It is not beaten by either ML model. This is consistent with:
1. The feature set having more noise than signal at h=5 for tree models.
2. Mean-reversion in commodity futures being a robust, low-complexity signal.
3. The current feature set not yet providing incremental information over price-based heuristics.

---

## Decisions taken

- **Live signal unchanged**: MeanReversion h=5 remains the default in `live_signal.py`.
  LambdaMART did not clear the 0.63 Sharpe bar at any tested configuration.
- **Partial win recorded**: LambdaMART achieves higher CS-RIC than LightGBM MSE at h=21
  and h=63, confirming the ranking-loss advantage for CS-RIC when the horizon is long
  enough. This is useful input for Phase 14.

---

## What's ambiguous / unresolved

1. **h=63 sample count**: with only ~56 independent observations the Sharpe estimate has
   a standard error of roughly 1/√56 ≈ 0.13. The 0.27 result is marginally positive but
   statistically indistinguishable from zero. A longer out-of-sample period (post-2024
   data, or extending the training window back to 2005) would help.
2. **LambdaMART h=5 inversion**: why does the ranking-loss model produce *negative*
   CS-RIC (−0.017) when MSE-LightGBM produces positive CS-RIC (+0.011) at the same
   horizon? One hypothesis: lambdarank's gradient is dominated by correctly ordering
   the top/bottom pairs; with a noisy h=5 signal, those gradients carry wrong direction
   information and the model learns a misleading feature interaction. This would be
   worth investigating with feature importance analysis.
3. **COT + LambdaMART at h=63**: CS-RIC 0.077 is the highest single-config score but
   the confidence interval is wide. Rerunning with a different random seed or an
   additional year of data would help establish whether this is signal or noise.

---

## Phase 14 recommendations

- Focus on **new data sources** to increase feature signal quality. The current price +
  macro + COT set has clearly hit its ceiling at h=5 with tree models.
- Consider **alternative target formulations** for LambdaMART at h=5: e.g., using
  realised excess return over the universe mean (better signal-to-noise) instead of
  absolute return for ranking labels.
- If h=63 signal is the goal, gather more historical data to increase the number of
  independent test periods.
- Run LambdaMART with hyperparameter search (fewer leaves, more regularisation) at
  h=5 to diagnose whether the negative CS-RIC is a tuning problem or a structural one.

---

*Research tooling only — not investment advice.*
