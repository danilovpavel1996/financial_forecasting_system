# PHASE_13_PLAN.md — LambdaMART + COT Horizon Sweep

> Read CLAUDE.md first. All non-negotiable principles still apply.
> This is a RESEARCH phase — multiple experiments, minimal new code.

---

## 1. Why these two together

Both experiments use existing infrastructure with minimal code changes:

**LambdaMART:** LightGBM natively supports `lambdarank` as an objective.
The current LightGBM model optimizes MSE on return predictions, then we
derive rankings from the predictions. LambdaMART skips the middleman —
it directly optimizes for the ranking to be correct. Since our metric
is CS-RIC (ranking quality), training with a ranking loss should improve
CS-RIC even if the return predictions themselves are less accurate.

**COT horizon sweep:** Phase 9 showed COT features hurt at h=5 (signal
plays out over weeks, not days). We tested h=21 briefly but never with
LambdaMART, and never at h=63. With a better model (LambdaMART) and
the right horizon (monthly/quarterly), COT might finally add value.

---

## 2. Experiment 1: LambdaMART ranking model

### 2a. New model class (`src/models/lambdamart.py`)

```python
class LambdaMARTModel:
    """LightGBM with lambdarank objective for cross-sectional ranking.

    Unlike the standard LightGBM (MSE objective), this model is trained
    to produce the correct ORDERING of assets, not accurate point
    predictions of returns.

    Training requires group labels: the number of items (assets) per
    query (date). For our pooled dataset with N assets per date,
    every group has size N.
    """
```

Key implementation details:
- **Objective:** `lambdarank` (LightGBM's built-in NDCG optimizer)
- **Group labels:** the `group` parameter tells LightGBM which rows
  belong to the same ranking query. For us: each date is a query,
  each date has N=9 assets, so `group = [9, 9, 9, ...]` with one
  entry per date.
- **Relevance labels:** LambdaMART needs integer relevance labels
  (0, 1, 2, ...) not continuous returns. Convert realized returns to
  ranks within each date: rank 1 (worst) to rank 9 (best). These
  become the relevance labels the model tries to reproduce.
- **Early stopping:** use a temporal validation set (last 20% of
  training dates) with NDCG as the metric. Same time-respecting
  split as the current LightGBM.
- **Prediction:** the model outputs relevance scores, not returns.
  Higher score = model thinks this asset will outperform. Ranking
  is derived directly from these scores.

### 2b. Integration into ranking backtester

The LambdaMARTModel must conform to the same interface the
RankingBacktester expects. It needs:
- `fit(X_train, y_train)` — but also needs group information
- `predict(X_test)` → returns scores (not returns)

The cleanest approach: pass the pooled DataFrame (with the date index)
so the model can compute groups internally. Or: add an optional
`groups` parameter to `fit()`.

### 2c. Integration into ranking pipeline

Add LambdaMARTModel to `pipeline_ranking.py`'s model factories,
alongside the existing LightGBM and baselines.

### 2d. Experiment runs (h=5, no COT)

Run the ranking pipeline at h=5 with NO COT features (Phase 8
baseline) and compare:

```
MeanReversion      (baseline — Sharpe 0.63)
LightGBM MSE       (current — Sharpe 0.43)
LambdaMART         (new)
```

**Success criterion:** LambdaMART CS-RIC > LightGBM MSE CS-RIC.
Bonus: LambdaMART Sharpe > 0.63 (beats MeanReversion baseline).

---

## 3. Experiment 2: COT horizon sweep

After LambdaMART is built, run a systematic sweep:

### 3a. Horizons to test

| horizon | embargo | description |
|---------|---------|-------------|
| 5       | 5       | weekly (Phase 8 baseline) |
| 21      | 21      | monthly |
| 63      | 63      | quarterly |

### 3b. Feature configurations

For each horizon, test:
1. **No COT** (price + macro features only)
2. **With COT** (price + macro + COT features)

### 3c. Models to run at each configuration

- MeanReversion
- LightGBM (MSE)
- LambdaMART (new)

### 3d. Total runs: 3 horizons × 2 feature sets × 3 models = 18 runs

Organize results in a summary matrix:

```
              h=5 no-COT   h=5 COT   h=21 no-COT   h=21 COT   h=63 no-COT   h=63 COT
MeanRev       0.63 / 0.02  ...       ...            ...         ...            ...
LightGBM MSE  0.43 / 0.01  ...       ...            ...         ...            ...
LambdaMART    ???  / ???    ...       ...            ...         ...            ...
```

Format: Sharpe / CS-RIC

### 3e. CLI for the sweep

Build a simple script `scripts/run_sweep.py` that loops through all
configurations and produces one combined summary table. This avoids
manually editing config.yaml 18 times.

```bash
python scripts/run_sweep.py
```

Output: a markdown table saved to `outputs/reports/phase13_sweep.md`.

---

## 4. Implementation order

1. Build `src/models/lambdamart.py`
2. Add tests: LambdaMARTModel runs, produces valid rankings, fits
   without error on synthetic data
3. Integrate into `pipeline_ranking.py`
4. Run h=5 no-COT comparison: MeanReversion vs LightGBM vs LambdaMART
5. Build `scripts/run_sweep.py`
6. Run the full 18-configuration sweep
7. Write `phase13_summary.md` with the full matrix and honest analysis
8. Update the live signal to use the best model from the sweep
   (if LambdaMART beats MeanReversion, make it the new default)

---

## 5. Definition of done

- [ ] LambdaMARTModel implemented with lambdarank objective and group
      labels derived from the date index.
- [ ] Tests pass: model fits, predicts, integrates with backtester.
- [ ] h=5 comparison table: MeanReversion vs LightGBM vs LambdaMART.
- [ ] Full 18-run sweep matrix (3 horizons × 2 feature sets × 3 models).
- [ ] The best configuration is identified and stated plainly.
- [ ] If a new best is found, update the live signal default.
- [ ] All existing tests pass + new LambdaMART tests.
- [ ] `phase13_summary.md` with the complete sweep matrix.

---

## 6. What success looks like

- **Clear win:** LambdaMART CS-RIC > 0.05 with stability > 0.55 at
  some horizon, AND Sharpe > 0.63 (beats current best). Update the
  live signal to use this model.
- **Partial win:** LambdaMART improves CS-RIC but not Sharpe, or COT
  helps at h=63 but not shorter horizons. Useful information for
  Phase 14.
- **Null result:** nothing beats MeanReversion h=5 Sharpe 0.63.
  This is still valuable — it tells us the signal ceiling with
  current features, and motivates Phase 14 (new data sources).

---

*Research tooling only — not investment advice.*
