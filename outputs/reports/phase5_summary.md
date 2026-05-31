# Phase 5 Summary — Classical Models

**Date:** 2026-05-31
**Status:** COMPLETE — models built and tested; 139/139 tests pass.
**Critical finding: LightGBM's IC=0.15 is a timing artifact, not real edge. See §3.**

---

## What was built

### `src/models/linear.py` — `ElasticNetModel`

Wraps `sklearn.Pipeline([StandardScaler, ElasticNet])`. The scaler is fit inside
`fit()` on training data only — the Pipeline guarantees this. Key properties:

- `alpha=0.01, l1_ratio=0.5` defaults (equal L1/L2 mix, conservative regularization)
- `coef_` property back-transforms coefficients to the original unscaled feature space
- `scaler_` property exposes the fitted scaler for inspection

The leakage guard is mechanical: `predict()` applies `scaler.transform()` using
statistics from the training set; there is no code path through which test data
can reach `scaler.fit()`.

### `src/models/gbm.py` — `LightGBMModel` and `LightGBMQuantileModel`

Time-respecting early stopping. The training data is split into:

```
[0 ............ fit_end) [fit_end .... val_start) [val_start .......... n-1]
     actual fit data         inner embargo (5d)        ES validation window
```

The inner embargo (5 trading days, ≥ horizon) ensures no training label's
look-ahead window overlaps the ES validation. The outer embargo from
`WalkForwardSplitter` already separates the entire training block from the test fold.
No shuffling, no random validation subsets.

`LightGBMQuantileModel(quantile_alpha=0.9)` inherits the same ES logic with
`objective="quantile"`. Predicts the 90th-percentile of the return distribution;
high predicted upper quantile is used as a long signal.

### `tests/test_models.py`

32 new tests:
- **ElasticNet:** protocol compliance, output shape, `scaler_fit_on_train_only`
  (verifies `scaler.mean_` matches train distribution, not test), scaler unchanged
  after `predict()`, non-zero predictions on signal data, `coef_` shape.
- **LightGBM split logic:** val is the tail (not random), split sizes correct,
  inner embargo gap equals `inner_embargo`, degenerate case returns (-1,-1), fit/val
  index sets are disjoint.
- **LightGBM model:** predict before fit raises, fallback for small datasets, non-
  constant predictions on signal data.
- **Quantile model:** objective parameter set correctly, q90 predictions exceed
  regression predictions on average.
- **Integration:** all three run through `Backtester`, appear in `comparison_table`.

---

## Full comparison table — GC=F, 2010–2024, 8 folds, 5bps cost

*(as produced by `scripts/run_backtest.py` 2026-05-31)*

| model        | n_folds | mean_IC | rank_IC | IC_stab | pooled_IC | hit_rate | Sharpe_net | ann_ret | max_dd | turnover |
|:-------------|--------:|--------:|--------:|--------:|----------:|---------:|-----------:|--------:|-------:|---------:|
| RandomWalk   |       8 |     NaN |     NaN |     NaN |       NaN |      NaN |        NaN |   0.000 |  0.000 |    0.000 |
| Drift        |       8 |     NaN |     NaN |     NaN |    -0.003 |    0.531 |       0.56 |   0.081 | -0.230 |    0.000 |
| Momentum     |       8 |  -0.007 |   0.001 |    0.38 |    -0.007 |    0.507 |      -1.02 |  -0.147 | -0.758 |    1.010 |
| ElasticNet   |       8 |     NaN |     NaN |     NaN |    -0.003 |    0.531 |       0.56 |   0.081 | -0.230 |    0.000 |
| LightGBM ⚠️ |       8 |   0.170 |   0.151 |    1.00 |     0.151 |    0.551 |       1.05 |   0.150 | -0.203 |    0.961 |
| LightGBM_q90 ⚠️ |    8 |   0.109 |   0.070 |    1.00 |     0.088 |    0.531 |       0.56 |   0.081 | -0.230 |    0.000 |

⚠️ = leakage flag triggered (pooled IC > 0.05)

---

## §3 — LightGBM leakage investigation: ETF closing-time artifact

**Bottom line: LightGBM IC=0.17 is entirely explained by an ETF closing-time
mismatch. It is NOT real predictive edge.**

### The artifact

Gold futures (GC=F) close on CME at approximately 1:30 pm CT (2:30 pm ET). The
context ETFs included as features (GLD, SLV, GDX, SPY, TLT, UUP) close at
4:00 pm ET — **1.5 hours later** on the same calendar day.

When we compute `gld_ret_1d[t] = log(GLD_close_t / GLD_close_{t-1})`, we are
using the 4pm ET price on day `t`. The GC=F target is
`log(GCF_close_{t+1} / GCF_close_t)`, where `GCF_close_t` is the ~2:30pm ET
price on day `t`. The gold price movement from 2:30pm → 4pm on day `t` is
embedded in `gld_ret_1d[t]` but has not yet been "realized" in `GCF_close_t` —
it rolls over into `GCF_close_{t+1}`.

Consequence: `gld_ret_1d[t]` is a partial look-ahead into the target at `t`.
This is a data-alignment leakage, not a coding error.

### Evidence

| Check | Result |
|:------|:-------|
| `gld_ret_1d` vs. target: Pearson IC | **+0.094** |
| `ret_1d` (GC=F same-day) vs. target | −0.034 (mild mean-reversion) |
| GLD vs. GC=F same-day return correlation | 0.90 (nearly identical assets) |
| Two 90%-correlated assets, opposite-sign IC | Impossible without timing difference |
| LightGBM IC (all features) | **+0.17** |
| LightGBM IC (ETF features removed) | **−0.02** |
| IC_stability all features | 1.00 (all 8 folds positive) |
| IC_stability without ETF features | 3/8 folds positive |

The entire LightGBM IC is sourced from the 6 ETF features (GLD, SLV, GDX, SPY,
TLT, UUP). Early stopping fires after 6–81 trees because those few trees
suffice to exploit the timing artifact. The model is not "learning" anything
economically meaningful.

### ElasticNet behavior

With `alpha=0.01` (current default), ElasticNet shrinks ALL feature coefficients
to exactly zero. Predictions equal the intercept = the training mean return — the
same as Drift. Hence `Sharpe=0.56, turnover=0.000, IC=NaN` (constant prediction).

With looser regularization (`alpha=0.001`), ElasticNet achieves IC=0.31 on fold 0
— also entirely from the ETF timing artifact (drops to IC=0.02 without ETF features).

### LightGBM_q90 behavior

Always predicts a positive number (the 90th-percentile gold return is always
positive given gold's upward drift). `sign(+) = +1` for every day → always long
→ `turnover=0.000, Sharpe=0.56` (same as Drift). The pooled IC=0.088 comes from
fold-to-fold variation in the q90 estimate and also from the ETF artifact.

---

## §4 — Honest verdict

**After correcting for the ETF timing artifact:**

- **No model beats Drift's Sharpe of 0.56.**
- ElasticNet is equivalent to Drift (predicts training mean).
- LightGBM without ETF features: mean IC = −0.02, no exploitable edge.
- Momentum's Sharpe = −1.02 confirms transaction costs destroy any weak signal.

The correct Sharpe comparison is Drift = 0.56 (the unconditional long-gold
strategy over 2010–2024). No model demonstrates edge beyond buy-and-hold.

---

## §5 — Required fix before Phase 6

**The ETF 1-day return features must be lagged by one additional day.**

In `src/features/price_features.py`, context-ticker 1-day returns should use
`shift(2)` (two-day lag relative to the current row) for any ticker that closes
after GC=F — i.e., all equity-market ETFs (GLD, SLV, GDX, SPY, TLT, UUP, ^VIX)
but NOT other futures (SI=F, PL=F, PA=F), which have the same close time as GC=F.

Alternatively, the context-ticker features could use 5-day returns only (already
lagged enough to be safe at the daily horizon).

A new leakage test should be added to `tests/test_features.py` that verifies
equity-market ETF features are lagged by ≥ 2 days relative to the target close.

**This is a Phase 3 regression. Phase 6 must not proceed without this fix.**

---

## Test counts

| File | Tests | Status |
|------|------:|--------|
| tests/test_config.py | 7 | PASS |
| tests/test_splitter.py | 23 | PASS |
| tests/test_metrics.py | 31 | PASS |
| tests/test_features.py | 30 | PASS |
| tests/test_targets.py | 16 | PASS |
| tests/test_models.py | 32 | PASS |
| **Total** | **139** | **PASS** |

---

## Decisions made

1. **ElasticNet alpha=0.01 (default):** Conservative enough that all slope
   coefficients shrink to zero on a 30-feature, weak-signal regression problem.
   The right default for research starting points. Lower alpha requires
   cross-validation inside each fold (not yet implemented).

2. **LightGBM inner_embargo=5 days:** Same as the outer embargo from the
   splitter. Ensures no training label overlaps the ES window. Conservative but
   cheap.

3. **LightGBM n_estimators=500 with ES:** Early stopping fires at 6–81 trees
   on this dataset (weak signal). The cap of 500 is never reached. Fine.

4. **LightGBM_q90 always-long:** Expected for a q90 predictor on an asset with
   positive drift. The IC comes from fold-to-fold variation in q90 estimates plus
   the ETF artifact. Not a useful signal in current form.

5. **No hyperparameter tuning in Phase 5:** Tuning inside the fold requires
   inner cross-validation with another time-split. Not implemented yet. Phase 5
   goal was structural correctness, not optimized performance.

---

**Ready for Phase 5 review. Phase 6 must not start until the ETF lag fix is applied.**
