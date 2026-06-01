# PHASE_17_PLAN.md — Cross-Asset Ensemble + Sector Turnover Fix

> Read CLAUDE.md first. All non-negotiable principles still apply.
> Two experiments in one phase, done sequentially.

---

## Experiment A: Cross-Asset Ensemble

### Goal

Blend the commodity signal (h=63 LightGBM, Sharpe 0.79) with the equity
sector signal (h=63 MeanReversion, Sharpe 0.35) to exploit cross-asset
diversification. If the two P&L series are uncorrelated, the combined
Sharpe ≈ √(0.79² + 0.35²) ≈ 0.86.

### Method

Identical to Phase 15's P&L blending approach:
1. Run the commodity h=63 LightGBM backtest (no COT) → daily P&L series
2. Run the equity sector h=63 MeanReversion backtest → daily P&L series
3. Align to the common date range
4. Report the correlation between the two P&L series
5. Blend with weight combinations: 100/0, 75/25, 50/50, 25/75, 0/100
6. For each blend: Sharpe (subsampled every 63 days), ann_ret, max_dd

### Script

`scripts/run_cross_asset_ensemble.py` — mirrors `run_ensemble.py` but
blends across asset classes instead of across horizons.

### Success criterion

Combined Sharpe > 0.79 (beats single-asset-class best).

---

## Experiment B: Sector LightGBM Turnover Fix

### Problem

LightGBM on equity sectors has CS-RIC 0.098 (excellent ranking accuracy)
but Sharpe only 0.12 — the model changes its rankings too frequently,
and each trade costs 5bps. The signal is there; turnover destroys it.

### Three approaches to try (in order of simplicity)

#### B1: Heavier regularization

Reduce LightGBM's complexity to make predictions more stable:
```python
LightGBMModel(
    n_estimators=50,        # was 100 (fewer trees = smoother)
    num_leaves=15,          # was 31 (shallower = less overfit)
    min_child_samples=50,   # was 20 (more data per leaf = stable)
    learning_rate=0.05,     # was 0.1 (smaller steps = smoother)
    random_state=42,
)
```

Run the equity sector sweep at h=63 only with these settings.
Compare Sharpe and turnover vs the default LightGBM.

#### B2: Position smoothing filter

After LightGBM produces rankings, apply a smoothing rule: only change
positions when the new ranking DIFFERS from the current ranking by
at least K ranks. This reduces unnecessary trades from minor ranking
shifts.

Implementation: in the backtester's per-date loop, compare new_pos to
prev_pos. If the change is below a threshold (e.g., the same assets
are in top-2/bottom-2 as yesterday), keep prev_pos.

This is a simple post-prediction filter — no model change needed.

#### B3: Prediction averaging

Average the LightGBM predictions over the last N days (e.g., N=5 or
N=21) before ranking. This smooths out day-to-day prediction noise
without changing the model itself.

```python
# In the backtester, after model.predict(X_test):
smoothed_pred = rolling_mean(raw_predictions, window=21)
rankings = rank(smoothed_pred)
```

### Evaluation

For each approach (B1, B2, B3), report:
- Sharpe at h=63 on equity sectors
- CS-RIC (should stay near 0.098 — don't sacrifice accuracy for stability)
- Turnover (should decrease significantly from 0.131)
- Compare to default LightGBM (Sharpe 0.12, turnover 0.131)

### Script

`scripts/run_sector_turnover_fix.py` — runs all three variants at h=63
on equity sectors and compares.

---

## Execution order

1. Build and run `scripts/run_cross_asset_ensemble.py` → report table
2. Build and run `scripts/run_sector_turnover_fix.py` → report table
3. If either experiment produces a new best, update live signal
4. Write `phase17_summary.md` with both result tables

---

## Definition of done

- [ ] Cross-asset P&L correlation reported
- [ ] Cross-asset ensemble weight sweep table (5 combos)
- [ ] Sector turnover fix: 3 variants compared (B1, B2, B3)
- [ ] Best sector LightGBM Sharpe after fix reported
- [ ] If cross-asset ensemble > 0.79: update live signal
- [ ] `phase17_summary.md` with both experiments
- [ ] All existing tests pass

---

*Research tooling only — not investment advice.*
