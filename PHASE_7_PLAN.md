# PHASE_7_PLAN.md — Cross-Sectional Ranking

> This extends PROJECT_PLAN.md. Read CLAUDE.md first — all non-negotiable
> principles still apply. This phase adds a **cross-sectional ranking** mode
> alongside the existing single-asset return-prediction system.

---

## 1. Why this is the next move

Phases 1–6 established an honest result: at daily and 5-day horizons, no model
beats buy-and-hold on gold (GC=F) after costs. LightGBM found a weak IC (~0.10
at 5-day) but turnover destroyed it in a single-asset long/short framework.

Cross-sectional ranking changes the problem fundamentally:
- **Predict relative order, not absolute returns.** "Gold beats silver" is far
  easier than "gold goes up 0.3%." The unconditional drift cancels out.
- **Market-neutral by construction.** Long the top-ranked metal, short the
  bottom-ranked. Net exposure ≈ 0, so the strategy doesn't depend on the
  overall metals market going up (which is all Drift captured).
- **Turnover is structural, not signal-dependent.** You rebalance on a fixed
  schedule (weekly), so cost is predictable and doesn't scale with noise.
- **The basket amplifies weak signals.** Even an IC of 0.05 per asset, when
  consistently picking the right rank ordering, compounds meaningfully across
  a 4-asset basket.

---

## 2. Universe

The metals basket — all already in the data pipeline:
- `GC=F` — Gold futures
- `SI=F` — Silver futures
- `PL=F` — Platinum futures
- `PA=F` — Palladium futures

Context tickers (features only, not ranked): `GLD`, `SLV`, `GDX`, `SPY`,
`TLT`, `UUP`, `^VIX`. Same as before, with the ETF late-close lag already
applied.

---

## 3. How cross-sectional ranking works

At each rebalancing date `t`:

1. **Feature matrix per asset:** build features for each of the 4 metals using
   data available at `t`. These are the same price + macro features from Phase 3,
   computed independently per asset (each metal gets its own lagged returns,
   vol, momentum, etc.), plus **cross-asset features** like the gold/silver
   ratio, relative momentum (asset momentum minus basket-average momentum),
   and relative volatility.

2. **Predict forward return per asset:** each model produces 4 predictions
   (one per metal) for the forward return over the next `horizon` days.

3. **Rank the predictions:** sort the 4 predictions. The model doesn't need to
   get the magnitude right — just the ordering.

4. **Position construction:** go long the top-ranked metal(s), short the
   bottom-ranked metal(s). Simplest version: long top-1, short bottom-1,
   equal weight. More nuanced: long top-2, short bottom-2, or weight by
   prediction spread.

5. **Evaluate with cross-sectional rank IC:** at each rebalancing date,
   compute Spearman correlation between the 4 predicted ranks and the 4
   realized-return ranks. Average over time. This is the primary metric.

---

## 4. What to build

### 4a. Cross-asset feature builder (`src/features/cross_features.py`)

New features that only exist in a cross-sectional context:
- **Relative momentum:** asset's 10d/21d momentum minus the equal-weighted
  basket average momentum. "Is this metal trending more/less than its peers?"
- **Relative volatility:** asset's rolling vol minus basket average vol.
- **Cross-ratios:** gold/silver ratio (already exists), gold/platinum ratio,
  silver/platinum ratio. These mean-revert and are classic metals-spread
  signals.
- **Relative distance from MA:** asset's dist-from-MA minus basket average.

All must be strictly lagged (data ≤ t), same as Phase 3. Use the same
shift-and-compare leakage test pattern.

### 4b. Pooled dataset builder (`src/features/pooled_dataset.py`)

Stack the 4 per-asset feature matrices into one tall DataFrame with an
`asset` column. Each row is (date, asset, features, forward_return). This is
the training format for the ranking model.

**Critical leakage rule:** the walk-forward split is on TIME, not on
asset × time. All 4 assets at date `t` are in the same fold. You cannot have
gold-on-Tuesday in training and silver-on-Tuesday in test — that leaks the
cross-sectional structure.

### 4c. Cross-sectional metrics (`src/eval/cross_metrics.py`)

New metrics specific to ranking:
- **Cross-sectional rank IC (CS-RIC):** for each date, Spearman corr of
  predicted ranks vs realized ranks across the 4 assets. Report the
  time-series mean, std, and fraction-positive (stability).
- **Long-short return:** at each rebalancing date, the return of
  (long top-1 − short bottom-1) / 2, net of costs. This is the tradeable
  P&L.
- **Spread capture:** what fraction of the actual best-minus-worst spread
  does the strategy capture? (realized L/S return) / (best − worst return).

### 4d. Ranking backtester (`src/eval/rank_backtester.py`)

Extends the existing Backtester for the ranking paradigm:
- Accepts a pooled dataset (date × asset × features × target).
- Walk-forward splits on date only (all assets move together).
- At each test date: predict returns for all 4 assets, rank them, construct
  long-short positions, compute net P&L.
- Reports CS-RIC, long-short Sharpe, spread capture, and turnover.
- Baselines: **EqualWeight** (predict equal returns → no ranking → zero P&L,
  the null), **MomentumRank** (rank by trailing 21d return), **MeanReversion**
  (rank by inverse of trailing 5d return).

### 4e. Pipeline + CLI (`src/pipeline_ranking.py`, `scripts/run_ranking.py`)

Parallel to the existing single-asset pipeline. One command:
```
python scripts/run_ranking.py [--horizon 5] [--refresh]
```
Produces a ranking comparison table + a cumulative long-short equity curve.

---

## 5. Horizons to test

Run at both `horizon=1` and `horizon=5`. The 5-day horizon is especially
interesting for ranking because:
- Rebalancing weekly means ~50 trades/year, not 252 → cost is manageable.
- Weekly relative momentum is a documented effect in commodities.
- The non-overlapping Sharpe fix from the metrics bugfix applies here too.

---

## 6. Model approach

**Start simple:** use the SAME ElasticNet and LightGBM from Phase 5, trained
on the pooled dataset. The model sees (features, asset_id_onehot) and predicts
forward return. Ranking is derived from the predictions, not built into the
loss function.

**Later (not now):** LambdaMART / LightGBM with a ranking objective
(`lambdarank`), which directly optimizes for ranking quality rather than
return-prediction accuracy. This is the "real" approach but adds complexity.
Build the evaluation first so you can compare.

---

## 7. Definition of done

- [ ] Cross-asset features built and leakage-tested (shift-and-compare).
- [ ] Pooled dataset builder with a test asserting that all 4 assets at the
      same date are in the same walk-forward fold.
- [ ] CS-RIC metric implemented and tested on synthetic data with known ranks.
- [ ] Ranking backtester produces a comparison table: baselines (EqualWeight,
      MomentumRank, MeanReversion) vs ElasticNet vs LightGBM.
- [ ] Results at both horizon=1 and horizon=5 reported honestly.
- [ ] `phase7_summary.md` states plainly which (if any) models beat the
      ranking baselines and by how much.

---

## 8. What success looks like (and what doesn't)

- **Success:** CS-RIC mean > 0.05 with stability > 0.55, AND the long-short
  Sharpe is positive after costs. Even a Sharpe of 0.3–0.5 on a market-neutral
  metals strategy would be a meaningful, non-trivial result.
- **Not success but still valuable:** CS-RIC ≈ 0 everywhere. This tells you
  the features don't carry cross-sectional information and you need different
  inputs (CFTC positioning, term structure, etc.).
- **Red flag:** CS-RIC > 0.15 consistently. Investigate for leakage — same
  discipline as before.

---

## 9. Execution order

This is ONE phase, not six. But within it, build in this order:
1. Cross-asset features + leakage tests
2. Pooled dataset builder + fold-integrity test
3. CS-RIC metric + synthetic test
4. Ranking backtester + baselines
5. Wire pipeline + CLI
6. Run, report, stop for review

Effort level: `/effort high` — the fold-integrity constraint (all assets
at same date in same fold) is a new leakage vector that needs careful
reasoning.

---

*Research tooling only — not investment advice.*
