# Phase 8 Summary — 9-Asset Commodity Basket

**Date:** 2026-05-31  
**Assets:** GC=F, SI=F, PL=F, PA=F, CL=F, NG=F, HG=F, ZC=F, ZS=F (9-asset basket)  
**Period:** 2010-01-01 – 2024-12-31  **Folds:** 8 walk-forward  **Cost:** 5 bps round-trip  
**Long/short:** top-2 / bottom-2 (vs top-1/bottom-1 in Phase 7)

---

## What was built

| Component | Change |
|-----------|--------|
| `config/config.yaml` | Added `universe.ranked_assets` with 9 tickers (metals, energy, industrial, agricultural) |
| `src/data/universe.py` | Added `ranked_tickers(cfg)` helper; `price_tickers()` now returns ranked ∪ context |
| `src/features/cross_features.py` | Docstring update — already N-asset generic, parameter renamed `basket_tickers` |
| `src/features/pooled_dataset.py` | Full rewrite: N-asset generic, dynamic one-hot map, non-positive close sanitization |
| `src/pipeline_ranking.py` | Uses `ranked_tickers()`; auto-selects n_long=2/n_short=2 for N≥6 |
| `scripts/fetch_data.py` | No change — already ticker-agnostic via `price_tickers()` |
| Tests | +7 new tests (224 total, all pass) |

---

## Data coverage

All 9 ranked tickers fetched 2010-01-04 – 2024-12-30.  
No ticker has < 2000 rows — all are >> threshold.

| Ticker | Rows | Notes |
|--------|------|-------|
| GC=F   | 3770 | — |
| SI=F   | 3770 | — |
| PL=F   | 3769 | — |
| PA=F   | 3753 | Lowest count; still >> 2000 |
| CL=F   | 3771 | **1 non-positive close** (2020-04-20, WTI negative-price event); replaced with NaN, date excluded |
| NG=F   | 3772 | — |
| HG=F   | 3771 | — |
| ZC=F   | 3769 | — |
| ZS=F   | 3771 | — |

**Calendar gaps:** CBOT (agricultural) vs CME calendar differences and the CL=F negative-price exclusion removed 42 dates from the intersection (3668 common dates vs ~3710 individual per-asset).  
**Decision:** drop-date intersection (conservative). 42 dates ≈ 1.1% of the period — negligible.

---

## Results: horizon = 1

```
               n_folds  mean_CS_RIC  std_CS_RIC  CS_RIC_stab  Sharpe_net  ann_ret  max_dd  spr_capture  turnover
model
EqualWeight          8          NaN         NaN          NaN         NaN    0.000   0.000          NaN     0.000
MomentumRank         8      -0.0065      0.4063        0.482       -0.41   -0.081  -0.647       -0.003     0.167
MeanReversion        8       0.0174      0.3926        0.508       -0.01   -0.001  -0.438        0.009     0.326
ElasticNet           8          NaN         NaN          NaN         NaN    0.000   0.000          NaN     0.000
LightGBM             8       0.0098      0.3943        0.496       -0.23   -0.040  -0.597        0.006     0.454
```

**Verdict: no edge at horizon=1.** MeanReversion shows the weakest-possible positive CS-RIC (+0.0174) but Sharpe is −0.01 — zero net P&L after costs. No model reaches a positive net Sharpe.

---

## Results: horizon = 5

```
               n_folds  mean_CS_RIC  std_CS_RIC  CS_RIC_stab  Sharpe_net  ann_ret  max_dd  spr_capture  turnover
model
EqualWeight          8          NaN         NaN          NaN         NaN    0.000   0.000          NaN     0.000
MomentumRank         8      -0.0220      0.3981        0.476       -0.29   -0.059  -0.572       -0.007     0.167
MeanReversion        8       0.0197      0.3877        0.501        0.63    0.114  -0.302        0.009     0.326
ElasticNet           8          NaN         NaN          NaN         NaN    0.000   0.000          NaN     0.000
LightGBM             8       0.0109      0.4329        0.491        0.43    0.060  -0.276        0.007     0.110
```

**Verdict: weak signals persist, LightGBM improved substantially.**  
MeanReversion and LightGBM both show positive CS-RIC and positive net Sharpe at h=5.  
The signals are not strong enough to call "edge" (CS-RIC < 0.05 for both), but the direction is consistent.

---

## Phase 7 → Phase 8 comparison

### CS-RIC and stability (the key Phase 8 hypothesis)

| Model | Ph7 h=1 CS_RIC | Ph8 h=1 CS_RIC | Ph7 h=5 CS_RIC | Ph8 h=5 CS_RIC |
|-------|--------------|--------------|--------------|--------------|
| MomentumRank  | -0.0078 | **-0.0065** | -0.0350 | **-0.0220** |
| MeanReversion | -0.0032 | **+0.0174** | +0.0607 | **+0.0197** |
| LightGBM      | -0.0554 | **+0.0098** | -0.0066 | **+0.0109** |

| Model | Ph7 h=1 stab | Ph8 h=1 stab | Ph7 h=5 stab | Ph8 h=5 stab |
|-------|------------|------------|------------|------------|
| MomentumRank  | 0.463 | **0.482** | 0.450 | **0.476** |
| MeanReversion | 0.460 | **0.508** | 0.514 | **0.501** |
| LightGBM      | 0.441 | **0.496** | 0.476 | **0.491** |

### Net Sharpe comparison

| Model | Ph7 h=1 | Ph8 h=1 | Ph7 h=5 | Ph8 h=5 |
|-------|---------|---------|---------|---------|
| MomentumRank  | -0.49 | -0.41 | -0.67 | -0.29 |
| MeanReversion | -0.53 | -0.01 | +0.33 | **+0.63** |
| LightGBM      | -0.74 | -0.23 | -0.25 | **+0.43** |

---

## Did CS-RIC stability improve?

**Partially.** The Phase 8 hypothesis was that n=9 would give Spearman correlation more statistical power per date, improving CS-RIC stability.

- At **h=1**: stability improved for all models (MeanReversion: 0.460→0.508, LightGBM: 0.441→0.496). This is consistent with the hypothesis.
- At **h=5**: stability did NOT improve for MeanReversion (0.514→0.501), and showed only a small improvement for LightGBM (0.476→0.491). The mean-reversion signal at h=5 is genuinely weaker across the diverse 9-asset basket than it was across 4 closely correlated metals.

**Key finding:** the Phase 7 MeanReversion CS-RIC at h=5 (0.0607) was unusually high for the 4-metal basket and reflected tight co-movement within metals. The same signal applied to a more diverse basket (energy + industrial + agricultural) drops to 0.0197 — the metals-specific mean-reversion pattern does not generalise across commodities in a simple way.

**However**, the broader basket delivers a better Sharpe despite lower CS-RIC: MeanReversion h=5 goes from +0.33 to +0.63. This is likely because top-2/bottom-2 (vs top-1/bottom-1) reduces position concentration and provides better diversification across the P&L stream.

---

## Data quality flags

1. **CL=F non-positive close (2020-04-20):** WTI crude oil traded at -$37.63 on that date. The price is stored as-is (preserving data integrity); the pooled dataset replaces it with NaN for target computation and drops that date via intersection. Logged as WARNING.

2. **CL=F extra NaN rows at warm-up:** 81 rows vs 59 for other assets. The NaN/negative-price region around 2020-04-20 propagates through `log_returns`, causing 22 additional NaN feature rows beyond the rolling warm-up period.

3. **PA=F thin market:** 3753 rows — 17 fewer than GC=F. Occasional gaps are absorbed by the intersection with no special treatment.

4. **NG=F:** No data anomalies detected. The expected volatility is present in the data but does not cause gaps.

---

## Acceptance criteria

| Criterion | Status |
|-----------|--------|
| All 9 tickers fetched and cached (log any < 2000 rows) | ✅ All have 3700+ rows; CL=F negative-price noted |
| Cross-features work with 9 assets (rel features sum to zero) | ✅ Tested in `test_relative_features_sum_to_zero_9_assets` |
| Pooled dataset has 9 assets per date, fold-integrity test passes | ✅ `test_fold_integrity_9_assets` passes |
| Ranking backtester uses configurable n_long/n_short (default 2/2) | ✅ `pipeline_ranking.py` auto-selects 2/2 for N≥6 |
| Comparison tables at horizon=1 and horizon=5 | ✅ above |
| `phase8_summary.md` compares 9-asset vs Phase 7 4-asset results | ✅ this document |
| All tests pass (existing + new) | ✅ 224/224 pass |
| `ranked_tickers(cfg)` returns exactly the configured list | ✅ `test_ranked_tickers_returns_configured_list` |

---

## Key decisions

1. **`ranked_assets` nested under `universe` in config.** Keeps the Config dataclass unchanged (universe is already `dict`); backward compat for single-asset pipeline maintained.

2. **`price_tickers()` updated to return ranked ∪ context.** Legacy `metals` key still present for backward compat when `ranked_assets` is empty.

3. **Dynamic one-hot encoding.** One-hot map built from sorted(ranked_tickers); last ticker (ZS=F) is reference category. This generalises to any N without hardcoding.

4. **Non-positive price sanitization in pooled_dataset.py only.** `targets.py` keeps its strict validation; the pooled builder applies `where(close > 0)` before calling it, producing NaN targets that the intersection naturally drops.

5. **n_long=n_short=2 auto-selected for N≥6.** Threshold of 6 keeps 4-asset runs at 1/1 (backward compat) and 9-asset runs at 2/2.

6. **42-date drop from calendar intersection.** Chosen over forward-fill: forward-filling would hide data gaps and risk subtle leakage for thinly-traded days. The loss is only 1.1% of the total period.

---

## Honest verdict

The 9-asset expansion produced **incremental improvement** — not a clear breakthrough:

- CS-RIC stability at h=5 essentially unchanged (0.514 → 0.501).
- The specific mean-reversion signal that looked stronger with 4 metals is diluted by the diverse basket.
- Positive Sharpe emerged at h=5 for MeanReversion (+0.63) and LightGBM (+0.43), but these are marginal and driven partly by top-2/bottom-2 diversification rather than a stronger underlying ranking signal.
- ElasticNet continues to produce flat predictions (all positions zero) — the cross-sectional SNR is below the regularization threshold.

**This does not constitute a tradeable edge.** The system remains an honest research harness.

---

*Research tooling only — not investment advice.*
