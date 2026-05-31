# Phase 7 Summary — Cross-Sectional Metals Ranking

**Date:** 2026-05-31  
**Assets:** GC=F, SI=F, PL=F, PA=F (4-metal basket)  
**Period:** 2010-01-01 – 2024-12-31  **Folds:** 8 walk-forward  **Cost:** 5 bps round-trip

---

## What was built

| Component | File |
|-----------|------|
| Cross-asset relative features (rel_mom, rel_vol, ratios) | `src/features/cross_features.py` |
| Pooled (date × asset) dataset builder | `src/features/pooled_dataset.py` |
| CS-RIC, spread capture, L/S return metrics | `src/eval/cross_metrics.py` |
| RankingBacktester + EqualWeight/MomentumRank/MeanReversion baselines | `src/eval/rank_backtester.py` |
| End-to-end pipeline | `src/pipeline_ranking.py` |
| CLI | `scripts/run_ranking.py` |
| Tests (39 new, 217 total pass) | `tests/test_cross_features.py`, `test_pooled_dataset.py`, `test_cross_metrics.py`, `test_rank_backtester.py` |

---

## Acceptance criteria

| Criterion | Status |
|-----------|--------|
| Cross-asset features leakage-tested (shift-and-compare) | ✅ PASS |
| Pooled dataset: all 4 assets at same date in same fold | ✅ PASS (test_fold_integrity) |
| CS-RIC metric tested on known-rank synthetic data | ✅ PASS (CS-RIC=1, -1, NaN verified) |
| Ranking backtester: baselines + ElasticNet + LightGBM run | ✅ PASS |
| Results at both horizon=1 and horizon=5 reported honestly | ✅ below |

---

## Results: horizon = 1 (daily rebalancing)

```
               n_folds  mean_CS_RIC  std_CS_RIC  CS_RIC_stab  Sharpe_net  ann_ret  max_dd  spr_capture  turnover
model
EqualWeight          8          NaN         NaN          NaN         NaN    0.000   0.000          NaN     0.000
MomentumRank         8      -0.0078      0.6153        0.463       -0.49   -0.082  -0.599       -0.000     0.184
MeanReversion        8      -0.0032      0.6148        0.460       -0.53   -0.090  -0.672       -0.002     0.362
ElasticNet           8          NaN         NaN          NaN         NaN    0.000   0.000          NaN     0.000
LightGBM             8      -0.0554      0.5959        0.441       -0.74   -0.077  -0.514       -0.024     0.201
```

**Verdict: no edge at horizon=1.** Every model with a signal (MomentumRank, MeanReversion, LightGBM) shows negative CS-RIC and negative net Sharpe. ElasticNet collapses to flat predictions — regularization drives coefficients toward zero, meaning the model cannot find cross-sectional differentiation and degenerates to EqualWeight (all positions = 0).

---

## Results: horizon = 5 (weekly rebalancing)

```
               n_folds  mean_CS_RIC  std_CS_RIC  CS_RIC_stab  Sharpe_net  ann_ret  max_dd  spr_capture  turnover
model
EqualWeight          8          NaN         NaN          NaN         NaN    0.000   0.000          NaN     0.000
MomentumRank         8      -0.0350      0.6071        0.450       -0.67   -0.106  -0.663       -0.014     0.184
MeanReversion        8       0.0607      0.6074        0.514        0.33    0.055  -0.352        0.029     0.362
ElasticNet           8          NaN         NaN          NaN         NaN    0.000   0.000          NaN     0.000
LightGBM             8      -0.0066      0.6123        0.476       -0.25   -0.035  -0.454       -0.003     0.237
```

**Verdict: one weak signal, not full success.**

- **MeanReversion** is the only model with positive CS-RIC (+0.0607) and positive net Sharpe (+0.33). The 5-day short-term reversal effect — metals that underperformed the basket over the past 5 days tending to outperform next week — is weakly present in this data.
- This **partially** meets the Phase 7 success criteria:  
  - CS-RIC > 0.05: ✅ (0.0607)  
  - CS-RIC stability > 0.55: ✗ (0.514 — just below the threshold)  
  - Net Sharpe positive after costs: ✅ (+0.33)
- **MomentumRank** is clearly destructive at both horizons (CS-RIC negative). Cross-sectional momentum in metals does not appear over a 21d trailing window.
- **LightGBM** shows near-zero CS-RIC at h=5 (−0.0066, Sharpe −0.25). The ML model finds no robust ranking signal that survives costs in this feature set.

---

## Key decisions made during implementation

1. **Walk-forward split on unique dates, not on rows.** The splitter sees N_dates positions; all 4 asset rows at date t always go to the same fold. This is enforced and tested.

2. **EqualWeight implemented as predict-0 → all-zero positions.** When `std(pred) < 1e-12`, the backtester assigns zero positions. This is the cleanest null: zero P&L, no spurious ranking from tie-breaking.

3. **ElasticNet degenerates to EqualWeight.** With the current regularization (alpha=0.01), the pooled 4× dataset and relatively low cross-sectional SNR means ElasticNet predicts approximately the same value for all 4 assets at each date. This is not a bug — it is an honest reflection of the regularizer's behaviour. A lower alpha or asset-specific model might differ.

4. **Non-overlapping Sharpe applied to ranking P&L.** The `_ls_sharpe_stats_with_horizon` function subsamples the P&L series at every `horizon`-th observation, consistent with the metrics.py fix.

5. **Context ETF features are shared across all 4 assets.** Since all metals futures close at ~2:30pm ET, the same late-close lag applied to ETF context features (GLD, SLV, GDX, SPY, TLT, UUP) in Phase 3 applies uniformly across all 4 metals. Features were built with GC=F as the reference calendar and reused for the pooled dataset.

---

## What would need to change to find edge

- **Different short-term reversal window.** The 5d return as the reversal signal is a single choice. The effect may be stronger at 3d or 10d.
- **Lower alpha on ElasticNet** (e.g., 0.001) to allow more cross-sectional differentiation.
- **Asset-specific models** rather than pooled (removes the need for one-hot encoding).
- **Alternative features:** CFTC positioning data, term structure of futures, seasonal patterns, spread mean-reversion levels rather than momentum.
- **Larger basket:** adding other commodity futures beyond 4 metals would give CS-RIC more statistical power per date.

---

*Research tooling only — not investment advice.*
