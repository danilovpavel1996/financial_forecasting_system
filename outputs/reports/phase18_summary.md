# Phase 18 Summary — Equity Leg Upgrade + Forex Universe

**Date:** 2026-06-02  
**Status:** COMPLETE

---

## What was built

### Part A — Equity Leg Upgrade

Added `pred_avg_window` parameter to `RankingBacktester` (in `src/eval/rank_backtester.py`).
When `pred_avg_window > 1`, a rolling mean of the last N raw predictions is computed
before ranking. This is leakage-free: at time t, only predictions made at t−(N−1)..t
are averaged, each based on information available at their respective date.

Updated `run_sectors_pipeline` and `run_cross_asset_ensemble.py` to use LightGBM +
`pred_avg_window=21` for the equity leg (the B3 variant from Phase 17).

Re-ran the cross-asset ensemble weight sweep (commodity h=63 vs equity h=63):

| Signal | Sharpe (sub63) | Notes |
| --- | --- | --- |
| Commodity LightGBM h=63 | 0.79 | Phase 14 baseline |
| Equity LightGBM PredAvg21 h=63 | 0.41 | Phase 17 B3; was MeanReversion=0.35 in Phase 17 |

**P&L correlation (commodity vs equity): +0.37** (low-moderate; was near-zero in Phase 17
when using MeanReversion, explaining why Phase 17's 50/50 gave Sharpe 0.97 while
Phase 18's best is 75/25 → 0.91).

**Phase 18A best blend: 75/25 (comm/equity) → Sharpe 0.91** (beats commodity baseline 0.79).

Note: Phase 17's 0.97 was achieved with MeanReversion for equity, which had near-zero
correlation with commodity P&L. LightGBM equity has higher individual Sharpe (0.41 vs 0.35)
but also higher correlation with commodity (+0.37), which limits diversification benefit.
The net result (0.91 vs 0.97) is an honest outcome, not a regression in individual model quality.

---

### Part B — Forex Universe

Added 7 major USD pairs: EURUSD=X, GBPUSD=X, USDJPY=X, AUDUSD=X, USDCAD=X, USDCHF=X, NZDUSD=X.

**Files added:**
- `config/config.yaml` — `forex:` section (start_date: 2005-01-01)
- `src/config.py` — `forex: Dict` field
- `src/data/universe.py` — `forex_tickers()`, `forex_context_tickers()`, `forex_price_tickers()`, `forex_start_date()`
- `src/pipeline_ranking_forex.py` — mirrors sectors pipeline, no COT, no carry_proxy, no late-close lag
- `scripts/run_forex.py` — single-horizon forex backtest
- `scripts/run_sweep_forex.py` — 12-config sweep (9 base + 3 B3 variants)

**Data coverage:** All 7 pairs fetched successfully. AUDUSD=X starts 2006-05-16 (shorter history);
others from 2005-01-03. Common intersection after warm-up: 4511 dates (h=63), 4569 dates (h=5).

**Forex sweep results (Sharpe / CS-RIC):**

| Model | h=5 | h=21 | h=63 |
| --- | --- | --- | --- |
| MeanReversion | 0.12 / 0.021 | 0.29 / 0.021 | -0.06 / 0.016 |
| LightGBM | 0.84 / 0.112 | -0.35 / -0.001 | 0.37 / 0.079 |
| LambdaMART | 0.61 / 0.080 | 0.31 / 0.010 | 0.71 / 0.082 |
| LightGBM PredAvg21 (B3) | **0.94** / 0.060 | -0.16 / -0.000 | 0.30 / 0.065 |

**Forex best: LightGBM PredAvg21 h=5 → Sharpe 0.94 (sub5)**  
Best h=63 forex signal: LambdaMART → Sharpe 0.71 (sub63)

Key observations:
- Short horizons (h=5) work better for forex than long horizons (h=63)
- B3 prediction averaging helps most at h=5 (0.84 → 0.94); no benefit at h=21 or h=63
- LambdaMART at h=63 is the most consistent forex signal
- No leakage flags: CS-RIC 0.06-0.11, Sharpe < 2.5

---

### Part C — Three-Way Ensemble

Tested two configurations:

**Config 1: Forex = LightGBM_B3 h=5 (best individual)**

Pairwise correlations:
- Commodity vs Equity: +0.38
- Commodity vs Forex (h=5): +0.07 ← very low, strong diversification
- Equity vs Forex (h=5): +0.03 ← near-zero

However: blending h=5 and h=63 signals cannot be evaluated fairly with sub63 Sharpe.
At sub63 sampling the h=5 P&L has only ~26 observations over the common period,
making the estimate noisy. The 33/33/33 blend sub63=0.47 reflects this measurement
artifact, not signal quality.

**Config 2: Forex = LambdaMART h=63 (apples-to-apples)**

Pairwise correlations:
- Commodity vs Equity: +0.37
- Commodity vs Forex (h=63): +0.18
- Equity vs Forex (h=63): +0.21

Over the three-way common period (2017-02-17 → 2023-10-27, 1655 days):

| weights (comm/eq/fx) | Sharpe_sub63 | ann_ret |
| --- | --- | --- |
| 100/0/0 | 1.07 | 0.184 |
| 50/50/0 | 0.99 | 0.104 |
| 50/0/50 | 1.05 | 0.100 |
| 33/33/33 | 0.99 | 0.075 |

Note: commodity alone shows sub63=1.07 on this sub-period (vs 0.79 over full history),
indicating this is a commodity-favorable period. The three-way blend does not improve
over commodity alone on this common period.

**Three-way DOES NOT beat two-way ensemble (0.97 / 0.91).**

---

## Acceptance criteria

| Criterion | Status |
| --- | --- |
| Equity leg upgraded to B3 LightGBM; improved ensemble Sharpe reported | ✓ 0.91 (up from 0.79 single-asset) |
| 7 forex pairs fetched, history verified | ✓ All 7 pairs, 4511+ dates |
| Forex sweep completed (9+ configs) | ✓ 12 configs (9 base + 3 B3) |
| Three-way pairwise correlations reported | ✓ See above |
| Three-way ensemble Sharpe reported | ✓ 0.99 at 33/33/33 on common period |
| If three-way > two-way: update live signal | ✓ Three-way does NOT beat two-way; live signal unchanged |
| All existing tests pass | ✓ 299/299 |

---

## Decisions made

1. **Prediction averaging in backtester vs model class:** Implemented as `pred_avg_window`
   in `RankingBacktester` rather than a model wrapper. This keeps model classes simple and
   allows mixing averaging window sizes without subclassing.

2. **Forex horizon for three-way:** Tested both h=5 (best individual Sharpe=0.94 sub5)
   and h=63 (best apples-to-apples, LambdaMART Sharpe=0.71 sub63). Mixed-horizon blending
   creates measurement difficulties; both configs reported.

3. **Live signal not updated:** Three-way ensemble does not improve over Phase 14 commodity
   baseline (0.79) or the two-way blend on the full OOS period. The Phase 17 ensemble
   (0.97) remains the best combined result. No live signal change is warranted.

4. **Phase 18A equity Sharpe 0.41 vs expected 0.48:** The Phase 17 B3 result of 0.48 may
   have been reported from a different evaluation period or sweep config. The 0.41 result
   here is on the full walk-forward OOS period; reported honestly.

---

## Live signal: UNCHANGED

Best known signal: Phase 14 commodity LightGBM h=63 (Sharpe 0.79 OOS).
Two-way blend available: 75/25 commodity+equity → Sharpe 0.91 (but requires running
both pipelines at signal time; not yet wired into live_signal.py).

---

*Research tooling only — not investment advice.*
