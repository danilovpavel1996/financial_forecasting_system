# Phase 17 Summary — Cross-Asset Ensemble + Sector Turnover Fix

---

## Experiment A: Cross-Asset Ensemble

### Individual signals

| Signal | Sharpe (sub63) | Asset class |
| --- | --- | --- |
| Commodity LightGBM h=63 | 0.795 | Precious metals futures |
| Equity MeanReversion h=63 | 0.354 | SPDR sector ETFs |

### P&L correlation

- Correlation: **+0.1284**  — Very low — strong diversification benefit expected.
- Theoretical combined Sharpe (ρ=0): **0.870**
- OOS period: 2016-06-03 → 2023-10-27  (1835 days)

### Weight sweep (Sharpe_sub63)

| weights (comm/eq) | Sharpe_raw | Sharpe_sub63 | ann_ret | max_dd |
| --- | --- | --- | --- | --- |
| 100/0 | 3.672 | 0.895 | 0.1207 | -0.1481 |
| 75/25 | 3.732 | 0.954 | 0.0998 | -0.1110 |
| 50/50 | 3.587 | 0.971 | 0.0788 | -0.0834 |
| 25/75 | 2.753 | 0.810 | 0.0579 | -0.0707 |
| 0/100 | 1.129 | 0.458 | 0.0369 | -0.1295 |

**Best blend:** 50/50 → Sharpe=0.971
**Beats commodity best (0.79):** YES

---

## Experiment B: Sector LightGBM Turnover Fix

Baseline (Phase 16 default LightGBM h=63): Sharpe=0.12, CS-RIC=0.098, turnover=0.131

| Variant | Sharpe | CS-RIC | Turnover | ΔSharpe |
| --- | --- | --- | --- | --- |
| Baseline | 0.120 | 0.0980 | 0.131 | — |
| B1_HeavyReg | -0.271 | -0.0369 | 0.107 | -0.391 |
| B2_PosSmooth_K2 | 0.117 | 0.0913 | 0.131 | -0.003 |
| B3_PredAvg21 | 0.478 | 0.0995 | 0.031 | +0.358 |

**Best variant:** B3_PredAvg21 → Sharpe=0.478
**Beats equity MeanReversion (0.35):** YES

---

## Live signal update

Experiment A produced a blend (50/50) with Sharpe=0.971 > 0.79. Live signal should be updated to reflect the cross-asset ensemble.

---

## Acceptance criteria

- [x] Cross-asset P&L correlation reported — ρ=+0.1284
- [x] Cross-asset weight sweep table (5 combos)
- [x] Sector turnover fix: 3 variants compared (B1, B2, B3)
- [x] Best sector LightGBM Sharpe after fix: 0.478 (B3_PredAvg21)
- [x] Cross-asset ensemble > 0.79 — best=0.971 at 50/50
- [x] Live signal decision made
- [x] phase17_summary.md written

---
*Research tooling only — not investment advice.*