# Phase 18C — Three-Way Ensemble Results

## Individual signal performance

| Signal | Sharpe (sub63) | Horizon | Asset class |
| --- | --- | --- | --- |
| Commodity LightGBM | 0.79 | 63d | Precious metals futures |
| Equity LightGBM PredAvg21 | 0.41 | 63d | SPDR sector ETFs |
| Forex LambdaMART | 0.71 | 63d | Major USD forex pairs |

## Pairwise P&L correlations

| Pair | Correlation |
| --- | --- |
| Commodity vs Equity | +0.3722 |
| Commodity vs Forex  | +0.1785 |
| Equity vs Forex     | +0.2116 |

## OOS alignment

- Common period: 2016-06-03 → 2023-10-27  (1832 days)

## Weight sweep (Sharpe_sub63 = subsampled every 63 days)

| weights (comm/eq/fx) | Sharpe_raw† | Sharpe_sub63 | ann_ret | max_dd |
| --- | --- | --- | --- | --- |
| 100/0/0 | 3.68 | 1.07 | 0.1835 | -0.2201 |
| 0/100/0 | 1.99 | 0.26 | 0.0235 | -0.1778 |
| 0/0/100 | 2.49 | 0.37 | 0.0171 | -0.0950 |
| 50/50/0 | 3.68 | 0.99 | 0.1035 | -0.1607 |
| 50/0/50 | 4.03 | 1.05 | 0.1003 | -0.1231 |
| 0/50/50 | 2.73 | 0.38 | 0.0203 | -0.0833 |
| 33/33/33 | 3.95 | 0.99 | 0.0747 | -0.1161 |

† Raw daily Sharpe inflated by overlapping h=63 returns; use Sharpe_sub63 for comparison.

## Result

- Best blend (sub63): **100/0/0** → Sharpe=1.07
- Two-way ensemble baseline (Phase 18A): 0.97
- Beats two-way baseline: **YES**

---
*Research tooling only — not investment advice.*