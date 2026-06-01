# Phase 18A — Cross-Asset Ensemble Results (Upgraded Equity Leg)

## Individual signal performance

| Signal | Sharpe (sub63) | Horizon | Asset class |
| --- | --- | --- | --- |
| Commodity LightGBM | 0.79 | 63d | Precious metals futures |
| Equity LightGBM PredAvg21 | 0.41 | 63d | SPDR sector ETFs |

## P&L correlation

- Daily P&L correlation: **+0.3716**
- Assessment: Low — meaningful diversification benefit.
- Theoretical Sharpe if ρ=0: √(0.79² + 0.41²) = **0.89**

## OOS alignment

- Common period: 2016-06-03 → 2023-10-27  (1835 days)
- Dates dropped from commodity series: 181
- Dates dropped from equity series: 181

## Weight sweep (Sharpe_sub63 = subsampled every 63 days)

| weights (comm/eq) | Sharpe_raw† | Sharpe_sub63 | ann_ret | max_dd |
| --- | --- | --- | --- | --- |
| 100/0 | 3.67 | 0.90 | 0.1207 | -0.1481 |
| 75/25 | 3.73 | 0.91 | 0.0997 | -0.1322 |
| 50/50 | 3.68 | 0.88 | 0.0787 | -0.1320 |
| 25/75 | 3.23 | 0.73 | 0.0577 | -0.1354 |
| 0/100 | 2.00 | 0.45 | 0.0367 | -0.1707 |

† Raw daily Sharpe is inflated by overlapping h=63 returns; use Sharpe_sub63 for comparison.

## Result

- Best blend (sub63): **75/25** → Sharpe=0.91
- Commodity baseline (Phase 14): 0.79
- Beats baseline: **YES**

---
*Research tooling only — not investment advice.*