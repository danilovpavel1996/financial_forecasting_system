# Phase 17A — Cross-Asset Ensemble Results

## Individual signal performance

| Signal | Sharpe (sub63) | Horizon | Asset class |
| --- | --- | --- | --- |
| Commodity LightGBM | 0.79 | 63d | Precious metals futures |
| Equity MeanReversion | 0.35 | 63d | SPDR sector ETFs |

## P&L correlation

- Daily P&L correlation: **+0.1284**
- Assessment: Very low — strong diversification benefit expected.
- Theoretical Sharpe if ρ=0: √(0.79² + 0.35²) = **0.87**

## OOS alignment

- Common period: 2016-06-03 → 2023-10-27  (1835 days)
- Dates dropped from commodity series: 181
- Dates dropped from equity series: 181

## Weight sweep (Sharpe_sub63 = subsampled every 63 days)

| weights (comm/eq) | Sharpe_raw† | Sharpe_sub63 | ann_ret | max_dd |
| --- | --- | --- | --- | --- |
| 100/0 | 3.67 | 0.90 | 0.1207 | -0.1481 |
| 75/25 | 3.73 | 0.95 | 0.0998 | -0.1110 |
| 50/50 | 3.59 | 0.97 | 0.0788 | -0.0834 |
| 25/75 | 2.75 | 0.81 | 0.0579 | -0.0707 |
| 0/100 | 1.13 | 0.46 | 0.0369 | -0.1295 |

† Raw daily Sharpe is inflated by overlapping h=63 returns; use Sharpe_sub63 for comparison.

## Result

- Best blend (sub63): **50/50** → Sharpe=0.97
- Commodity baseline (Phase 14): 0.79
- Beats baseline: **YES**

---
*Research tooling only — not investment advice.*