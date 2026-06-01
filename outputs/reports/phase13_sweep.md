# Phase 13 Sweep Results

*Format: Sharpe (net, non-overlapping) / CS-RIC (mean OOS)*

*Embargo = horizon for each run. All results are out-of-sample walk-forward.*

## Sweep Matrix

| Model | h=5 no-COT | h=5 COT | h=21 no-COT | h=21 COT | h=63 no-COT | h=63 COT |
| --- | --- | --- | --- | --- | --- | --- |
| MeanReversion | 0.63 / 0.0197 | 0.29 / 0.0264 | -0.07 / 0.0215 | -0.07 / 0.0258 | 0.03 / 0.0131 | -0.12 / 0.0149 |
| LightGBM | 0.43 / 0.0109 | -0.33 / -0.0016 | -0.21 / -0.0214 | 0.11 / 0.0248 | 0.10 / 0.0247 | -0.48 / -0.0057 |
| LambdaMART | -0.59 / -0.0174 | -0.38 / -0.0073 | 0.09 / 0.0246 | 0.15 / 0.0298 | 0.02 / 0.0827 | 0.27 / 0.0768 |

## Best Configuration

**MeanReversion** at h=5 without COT — Sharpe 0.63

MeanReversion h=5 no-COT Sharpe (Phase 8 baseline): 0.63

LambdaMART beats baseline: **NO**

## Notes

- Sharpe uses non-overlapping subsampling every `horizon` steps to avoid autocorrelation inflation.
- CS-RIC is mean cross-sectional rank information coefficient across all OOS test dates.
- Embargo = horizon ensures forward return labels cannot leak across the train/test boundary.
- Costs charged at 5.0 bps round-trip on turnover.

*Research tooling only — not investment advice.*
