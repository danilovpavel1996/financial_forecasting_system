# Phase 14 Sweep Results

*New features: basis_momentum, carry_proxy, carry_proxy_chg_21d, rel_basis_momentum,
rel_carry_proxy, month_sin, month_cos, yield_curve_slope, yield_curve_slope_chg_21d,
yield_curve_slope_zscore.  History extended: 2005-01-01 → 2024-12-31.*

*Format: Sharpe (net, non-overlapping) / CS-RIC (mean OOS)*

*Embargo = horizon for each run. All results are out-of-sample walk-forward.*

---

## Phase 14 Sweep Matrix

| Model | h=5 no-COT | h=5 COT | h=21 no-COT | h=21 COT | h=63 no-COT | h=63 COT |
| --- | --- | --- | --- | --- | --- | --- |
| MeanReversion | 0.40 / 0.0234 | 0.16 / 0.0268 | -0.02 / 0.0261 | -0.22 / 0.0263 | -0.71 / 0.0159 | 0.10 / 0.0147 |
| LightGBM | -0.26 / 0.0005 | -0.71 / -0.0286 | 0.59 / 0.0403 | 0.36 / 0.0176 | 0.79 / 0.0550 | 0.36 / 0.0316 |
| LambdaMART | -0.54 / -0.0081 | -0.28 / 0.0005 | 0.20 / 0.0184 | 0.59 / 0.0237 | 0.61 / 0.0497 | 0.45 / 0.0737 |

## Phase 13 Sweep Matrix (baseline)

| Model | h=5 no-COT | h=5 COT | h=21 no-COT | h=21 COT | h=63 no-COT | h=63 COT |
| --- | --- | --- | --- | --- | --- | --- |
| MeanReversion | 0.63 / 0.0197 | 0.29 / 0.0264 | -0.07 / 0.0215 | -0.07 / 0.0258 | 0.03 / 0.0131 | -0.12 / 0.0149 |
| LightGBM | 0.43 / 0.0109 | -0.33 / -0.0016 | -0.21 / -0.0214 | 0.11 / 0.0248 | 0.10 / 0.0247 | -0.48 / -0.0057 |
| LambdaMART | -0.59 / -0.0174 | -0.38 / -0.0073 | 0.09 / 0.0246 | 0.15 / 0.0298 | 0.02 / 0.0827 | 0.27 / 0.0768 |

## Delta: Phase 14 Sharpe minus Phase 13 Sharpe

| Model | h=5 no-COT | h=5 COT | h=21 no-COT | h=21 COT | h=63 no-COT | h=63 COT |
| --- | --- | --- | --- | --- | --- | --- |
| MeanReversion | -0.23 | -0.13 | +0.05 | -0.15 | -0.74 | +0.22 |
| LightGBM | -0.69 | -0.38 | +0.80 | +0.25 | +0.69 | +0.84 |
| LambdaMART | +0.05 | +0.10 | +0.11 | +0.44 | +0.59 | +0.18 |

---

## Best Configuration (Phase 14)

**LightGBM** at h=63 without COT — Sharpe 0.79

MeanReversion h=5 no-COT (Phase 13 baseline): 0.63
MeanReversion h=5 no-COT (Phase 14): 0.40

Phase 14 best beats Phase 13 baseline (Sharpe 0.63): **YES**


## Notes

- Sharpe uses non-overlapping subsampling every `horizon` steps to reduce autocorrelation.
- CS-RIC is mean cross-sectional rank information coefficient across all OOS test dates.
- Embargo = horizon; forward return labels cannot leak across the train/test boundary.
- Costs charged at 5.0 bps round-trip on turnover.
- COT data (Disaggregated CFTC) available from ~2006; warm-up starts 2007 for percentile features.
- carry_proxy available for GC=F (vs GLD) and SI=F (vs SLV) only.
- yield_curve_slope requires both DGS10 and DGS2; zscore requires 252-day history warm-up.

*Research tooling only — not investment advice.*
