# Phase 20 Expanded Forex Sweep Results (15 pairs, cost_bps=3)

*Format: Sharpe (net, non-overlapping) / CS-RIC (mean OOS)*

*Embargo = horizon for each run. All results are out-of-sample walk-forward.*

*Universe (15 pairs): EURUSD=X, GBPUSD=X, USDJPY=X, AUDUSD=X, USDCAD=X, USDCHF=X, NZDUSD=X, EURJPY=X, GBPJPY=X, EURGBP=X, AUDJPY=X, EURAUD=X, GBPAUD=X, AUDNZD=X, CADJPY=X*

*No COT. No carry proxy. No late-close lag (all pairs 5pm ET).*

*History start: 2005-01-01  |  End: 2024-12-31*

*Cost: 3 bps round-trip (Phase 18 reference used 5 bps)*

---

## Forex Expanded Sweep Matrix

| Model | h=5 | h=21 | h=63 |
| --- | --- | --- | --- |
| MeanReversion | -0.20 / 0.0140 | 0.39 / 0.0212 | -0.44 / 0.0261 |
| LightGBM | 1.32 / 0.0714 | 0.05 / 0.0062 | -0.21 / 0.0630 |
| LambdaMART | 0.99 / 0.0392 | -0.17 / -0.0040 | 0.25 / 0.0506 |
| LightGBM PredAvg21 | 0.79 / 0.0342 | -0.21 / 0.0002 | 0.35 / 0.0117 |

---

## Turnover Matrix

| Model | h=5 | h=21 | h=63 |
| --- | --- | --- | --- |
| MeanReversion | 0.308 | 0.308 | 0.309 |
| LightGBM | 0.283 | 0.095 | 0.091 |
| LambdaMART | 0.227 | 0.146 | 0.147 |
| LightGBM PredAvg21 | 0.047 | 0.031 | 0.022 |

---

## Summary

**Forex expanded best:** **LightGBM** at h=5 — Sharpe 1.32

**Forex 7-pair reference (Phase 18):** LightGBM B3 h=5 — Sharpe 0.94

---

## Notes

- Sharpe uses non-overlapping subsampling every `horizon` steps to avoid autocorrelation inflation.
- CS-RIC is mean cross-sectional rank information coefficient across all OOS test dates.
- Embargo = horizon ensures forward return labels cannot leak across train/test boundary.
- Costs charged at 3 bps round-trip on turnover (realistic for spread-only broker).
- LightGBM PredAvg21 applies rolling 21-day mean to predictions before ranking (B3 variant).
- With 15 pairs: top-3 long, bottom-3 short (vs top-2/bottom-2 for 7 pairs).

*Research tooling only — not investment advice.*
