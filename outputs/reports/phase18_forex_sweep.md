# Phase 18 Forex Sweep Results

*Format: Sharpe (net, non-overlapping) / CS-RIC (mean OOS)*

*Embargo = horizon for each run. All results are out-of-sample walk-forward.*

*Universe: EURUSD=X, GBPUSD=X, USDJPY=X, AUDUSD=X, USDCAD=X, USDCHF=X, NZDUSD=X*

*No COT (not configured for forex). No carry proxy. No late-close lag (all pairs 5pm ET).*

*History start: 2005-01-01  |  End: 2024-12-31*

---

## Forex Sweep Matrix

| Model | h=5 | h=21 | h=63 |
| --- | --- | --- | --- |
| MeanReversion | 0.12 / 0.0210 | 0.29 / 0.0211 | -0.06 / 0.0161 |
| LightGBM | 0.84 / 0.1124 | -0.35 / -0.0012 | 0.37 / 0.0786 |
| LambdaMART | 0.61 / 0.0796 | 0.31 / 0.0095 | 0.71 / 0.0818 |
| LightGBM PredAvg21 | 0.94 / 0.0604 | -0.16 / -0.0003 | 0.30 / 0.0649 |

---

## Summary

**Forex best:** **LightGBM PredAvg21** at h=5 — Sharpe 0.94

**Commodity reference (Phase 14):** LightGBM h=63 — Sharpe 0.79

**Equity reference (Phase 17/18):** LightGBM PredAvg21 h=63 — Sharpe 0.48

---

## Notes

- Sharpe uses non-overlapping subsampling every `horizon` steps to avoid autocorrelation inflation.
- CS-RIC is mean cross-sectional rank information coefficient across all OOS test dates.
- Embargo = horizon ensures forward return labels cannot leak across the train/test boundary.
- Costs charged at 5.0 bps round-trip on turnover.
- LightGBM PredAvg21 applies rolling 21-day mean to predictions before ranking (B3 variant).
- Quote convention: EURUSD/GBPUSD/AUDUSD/NZDUSD are foreign/USD; USDJPY/USDCAD/USDCHF are USD/foreign.
  No inversion — model learns each pair's behavior via one-hot encoding.

*Research tooling only — not investment advice.*
