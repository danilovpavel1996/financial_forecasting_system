# Phase 16 Equity Sector Sweep Results

*Format: Sharpe (net, non-overlapping) / CS-RIC (mean OOS)*

*Embargo = horizon for each run. All results are out-of-sample walk-forward.*

*Universe: XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY — 9 SPDR sector ETFs*

*No COT features (futures-specific). No carry proxy (futures-specific).*

*History start: 1999-01-01  |  End: 2024-12-31*

---

## Equity Sector Sweep Matrix

| Model | h=5 | h=21 | h=63 |
| --- | --- | --- | --- |
| MeanReversion | -0.71 / -0.0083 | -0.01 / 0.0155 | 0.35 / 0.0271 |
| LightGBM | 0.31 / 0.0068 | 0.07 / 0.0453 | 0.12 / 0.0983 |
| LambdaMART | -0.25 / -0.0197 | -0.02 / 0.0070 | -0.32 / 0.0046 |

---

## Cross-Asset Comparison: Equity Sectors vs Commodities (Phase 14)

| Model | Horizon | Equity Sharpe / CS-RIC | Commodity Sharpe / CS-RIC (Phase 14) |
| --- | --- | --- | --- |
| MeanReversion | 5 | -0.71 / -0.0083 | 0.40 / 0.0200 |
| MeanReversion | 21 | -0.01 / 0.0155 | N/A / N/A |
| MeanReversion | 63 | 0.35 / 0.0271 | N/A / N/A |
| LightGBM | 5 | 0.31 / 0.0068 | N/A / N/A |
| LightGBM | 21 | 0.07 / 0.0453 | N/A / N/A |
| LightGBM | 63 | 0.12 / 0.0983 | 0.79 / 0.0550 |
| LambdaMART | 5 | -0.25 / -0.0197 | N/A / N/A |
| LambdaMART | 21 | -0.02 / 0.0070 | N/A / N/A |
| LambdaMART | 63 | -0.32 / 0.0046 | N/A / N/A |

---

## Summary

**Equity best:** **MeanReversion** at h=63 — Sharpe 0.35

**Commodity best (Phase 14):** LightGBM at h=63 — Sharpe 0.79

**Equity beats commodity best (0.79):** NO

---

## Notes

- Sharpe uses non-overlapping subsampling every `horizon` steps to avoid autocorrelation inflation.
- CS-RIC is mean cross-sectional rank information coefficient across all OOS test dates.
- Embargo = horizon ensures forward return labels cannot leak across the train/test boundary.
- Costs charged at 5.0 bps round-trip on turnover.
- Equity sector history starts 1999-01-01; longer history than commodity backtest (2005-01-01).

*Research tooling only — not investment advice.*
