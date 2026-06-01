# Phase 17B — Sector LightGBM Turnover Fix Results

## Setup

- Universe: 9 SPDR sector ETFs (same as Phase 16)
- Horizon: h=63
- Baseline: default LightGBM (n_estimators=500, num_leaves=31, min_child_samples=20)
- Baseline Sharpe 0.12, CS-RIC 0.098, turnover 0.131 (Phase 16)

## Variant descriptions

- **B1 HeavyReg**: n_estimators=50, num_leaves=15, min_child_samples=50 — fewer, shallower trees → smoother predictions.
- **B2 PosSmooth K=2**: only update positions when any asset's rank changes by ≥2 positions — filters minor noise-driven rank flips.
- **B3 PredAvg 21d**: 21-day rolling mean of raw predictions before ranking — smooths prediction noise without changing the model.

## Results

| Variant | Sharpe | CS-RIC | Turnover | ΔSharpe vs baseline |
| --- | --- | --- | --- | --- |
| Baseline (Phase 16 default LightGBM) | 0.120 | 0.0980 | 0.131 | — |
| B1_HeavyReg | -0.271 | -0.0369 | 0.107 | -0.391 |
| B2_PosSmooth_K2 | 0.117 | 0.0913 | 0.131 | -0.003 |
| B3_PredAvg21 | 0.478 | 0.0995 | 0.031 | +0.358 |

## Conclusion

- Best variant: **B3_PredAvg21** → Sharpe=0.478
- Beats equity MeanReversion baseline (0.35): **YES**

---
*Research tooling only — not investment advice.*