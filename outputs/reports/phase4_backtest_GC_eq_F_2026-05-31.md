# Phase 4 Backtest Report — GC=F

**Date:** 2026-05-31
**Ticker:** GC=F
**Period:** 2010-01-01 → 2024-12-31
**Horizon:** 1 trading day(s)
**Cost:** 5.0 bps round-trip
**Splitter:** 8 folds, train=3yr, test=1yr, embargo=5d

---

## Baseline Comparison

| model        |   n_folds |   mean_IC |   rank_IC |   IC_stability |   pooled_IC |   hit_rate |   Sharpe_net |   ann_ret |   max_dd |   turnover |
|:-------------|----------:|----------:|----------:|---------------:|------------:|-----------:|-------------:|----------:|---------:|-----------:|
| RandomWalk   |         8 |  nan      |  nan      |         nan    |    nan      |    nan     |       nan    |     0     |    0     |      0     |
| Drift        |         8 |  nan      |  nan      |         nan    |     -0.0032 |      0.531 |         0.56 |     0.081 |   -0.23  |      0     |
| Momentum     |         8 |   -0.0073 |    0.0011 |           0.38 |     -0.0068 |      0.507 |        -1.02 |    -0.147 |   -0.758 |      1.01  |
| ElasticNet   |         8 |  nan      |  nan      |         nan    |     -0.0032 |      0.531 |         0.56 |     0.081 |   -0.23  |      0     |
| LightGBM     |         8 |   -0.0289 |   -0.0229 |           0.25 |     -0.0036 |      0.511 |        -0.31 |    -0.044 |   -0.537 |      0.565 |
| LightGBM_q90 |         8 |    0.0106 |   -0.0039 |           0.5  |      0.0136 |      0.531 |         0.56 |     0.081 |   -0.23  |      0     |

---

## Per-fold IC

### RandomWalk

|   fold |   train_size |   test_size |   IC |   rank_IC |   DA |
|-------:|-------------:|------------:|-----:|----------:|-----:|
|      0 |         1512 |         252 |  nan |       nan |  nan |
|      1 |         1764 |         252 |  nan |       nan |  nan |
|      2 |         2016 |         252 |  nan |       nan |  nan |
|      3 |         2268 |         252 |  nan |       nan |  nan |
|      4 |         2520 |         252 |  nan |       nan |  nan |
|      5 |         2772 |         252 |  nan |       nan |  nan |
|      6 |         3024 |         252 |  nan |       nan |  nan |
|      7 |         3276 |         252 |  nan |       nan |  nan |

### Drift

|   fold |   train_size |   test_size |   IC |   rank_IC |    DA |
|-------:|-------------:|------------:|-----:|----------:|------:|
|      0 |         1512 |         252 |  nan |       nan | 0.492 |
|      1 |         1764 |         252 |  nan |       nan | 0.56  |
|      2 |         2016 |         252 |  nan |       nan | 0.468 |
|      3 |         2268 |         252 |  nan |       nan | 0.563 |
|      4 |         2520 |         252 |  nan |       nan | 0.556 |
|      5 |         2772 |         252 |  nan |       nan | 0.571 |
|      6 |         3024 |         252 |  nan |       nan | 0.504 |
|      7 |         3276 |         252 |  nan |       nan | 0.532 |

### Momentum

|   fold |   train_size |   test_size |      IC |   rank_IC |    DA |
|-------:|-------------:|------------:|--------:|----------:|------:|
|      0 |         1512 |         252 |  0.108  |    0.0853 | 0.544 |
|      1 |         1764 |         252 |  0.0251 |    0.0426 | 0.532 |
|      2 |         2016 |         252 | -0.014  |   -0.02   | 0.534 |
|      3 |         2268 |         252 | -0.0228 |    0.0294 | 0.492 |
|      4 |         2520 |         252 | -0.0003 |   -0.0173 | 0.49  |
|      5 |         2772 |         252 | -0.1116 |   -0.0735 | 0.476 |
|      6 |         3024 |         252 |  0.0489 |    0.0405 | 0.518 |
|      7 |         3276 |         252 | -0.0912 |   -0.0778 | 0.474 |

### ElasticNet

|   fold |   train_size |   test_size |   IC |   rank_IC |    DA |
|-------:|-------------:|------------:|-----:|----------:|------:|
|      0 |         1512 |         252 |  nan |       nan | 0.492 |
|      1 |         1764 |         252 |  nan |       nan | 0.56  |
|      2 |         2016 |         252 |  nan |       nan | 0.468 |
|      3 |         2268 |         252 |  nan |       nan | 0.563 |
|      4 |         2520 |         252 |  nan |       nan | 0.556 |
|      5 |         2772 |         252 |  nan |       nan | 0.571 |
|      6 |         3024 |         252 |  nan |       nan | 0.504 |
|      7 |         3276 |         252 |  nan |       nan | 0.532 |

### LightGBM

|   fold |   train_size |   test_size |      IC |   rank_IC |    DA |
|-------:|-------------:|------------:|--------:|----------:|------:|
|      0 |         1512 |         252 |  0.0127 |   -0.0137 | 0.472 |
|      1 |         1764 |         252 | -0.0417 |    0.0245 | 0.548 |
|      2 |         2016 |         252 | -0.0197 |   -0.0157 | 0.52  |
|      3 |         2268 |         252 |  0.024  |   -0.0416 | 0.544 |
|      4 |         2520 |         252 | -0.0468 |   -0.0544 | 0.488 |
|      5 |         2772 |         252 | -0.0437 |   -0.0409 | 0.508 |
|      6 |         3024 |         252 | -0.0337 |   -0.0121 | 0.48  |
|      7 |         3276 |         252 | -0.0819 |   -0.0291 | 0.532 |

### LightGBM_q90

|   fold |   train_size |   test_size |      IC |   rank_IC |    DA |
|-------:|-------------:|------------:|--------:|----------:|------:|
|      0 |         1512 |         252 | -0.0303 |   -0.0395 | 0.492 |
|      1 |         1764 |         252 | -0.0068 |   -0.0071 | 0.56  |
|      2 |         2016 |         252 | -0.0192 |   -0.0167 | 0.468 |
|      3 |         2268 |         252 |  0.0596 |    0.0369 | 0.563 |
|      4 |         2520 |         252 |  0.0939 |    0.05   | 0.556 |
|      5 |         2772 |         252 |  0.0652 |    0.0241 | 0.571 |
|      6 |         3024 |         252 |  0.0218 |   -0.0302 | 0.504 |
|      7 |         3276 |         252 | -0.0995 |   -0.049  | 0.532 |

---

## Interpretation

Near-zero IC and Sharpe for all baselines is the expected and correct result. These models carry no exploitable information about future returns beyond the unconditional drift. Pooled IC in the range ±0.03 is consistent with an efficient market at the 1-day horizon.

**Red-flag threshold:** pooled IC > 0.05 or annualised Sharpe > 2.5 for a baseline is a sign of data leakage, not genuine edge. Investigate before proceeding.

**Equity curve:** `outputs/figures/equity_curve_GC_eq_F_2026-05-31.png`

---

*Research tooling only — not investment advice.*