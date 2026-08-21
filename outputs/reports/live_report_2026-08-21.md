# Live forex report — 2026-08-21

Window: 2026-06-03 → 2026-08-14 (10 realizable signal weeks; the newest signal has no realized 5d return yet).

## 1. Signal quality — weekly cross-sectional rank IC

Spearman(predicted, realized 5d fwd log return) across all 15 pairs, per week.

| date       |   cs_ric |   n_pairs | active_hit   |   paper_ret_net |
|:-----------|---------:|----------:|:-------------|----------------:|
| 2026-06-03 |  -0.3324 |        15 | 3/6          |         -0.0059 |
| 2026-06-16 |  -0.1643 |        15 | 1/6          |         -0.0087 |
| 2026-06-23 |   0.3571 |        15 | 5/6          |          0.0065 |
| 2026-07-01 |   0.0884 |        15 | 4/6          |         -0.0014 |
| 2026-07-08 |  -0.7892 |        15 | 0/6          |         -0.0074 |
| 2026-07-15 |   0.2484 |        15 | 4/6          |          0.0009 |
| 2026-07-22 |  -0.4825 |        15 | 2/6          |         -0.0036 |
| 2026-07-30 |  -0.5536 |        15 | 3/6          |         -0.0133 |
| 2026-08-07 |   0.4786 |        15 | 5/6          |          0.0037 |
| 2026-08-14 |   0.2719 |        15 | 3/6          |          0.0015 |

- Mean CS-RIC: **-0.0878** (SE ≈ 0.1384), positive weeks: 5/10
- Backtest OOS CS-RIC for this model: **+0.071** (no stored weekly backtest IC distribution exists, so this is a point comparison, not a percentile test).

## 2. Execution fidelity — signal book vs MT5 book (end of signal day)

| date       | match   | missing              | extra   | wrong_dir   |
|:-----------|:--------|:---------------------|:--------|:------------|
| 2026-06-03 | ✗       | AUDNZD,EURAUD,GBPUSD | —       | —           |
| 2026-06-16 | ✗       | GBPAUD,USDCAD,USDJPY | —       | —           |
| 2026-06-23 | ✗       | AUDJPY,EURUSD,NZDUSD | —       | —           |
| 2026-07-01 | ✓       | —                    | —       | —           |
| 2026-07-08 | ✓       | —                    | —       | —           |
| 2026-07-15 | ✓       | —                    | —       | —           |
| 2026-07-22 | ✓       | —                    | —       | —           |
| 2026-07-30 | ✗       | —                    | —       | EURAUD      |
| 2026-08-07 | ✓       | —                    | —       | —           |
| 2026-08-14 | ✓       | —                    | —       | —           |
| 2026-08-21 | ✓       | —                    | —       | —           |

## 3. PnL — live vs paper

- Live closed-trade PnL across both demo accounts: **-63.97 USD** on 2000 (-3.20%).
  - Expired account 372709 (Jun 2 – Aug 14): -69.52 USD from the profit column; its MT5 footer read −77.15 including swaps, i.e. ≈ −7.5 USD of swap the backtest does not model.
  - Current account 438689 (from Aug 14): +5.55 USD closed, 6 positions still open.
- Of which manual-entry fumbles (opened and closed within minutes): -0.96 USD across 7 trades.
- Paper strategy (signal followed exactly, h=5 windows, 1/6 equal weight, 3 bps/side): **-2.77%** cumulative simple sum of weekly net returns.

## Honesty notes

- ~10 weekly observations: portfolio Sharpe/return over this window is statistically uninformative (SE of annualized Sharpe ≈ ±2.2). It is deliberately not reported. The IC row count (15 pairs × weeks) is the only metric here with any power.
- MT5 history for the first (expired) demo account was transcribed from a screenshot; `profit` values are as-displayed, three close prices were unreadable. History for the current account comes from the MetaApi API.
- The most recent week's IC is provisional: it is computed from the price snapshot taken during the signal run, before that day's close settles. Values shift slightly once the data finalizes (2026-08-07 read +0.67 last week, +0.48 now).
- Reconstructed signal weeks (predictions regenerated after the original file was lost, so their IC is approximate): 2026-08-21.
- Research tooling — not investment advice.