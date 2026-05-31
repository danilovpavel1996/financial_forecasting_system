# Phase 6 Summary — Reporting & Monitoring

**Date:** 2026-05-31
**Status:** COMPLETE — 177/177 tests pass
**Config hash for this run:** `cab3d8d5`

---

## What was built

### `src/reporting.py`

A self-contained reporting module with five public functions:

| Function | Purpose |
|:---------|:--------|
| `config_hash(cfg)` | 8-char MD5 hex over universe / dates / horizon / cost / splitter / seed |
| `daily_net_pnl(pred, realized, cost_bps)` | Daily net-of-cost P&L (sign sizing) |
| `equity_curve(pred, realized, cost_bps)` | Cumulative return array starting at 1.0 |
| `save_equity_figure(...)` | Equity-curve PNG (all models, same colours as rolling IC) |
| `save_rolling_ic_figure(...)` | 63-day rolling Pearson IC time-series; fold boundaries as grey dotted lines; constant-prediction models skipped |
| `build_verdict(results, cfg, ticker)` | One-paragraph plain-English verdict with three branches: leakage flag / edge found / no edge |
| `write_report(...)` | Orchestrates everything; returns markdown path |

#### Config hash

The hash is computed over a canonically JSON-serialised dict of all decision-relevant
config fields (universe, dates, horizon, cost, splitter params, seed). Running the same
experiment on a different date produces a **different date stamp but identical hash** —
making the pair `(date, hash)` a reproducible experiment identifier. Any change to the
config (new feature group, different cost, changed embargo) produces a new hash.

#### Rolling IC figure

Rolling Pearson IC is computed over the concatenated OOS predictions from all 8 folds
(2016 trading days) with a default 63-day window (~1 quarter). Models with constant
predictions (RandomWalk, Drift, ElasticNet under heavy regularization) are skipped because
their rolling IC is identically NaN. Fold boundaries are marked with grey dotted vertical
lines so temporal stability is visible at a glance.

#### Verdict logic

```
if any model has |pooled_IC| > 0.05 OR Sharpe > 2.5:
    → LEAKAGE FLAG (model names listed)
elif any non-baseline model beats Drift Sharpe AND IC_stability > 0.5:
    → EDGE FOUND (model names, Sharpe values)
else:
    → NO EDGE — expected and honest
```

### `scripts/run_backtest.py` (rewritten)

Single entry point for the full pipeline:

```
python scripts/run_backtest.py [--refresh] [--ticker GC=F] [--rolling-window 63]
```

Prints config hash, comparison table, and verdict to stdout. Calls
`src.reporting.write_report` to write all three output files.

### `tests/test_reporting.py`

33 new tests covering:
- `config_hash`: 8 hex chars, stable, changes on horizon / dates / cost changes
- `daily_net_pnl` / `equity_curve`: shape, zero-position flat, cost reduces PnL
- `build_verdict`: no-edge / edge / leakage branches, mentions ticker and period, non-empty
- `save_equity_figure` / `save_rolling_ic_figure`: files created, skip on constant predictions
- `write_report` (integration): all three files created, filename contains hash+date+ticker, report
  body contains hash, verdict section, comparison table, figure links, per-fold section;
  two runs with same config produce same hash in filenames

---

## Run output — GC=F, 2026-05-31, config `cab3d8d5`

**Files written:**

```
outputs/reports/backtest_GC_eq_F_2026-05-31_cab3d8d5.md
outputs/figures/equity_GC_eq_F_2026-05-31_cab3d8d5.png
outputs/figures/rolling_ic_GC_eq_F_2026-05-31_cab3d8d5.png
```

**Comparison table:**

| model | n_folds | mean_IC | pooled_IC | IC_stability | Sharpe_net | ann_ret | max_dd | turnover |
|:------|--------:|--------:|----------:|-------------:|-----------:|--------:|-------:|---------:|
| RandomWalk | 8 | NaN | NaN | NaN | NaN | 0.000 | 0.000 | 0.000 |
| Drift | 8 | NaN | -0.003 | NaN | 0.56 | 0.081 | -0.230 | 0.000 |
| Momentum | 8 | -0.007 | -0.007 | 0.38 | -1.02 | -0.147 | -0.758 | 1.010 |
| ElasticNet | 8 | NaN | -0.003 | NaN | 0.56 | 0.081 | -0.230 | 0.000 |
| LightGBM | 8 | -0.029 | -0.004 | 0.25 | -0.31 | -0.044 | -0.537 | 0.565 |
| LightGBM_q90 | 8 | 0.011 | 0.014 | 0.50 | 0.56 | 0.081 | -0.230 | 0.000 |

**Verdict:**

> No model demonstrates exploitable edge over the Drift baseline (net Sharpe 0.56) on GC=F
> daily returns over 2010-01-01–2024-12-31 (8 OOS folds, 5.0 bps round-trip cost, 1-day
> horizon). This is the expected and honest result for a 1-day-ahead forecast on a liquid
> futures market: after transaction costs, none of the tested classifiers consistently
> identifies direction well enough to generate positive risk-adjusted returns. The harness is
> operating correctly; the absence of edge is informative, not a failure.

**No leakage flags.** All models within expected IC/Sharpe range for an efficient market.

---

## Test counts

| File | Tests | Status |
|------|------:|--------|
| tests/test_config.py | 7 | PASS |
| tests/test_splitter.py | 23 | PASS |
| tests/test_metrics.py | 31 | PASS |
| tests/test_features.py | 35 | PASS |
| tests/test_targets.py | 16 | PASS |
| tests/test_models.py | 32 | PASS |
| tests/test_reporting.py | 33 | PASS |
| **Total** | **177** | **PASS** |

---

## Decisions made

1. **63-day rolling window.** One quarter of trading days gives stable estimates without
   hiding regime changes. Can be overridden with `--rolling-window N` at runtime.

2. **Skip constant-prediction models in rolling IC.** RandomWalk (always 0) and Drift (constant
   per fold) produce NaN rolling IC. Including them clutters the plot without information.
   The equity curve figure shows them.

3. **Fold boundaries as dotted lines on rolling IC.** The 63-day rolling window is shorter
   than the fold size (252), so the IC is computed entirely within each fold for most of the
   fold's duration. The boundary markers make it easy to see fold-level breaks.

4. **MD5 over sorted JSON.** The hash is built from `json.dumps(payload, sort_keys=True)`,
   which is deterministic across Python sessions. MD5 is not cryptographic here — it's a
   fingerprint. The first 8 chars give 16^8 ≈ 4 billion distinct values.

5. **Three-branch verdict.** Leakage → Edge → No edge in strict priority. If a model triggers
   a leakage flag, the verdict calls that out regardless of whether it technically "beats"
   the baseline. This prevents a leaked result from being reported as genuine edge.

6. **Report filename contains hash + date.** `backtest_{ticker}_{date}_{hash8}.md` lets you
   sort by name and immediately see: same hash = same config, different dates = same
   experiment over time; different hash = config changed.

---

## What to watch for in future runs

- Re-running with a changed config (e.g. adding features) will produce a new hash. Keep
  the old reports — they form the baseline for the config comparison.
- If a future model triggers the leakage flag (IC > 0.05), the verdict will say so
  explicitly and `run_backtest.py` will emit a WARNING log line. That is working as intended.
- The rolling IC plot for LightGBM shows mostly negative IC with no regime concentration,
  consistent with no exploitable signal after the ETF timing fix.

---

**Ready for Phase 6 review.**
