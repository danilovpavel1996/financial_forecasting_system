# Financial Forecasting System

A **cross-sectional forex ranking system** that forecasts forward 5-day returns
across 15 currency pairs inside a leakage-free, cost-aware, walk-forward
evaluation harness, and trades them automatically on an MT5 **demo** account.

Each week the model ranks the 15 pairs, goes long the top 3 and short the
bottom 3 at 0.01 lots, net of transaction costs. Every model is benchmarked
against RandomWalk and Drift baselines.

**Backtest (Phase 20 OOS 2005–2024):** LightGBM h=5, Sharpe 1.32, mean
cross-sectional rank IC +0.071.
**Live so far:** see the dashboard — as of 2026-08-31, 10 weeks in, the mean
weekly rank IC is −0.088 (SE 0.138), i.e. no edge demonstrated yet.

> **This is research tooling, not investment advice.**

---

## Setup

### 1. Python environment

Python 3.11+ required.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Secrets

```bash
cp .env.example .env
```

Then fill in:

- `FRED_API_KEY` — free key from <https://fred.stlouisfed.org/docs/api/api_key.html>
- `METAAPI_TOKEN`, `METAAPI_ACCOUNT_ID` — from <https://app.metaapi.cloud>,
  for placing orders on the MT5 demo account

### 3. Verify

```bash
pytest
```

---

## The weekly routine (automated)

Railway runs `scripts/railway_cron.sh` every **Friday at 15:00 UTC**:

1. `rebalance.py` — fetch prices, retrain, write `outputs/signals/signal_forex_<date>.json`
2. `execute_mt5.py --live` — reconcile against the account's real positions and trade via MetaApi
3. `fetch_mt5_history.py --undeploy` — pull trade history + account snapshot, then stop the MetaApi terminal to halt hourly billing
4. `live_report.py` — write `outputs/reports/live_report_<date>.md`
5. push the artifacts back to GitHub (needs `GITHUB_TOKEN`)
6. `preflight_check.py` — fail the run (→ Railway push notification) if the demo
   account is near expiry, the MetaApi balance is low, or `GITHUB_TOKEN` is missing

Nothing is required locally. To review results:

```bash
git pull
.venv/bin/streamlit run app.py
```

Any of the steps can also be run by hand; `execute_mt5.py` is **dry-run by
default** and needs `--live` to send orders.

---

## Dashboard

```bash
.venv/bin/streamlit run app.py
```

One page: current positions, weekly rank IC with the backtest reference,
execution fidelity, realized and floating P&L, and the full trade history.
It reads the files the cron produced, and shares all its calculations with
`live_report.py` via `src/live/report.py`, so page and report cannot disagree.

---

## Key scripts

```
scripts/rebalance.py                — generate the weekly signal
scripts/execute_mt5.py              — reconcile and trade via MetaApi (dry-run by default)
scripts/fetch_mt5_history.py        — pull trade history + account snapshot
scripts/live_report.py              — write the markdown live report
scripts/preflight_check.py          — weekly health checks that page via Railway
scripts/run_forex.py                — forex backtest
scripts/run_sweep_forex_expanded.py — hyperparameter sweep over the 15-pair universe
```

---

## Project structure

```
config/         — config.yaml (universe, dates, horizon, costs, splitter)
src/            — all production Python (no logic in notebooks)
  config.py     — load + validate config
  data/         — yfinance + FRED loaders, caching
  features/     — leakage-free feature engineering
  eval/         — walk-forward splitter, metrics, backtester
  models/       — baselines, ElasticNet, LightGBM
  live/         — live signal generation, training, and report data layer
  pipeline_ranking_forex.py — the forex research/backtest harness
scripts/        — CLI entry points (see above)
app.py          — Streamlit live dashboard
tests/          — pytest suite; the leakage tests are the critical ones
outputs/        — signals, execution receipts, reports, models
data/live/      — MT5 trade history + account snapshot
```

---

## Status

**20 phases completed. 309 tests passing.** See `PROJECT_PLAN.md` for the
research roadmap; `PHASE_*.md` document each phase.

The commodity, equity-sector and crypto pipelines were explored in phases 1–20
and removed from the working tree in favour of the forex system that is
actually traded — they remain in git history.
