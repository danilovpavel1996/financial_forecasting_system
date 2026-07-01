# Financial Forecasting System

A multi-asset **cross-sectional ranking system** that forecasts forward daily
returns across **commodities, forex (15 pairs), equity sectors, and crypto**,
inside a leakage-free, cost-aware, walk-forward evaluation harness.

The system ranks assets cross-sectionally each day and goes long the top /
short the bottom, net of transaction costs. Every model is benchmarked against
RandomWalk and Drift baselines.

**Best signal:** Forex 15-pair LightGBM, horizon h=5, **Sharpe 1.32** (net of costs).

> **This is research tooling, not investment advice.**

---

## Setup

### 1. Python environment

Python 3.11+ required.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 2. FRED API key

Get a free key at <https://fred.stlouisfed.org/docs/api/api_key.html>.

```bash
cp .env.example .env
# edit .env and set FRED_API_KEY=<your_key>
```

### 3. Verify setup

```bash
python -c "from src.config import load_config; print(load_config())"
pytest
```

---

## Weekly trading routine

Generate the current rebalance and MT5 instructions:

```bash
.venv/bin/python scripts/rebalance.py
```

Then follow the printed MT5 instructions to adjust positions.

---

## Dashboard

Interactive research dashboard (backtests, signals, experiments):

```bash
.venv/bin/streamlit run app.py
```

---

## Key scripts

```
scripts/rebalance.py               — weekly MT5 rebalance instructions for the forex portfolio
scripts/run_sweep_forex_expanded.py — hyperparameter sweep over the 15-pair forex universe
scripts/run_cross_asset_ensemble.py — cross-asset ensemble across commodities/forex/sectors/crypto
scripts/live_signal.py             — generate the live daily signal
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
  pipeline.py   — end-to-end orchestration
scripts/        — CLI entry points (rebalance, sweeps, ensembles, live signal)
app.py          — Streamlit research dashboard
tests/          — pytest suite; leakage tests are critical
outputs/        — figures, reports, signals, serialized models
data/           — raw + processed data cache (gitignored)
```

---

## Status

**20 phases completed. 299 tests passing.** See `PROJECT_PLAN.md` for the full
phased roadmap.
