# Financial Forecasting System

A research system for forecasting **forward daily returns** of a basket of assets
(starting with precious metals) inside a leakage-free, cost-aware, walk-forward
evaluation harness.

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
scripts/        — CLI entry points (fetch_data, run_backtest)
tests/          — pytest suite; leakage tests are critical
outputs/        — figures, reports, serialized models (gitignored except reports)
seed/           — reference implementation for porting; do not import
data/           — raw + processed data cache (gitignored)
```

---

## Running

```bash
# Pull and cache all raw data
python scripts/fetch_data.py

# Run full backtest and write report
python scripts/run_backtest.py
```

---

## Phased build plan

See `PROJECT_PLAN.md` for the full phased roadmap. Current status: **Phase 0 complete**.
