# PROJECT_PLAN.md — Financial Forecasting System

> Read `CLAUDE.md` first. It contains the non-negotiable principles that apply
> to every phase below. This document is the **what and in what order**.

---

## 1. Mission & scope

Build a research system that forecasts **forward daily returns** of a basket of
assets — starting with precious metals — and evaluates those forecasts in a
**leakage-free, cost-aware, walk-forward harness**. The system is built in
layers so each piece (data, features, models, evaluation) is independently
swappable and improvable over time.

**In scope now:** data ingestion from APIs, feature engineering, the evaluation
harness, baseline + classical ML models, and an honest reporting layer.

**Explicitly later (do NOT build yet):** regime-switching models, deep learning
(TFT/PatchTST), cross-sectional ranking, live trading, portfolio optimization.
These are noted in §9 so the architecture leaves room for them.

**Primary target:** forward log return over a configurable `horizon` (default 1
trading day; also support 5).

**Primary universe (configurable):**
- Metals: `GC=F` (gold), `SI=F` (silver), `PL=F` (platinum), `PA=F` (palladium)
- Metal ETFs / miners: `GLD`, `SLV`, `GDX`
- Macro/context: `SPY`, `TLT`, `UUP` (USD ETF), `^VIX`
- Crypto (optional): `BTC-USD`, `ETH-USD`

---

## 2. Data sources (free, no paid vendor required)

### Prices — `yfinance`
Daily OHLCV for all tickers above. No API key. Cache raw pulls to
`data/raw/` as parquet, keyed by ticker + date range, so re-runs are offline
and reproducible.

### Macro drivers — FRED (`fredapi`, free key)
These are the variables that actually move gold; fetch as daily series and
forward-fill to trading days (carefully — see leakage note):
- `DFII10` — 10Y TIPS real yield (the single most important gold driver)
- `T10YIE` — 10Y breakeven inflation expectations
- `DGS10` — 10Y nominal Treasury yield
- `DTWEXBGS` — broad trade-weighted USD index
- `VIXCLS` — VIX (risk appetite)

Key handling: read `FRED_API_KEY` from a gitignored `.env`. Document setup in
the README.

### Crypto (optional, later) — `ccxt` or CoinGecko
Only if/when crypto is added to the live universe. Not required for Phase 1.

**Leakage note on macro data:** FRED series are released with a lag and revised.
For now, lag every macro feature by at least 1 trading day before use, and add a
`TODO` noting that point-in-time/vintage data would be the rigorous fix later.

---

## 3. Target project structure

Create exactly this structure in Phase 0:

```
financial_forecasting_system/
├── README.md
├── CLAUDE.md                  # already present — do not overwrite
├── PROJECT_PLAN.md            # already present — do not overwrite
├── requirements.txt
├── pyproject.toml             # optional but preferred (editable install)
├── .gitignore                 # include .env, data/, outputs/models/, __pycache__
├── .env.example               # FRED_API_KEY=your_key_here
├── config/
│   └── config.yaml            # universe, dates, horizon, costs, splitter params
├── seed/
│   └── eval_harness_reference.py   # the human's reference impl — PORT, don't import
├── data/
│   ├── raw/                   # cached API pulls (gitignored)
│   └── processed/             # feature matrices, targets (gitignored)
├── src/
│   ├── __init__.py
│   ├── config.py              # load + validate config.yaml
│   ├── data/
│   │   ├── __init__.py
│   │   ├── universe.py        # asset/series definitions
│   │   ├── prices.py          # yfinance loader + cache
│   │   └── macro.py           # FRED loader + cache
│   ├── features/
│   │   ├── __init__.py
│   │   ├── price_features.py  # returns, momentum, vol, ratios (all lagged)
│   │   └── macro_features.py  # real-yield change, DXY change, etc. (all lagged)
│   ├── targets.py             # forward_log_return
│   ├── eval/
│   │   ├── __init__.py
│   │   ├── splitter.py        # WalkForwardSplitter (purge/embargo)
│   │   ├── metrics.py         # IC, rank-IC, hit rate, Sharpe, max DD
│   │   └── backtester.py      # orchestrator + model comparison
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py            # Model protocol (fit/predict)
│   │   ├── baselines.py       # RandomWalk, Drift, Momentum
│   │   ├── linear.py          # ElasticNet wrapper w/ train-only scaling
│   │   └── gbm.py             # LightGBM wrapper
│   └── pipeline.py            # end-to-end: load → features → target → backtest
├── scripts/
│   ├── fetch_data.py          # CLI: pull + cache all raw data
│   └── run_backtest.py        # CLI: run full pipeline, write report
├── outputs/
│   ├── figures/
│   ├── reports/               # phaseN_summary.md + backtest reports
│   └── models/                # serialized fitted models (gitignored)
├── notebooks/                 # exploration only — no production logic here
└── tests/
    ├── test_splitter.py       # CRITICAL leakage tests
    ├── test_targets.py        # forward return correctness + NaN handling
    └── test_metrics.py        # IC/Sharpe sanity on synthetic data
```

---

## 4. Phased build plan

Work **one phase at a time**. End each phase with `outputs/reports/phaseN_summary.md`
and STOP for human review.

### Phase 0 — Scaffolding
**Goal:** repo skeleton, environment, config, git hygiene.
**Tasks:** create the tree in §3; `requirements.txt` (pinned); `pyproject.toml`;
`.gitignore`; `.env.example`; a `config/config.yaml` with universe, start/end
dates, `horizon`, `cost_bps`, and splitter params; `src/config.py` that loads
and validates it; a stub `README.md`.
**Definition of done:** `pip install -e .` works; `python -c "from src.config import load_config; print(load_config())"` prints the parsed config; `pytest` runs (even with 0 tests) without import errors.

### Phase 1 — Data ingestion
**Goal:** reproducible, cached pulls from yfinance + FRED.
**Tasks:** `src/data/universe.py` (ticker/series registry); `src/data/prices.py`
(yfinance loader → tidy daily DataFrame, cached to parquet); `src/data/macro.py`
(FRED loader, cached); `scripts/fetch_data.py` CLI to populate `data/raw/`.
Handle missing days, holidays, and align everything to a common trading-day index.
**Definition of done:** `python scripts/fetch_data.py` produces cached parquet
for every configured asset; a second run reads from cache (offline); a summary
prints date ranges and row counts per series. No NaN-filled gaps silently
introduced (log them).

### Phase 2 — Evaluation harness (the moat)
**Goal:** port `seed/eval_harness_reference.py` into `src/eval/` as clean,
tested modules. **Do not import the seed file** — re-implement and improve it.
**Tasks:** `splitter.py` (WalkForwardSplitter with embargo); `metrics.py` (IC,
rank-IC, directional accuracy, net-of-cost Sharpe, max drawdown, turnover);
`backtester.py` (runs a model_factory across folds, pools predictions, compares
against baselines). Write `tests/test_splitter.py` and `tests/test_metrics.py`.
**Definition of done — these tests MUST pass:**
- A test asserts that for every fold, `max(train_idx) < min(test_idx) - embargo`.
- A test asserts that with `embargo >= horizon`, no training label's look-ahead
  window overlaps any test index.
- A test asserts metrics on a known synthetic series match hand-computed values.
- The baselines run and appear in the comparison table.

### Phase 3 — Feature engineering
**Goal:** a leakage-free feature matrix.
**Tasks:** `price_features.py` — lagged returns (1/5/10/21d), rolling volatility,
momentum, gold/silver ratio, distance-from-moving-average, all **strictly using
data ≤ t**. `macro_features.py` — daily changes in real yield, breakevens, DXY,
VIX level/change, all lagged ≥ 1 day. Z-score normalization must be **fit on
train only** inside the harness, not globally.
**Definition of done:** a test confirms no feature column at row `t` depends on
any value after `t` (shift-and-compare check); feature matrix aligns 1:1 with
the target index; `data/processed/` holds the cached matrix.

### Phase 4 — Targets, baselines, first honest backtest
**Goal:** the first end-to-end run on REAL data.
**Tasks:** `targets.py` (`forward_log_return`); wire `pipeline.py` to go
load → features → target → backtest with baselines only; `scripts/run_backtest.py`
writes a comparison table + an equity-curve figure to `outputs/`.
**Definition of done:** running on real gold (`GC=F`) daily data produces the
baseline comparison table and an `outputs/reports/` markdown that honestly
states the baseline IC/Sharpe. Expect near-zero edge — that is correct.

### Phase 5 — Classical models
**Goal:** add real models and compare honestly.
**Tasks:** `models/base.py` (protocol); `models/linear.py` (ElasticNet with
train-only `StandardScaler` in a pipeline); `models/gbm.py` (LightGBM with
sane regularization and early stopping on an inner time-split — no shuffling).
Add a quantile variant (LightGBM quantile objective) to produce prediction
intervals, not just point estimates.
**Definition of done:** all models appear in one comparison table vs. baselines,
with mean IC, rank-IC, IC-stability, net Sharpe, turnover, max DD. The phase
summary states plainly which (if any) beat the baselines and by how much, and
flags anything that looks too good (possible leakage).

### Phase 6 — Reporting & monitoring
**Goal:** make results legible to the human across runs.
**Tasks:** a reporting module that, per run, writes: the comparison table, an
equity curve, a rolling-IC plot, and a one-paragraph plain-English verdict.
Tag each report with config hash + date so runs are comparable over time.
**Definition of done:** one command (`python scripts/run_backtest.py`) regenerates
a full, timestamped report under `outputs/reports/`.

---

## 5. Conventions

- **Python style:** type hints, docstrings, `dataclass` for configs/results.
- **No notebooks in the critical path.** Notebooks are for exploration; anything
  that matters gets moved into `src/` with a test.
- **Determinism:** seed numpy/sklearn/lightgbm; log the seed in every report.
- **Logging over printing** in `src/`; CLIs may print summaries.
- **Fail loud on silent data corruption:** assert index alignment, assert no
  unexpected NaN, assert no duplicated dates.

---

## 6. How the human monitors this

After each phase, the human reads `outputs/reports/phaseN_summary.md`, runs the
tests, and (when results exist) eyeballs the comparison table and equity curve.
The human will bring results back to a separate planning chat for guidance when
stuck. So: **phase summaries must be self-contained and honest** — state what
was decided, what the numbers were, and what is uncertain.

---

## 7. First session instructions for Claude Code

1. Read `CLAUDE.md` and this file fully.
2. Confirm Python version and create the virtualenv / install path.
3. Execute **Phase 0 only**. Then write `outputs/reports/phase0_summary.md` and
   stop for review. Do not start Phase 1 until told to continue.

---

## 8. Acceptance-criteria philosophy

Every phase has a "definition of done" that is **testable**, not vibes. The
leakage tests in Phase 2 are the backbone — if they ever fail, all downstream
results are invalid and must be treated as such. When in doubt between
"impressive" and "honest," choose honest.

---

## 9. Out of scope now — architecture must leave room for it later

- **Regime layer:** HMM / vol-state classifier as a gating feature or position
  sizer. The backtester should later accept a per-sample regime label.
- **Deep models:** TFT / PatchTST once daily sample counts justify it; they plug
  in via the same `Model` protocol (`fit`/`predict`).
- **Cross-sectional ranking:** predict relative ordering across the basket; the
  metrics module should later support cross-sectional rank-IC.
- **Volatility forecasting:** a parallel, easier target (GARCH/HAR) that feeds
  position sizing.
- **Live/paper integration & portfolio construction:** far future, out of scope.
```
