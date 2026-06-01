# PHASE_11_PLAN.md — Research Dashboard (Streamlit)

> Read CLAUDE.md first. All non-negotiable principles still apply.
> This adds a web UI so the operator can run experiments, view results,
> and compare configurations without touching the terminal.

---

## 1. Why a dashboard

The system now has 10 phases of infrastructure: data loaders, feature
pipelines, evaluation harness, ranking backtester, vol targeting. Running
experiments requires editing config.yaml, typing terminal commands, and
reading markdown reports. A dashboard makes this interactive:

- Change parameters (horizon, vol target, COT on/off) with sliders/toggles
- Launch a backtest with one click
- View comparison tables, equity curves, and rolling-IC plots inline
- Browse and compare past runs by config hash
- Monitor vol forecasts and feature quality

---

## 2. Tech stack

**Streamlit** — pure Python, installs with pip, runs locally via
`streamlit run app.py`. No React, no frontend build step, no API server.
Integrates natively with pandas DataFrames, matplotlib/plotly charts, and
the existing src/ modules.

Add to requirements.txt:
```
streamlit>=1.45
plotly>=6.0
```

---

## 3. Dashboard layout (pages)

Streamlit supports multi-page apps via a `pages/` directory. Build these
pages in order:

### Page 1: Run Experiment (the main page — `app.py`)

**Sidebar controls:**
- Horizon: slider [1, 5, 10, 21, 63] (default 5)
- Vol targeting: checkbox + slider for target vol (0.05–0.20, step 0.01)
- Max leverage: number input (default 2.0)
- COT features: checkbox (default off — Phase 8 baseline is best)
- Models to run: multi-select checkboxes
  [MeanReversion, MomentumRank, ElasticNet, LightGBM] (default all)
- Asset universe: show the 9 tickers, allow deselecting individual ones
- "Run Backtest" button

**Main area (after run completes):**
- Config hash + timestamp at the top
- Comparison table (styled: best Sharpe highlighted, leakage flags in red)
- Equity curve chart (plotly, interactive — zoom, hover for dates)
- Rolling CS-RIC chart per model (plotly)
- One-paragraph auto-generated verdict (reuse `build_verdict` from
  src/reporting.py, adapted for ranking)
- "Save this run" button → writes report to outputs/reports/

**While running:**
- Streamlit spinner with "Running backtest..." message
- Stream the log output (optional: use st.status or st.expander for logs)

### Page 2: Run History (`pages/1_Run_History.py`)

- Scan `outputs/reports/` for all past backtest markdown reports
- Show a table: date, config hash, horizon, vol target, best model, best
  Sharpe, best CS-RIC
- Click a row → expand the full markdown report inline
- Side-by-side comparison: select 2 runs → show their tables side by side

### Page 3: Vol Forecast Monitor (`pages/2_Vol_Monitor.py`)

- For each of the 9 commodities, show:
  - EWMA 1-day vol forecast (time series chart)
  - Trailing 21-day realized vol overlay
  - Current vol regime (percentile of historical range)
- EWMA/GARCH correlation stats (from scripts/eval_vol.py logic)
- This page reads from cached price data — no backtest needed

### Page 4: Data Explorer (`pages/3_Data_Explorer.py`)

- Show the raw data coverage: table of tickers, row counts, date ranges
- Price charts per commodity (close price, log scale)
- COT positioning charts: mm_net and mm_net_pct per commodity
- Correlation heatmap of daily returns across the 9 commodities
- Feature matrix preview: first/last 20 rows, column stats

---

## 4. Architecture

### 4a. Do NOT duplicate business logic

The dashboard calls the existing `src/` modules directly:
- `src.config.load_config()` for defaults
- `src.pipeline_ranking.run_ranking_pipeline()` for backtests
- `src.eval.rank_backtester.ranking_comparison_table()` for results
- `src.models.vol_forecast.ewma_vol()` for vol charts
- `src.data.prices.fetch_ticker()` for price charts

The dashboard is a **thin UI layer** over the existing pipeline. No
business logic lives in the Streamlit code — it only handles layout,
user input, and chart rendering.

### 4b. Config override pattern

The user's sidebar selections override config.yaml values at runtime.
Build a helper that takes the loaded Config + sidebar overrides and
returns a modified Config object (or passes kwargs directly to
`run_ranking_pipeline`). Never write to config.yaml from the dashboard.

### 4c. Caching

Use `@st.cache_data` for expensive operations:
- Price data loading (cached by ticker + date range)
- Feature matrix building (cached by config hash)
- Past-run scanning (cached with TTL=60s)

Do NOT cache backtest results — each run should be fresh.

### 4d. File structure

```
financial_forecasting_system/
├── app.py                          # Main page: Run Experiment
├── pages/
│   ├── 1_Run_History.py
│   ├── 2_Vol_Monitor.py
│   └── 3_Data_Explorer.py
├── dashboard/                      # Dashboard helpers (thin)
│   ├── __init__.py
│   ├── charts.py                   # Plotly chart builders
│   ├── config_override.py          # Sidebar → Config translation
│   └── run_history.py              # Scan + parse past reports
└── ... (existing src/, scripts/, etc.)
```

---

## 5. Execution order

Build one page at a time. Each page should be runnable independently.

1. **app.py** — the Run Experiment page. This is the core deliverable.
   Definition of done: sidebar controls work, clicking "Run Backtest"
   executes the pipeline and displays the comparison table + equity curve.

2. **pages/3_Data_Explorer.py** — read-only, no backtest needed. Quick to
   build, immediately useful for understanding the data.

3. **pages/2_Vol_Monitor.py** — read-only, displays EWMA vol forecasts.

4. **pages/1_Run_History.py** — scans outputs/reports/ and displays past
   runs with comparison.

### After each page, verify:
```bash
streamlit run app.py
```
opens in the browser and the page works without errors.

---

## 6. Design guidance

- **Dark theme friendly:** Streamlit supports dark mode. Use plotly charts
  (they adapt automatically) rather than matplotlib (needs manual styling).
- **Responsive:** Streamlit handles this natively.
- **No auto-run on page load.** The backtest only runs when the user clicks
  "Run Backtest." Page load should be fast (< 2s).
- **Error handling:** if the pipeline fails (e.g., missing data), catch the
  exception and display it in a red st.error() box. Don't crash the app.
- **Disclaimer:** show "Research tooling — not investment advice" in the
  sidebar footer on every page.

---

## 7. Definition of done

- [ ] `streamlit run app.py` opens a working dashboard.
- [ ] Run Experiment page: sidebar controls → run backtest → comparison
      table + equity curve + verdict displayed.
- [ ] Data Explorer page: price charts, COT charts, correlation heatmap.
- [ ] Vol Monitor page: EWMA forecasts per commodity.
- [ ] Run History page: browse and compare past runs.
- [ ] No business logic in dashboard code — all calls go through src/.
- [ ] All 259 existing tests still pass (dashboard adds no new tests to
      the test suite — it's a UI layer).
- [ ] `phase11_summary.md` with screenshots described and STOP.

---

## 8. What NOT to build

- **No authentication / user management.** This runs locally.
- **No live data streaming.** Data is cached from API pulls.
- **No trade execution.** This is research, not a trading platform.
- **No database.** Past runs are markdown files on disk.

---

*Research tooling only — not investment advice.*
