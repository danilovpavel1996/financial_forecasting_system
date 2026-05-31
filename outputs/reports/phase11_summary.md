# Phase 11 Summary — Research Dashboard (Streamlit)

**Date:** 2026-05-31  
**Tests:** 259/259 pass (0 new tests — UI layer adds no business logic to test)

---

## What was built

| File | Role |
|------|------|
| `app.py` | Main page: Run Experiment |
| `pages/1_Run_History.py` | Browse and compare past reports |
| `pages/2_Vol_Monitor.py` | EWMA vol forecasts per commodity |
| `pages/3_Data_Explorer.py` | Prices, COT charts, correlation heatmap |
| `dashboard/charts.py` | Plotly chart builders (no Streamlit imports) |
| `dashboard/config_override.py` | Sidebar → pipeline kwargs translation |
| `dashboard/run_history.py` | Scan/parse past report markdown files |

---

## Page descriptions

### app.py — Run Experiment

**Sidebar controls:**
- Horizon select slider: [1, 5, 10, 21, 63] trading days
- Vol targeting toggle + target vol slider (5%–20%) + max leverage + lookback
- COT features checkbox (default off — Phase 8 baseline is cleanest)
- Model multi-select: EqualWeight / MomentumRank / MeanReversion / ElasticNet / LightGBM
- "▶ Run Backtest" primary button

**Main area after run:**
- Styled comparison table (best Sharpe highlighted green; cells > leakage threshold flagged red)
- Auto-generated verdict paragraph (`build_verdict()` in charts.py)
- Equity curve chart (plotly, interactive hover/zoom)
- Rolling 63-day CS-RIC chart
- Position scale factor chart (only when vol targeting is on)
- Per-model metric cards (Sharpe, CS-RIC, Ann. return, Max DD, Turnover)
- "💾 Save run report" button → writes markdown to `outputs/reports/`

**Architecture notes:**
- Backtest result stored in `st.session_state` — widget interaction does NOT re-run the pipeline
- Pipeline call only on button click, wrapped in `st.spinner()`
- All errors caught and shown in `st.error()` boxes — no crashes

### pages/3_Data_Explorer.py

Tabs:
1. **Coverage** — table of ticker row counts, date ranges, COT weeks
2. **Prices** — interactive close price chart (log/linear toggle), ann. vol / total return metrics
3. **COT Positioning** — mm_net bar chart + 52-week percentile subplot per commodity; 52w high/low metrics
4. **Correlations** — Pearson heatmap of daily log-returns; descriptive stats table

### pages/2_Vol_Monitor.py

- Summary table: current EWMA vol, current realized vol, vol regime percentile, forecast correlation
- Individual ticker chart: EWMA vs realized vol overlay (adjustable lambda and window via sidebar)
- Current vol regime metric (percentile vs full history)
- All-commodities overview chart (9 series in one figure)
- Lambda and realized vol window are adjustable in sidebar; charts regenerate via `@st.cache_data`

### pages/1_Run_History.py

- Scans `outputs/reports/*.md` (TTL=60s cache)
- Index table: filename, date, phase, horizon, title, size
- Single report viewer: full markdown rendered inline
- Side-by-side comparison: select any two reports; content shown in two columns

---

## Architecture: thin UI layer verified

The dashboard contains **zero business logic**. Every computation delegates to `src/`:

| Dashboard action | src/ call |
|-----------------|-----------|
| Run backtest | `src.pipeline_ranking.run_ranking_pipeline()` |
| Comparison table | `src.eval.rank_backtester.ranking_comparison_table()` |
| EWMA vol forecast | `src.models.vol_forecast.EWMAVolForecast.fit_predict()` |
| Realized vol | `src.models.vol_forecast.realized_vol()` |
| Correlation calc | `numpy` + `pandas` (pure math, no domain logic) |
| Load prices | `src.data.prices.fetch_all_tickers()` |
| Load COT | `src.data.cot.fetch_all_cot()` |
| Config | `src.config.load_config()` |

The `dashboard/` modules (`charts.py`, `config_override.py`, `run_history.py`) contain only:
- Plotly figure construction
- Sidebar → kwargs translation
- Filesystem scanning and markdown parsing

---

## Caching strategy

| Data | Cache policy |
|------|-------------|
| Price data | `@st.cache_data` — keyed by ticker list + date range |
| COT data | `@st.cache_data` — keyed by code dict + date range |
| EWMA/realized vol | `@st.cache_data` — keyed by lambda + window |
| Backtest results | NOT cached — run on button click only, stored in `st.session_state` |
| Report scans | `@st.cache_data(ttl=60)` — refreshes every 60s or on manual refresh |

---

## How to run

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. All four pages are accessible via the left sidebar navigation.

**First run prerequisites:**
- Data must be fetched first: `python scripts/fetch_data.py`
- `.env` file with `FRED_API_KEY` must exist for macro data

---

## Acceptance criteria

| Criterion | Status |
|-----------|--------|
| `streamlit run app.py` opens working dashboard | ✅ Health endpoint returns `ok` |
| Run Experiment: sidebar → backtest → table + equity curve | ✅ |
| Data Explorer: prices, COT, correlations | ✅ |
| Vol Monitor: EWMA forecasts per commodity | ✅ |
| Run History: browse + compare past runs | ✅ |
| No business logic in dashboard code | ✅ All logic in `src/` |
| 259 existing tests still pass | ✅ |

---

## Key design decisions

1. **`st.session_state` for backtest results.** Stores the `dict[str, RankingResult]` across widget interactions. This prevents the pipeline re-running on every slider movement.

2. **No new tests.** The dashboard is a UI wrapper. Adding pytest tests for Streamlit widgets requires `streamlit.testing.v1` and testing rendering logic — high cost for marginal value. The underlying `src/` functions are already comprehensively tested.

3. **`@st.cache_data` on all data loaders, NOT on the pipeline.** Data loading is deterministic and slow; the pipeline result depends on sidebar parameters that change between runs.

4. **Dark-theme-friendly charts.** All Plotly figures use `paper_bgcolor="rgba(0,0,0,0)"` and `plot_bgcolor="rgba(0,0,0,0)"` — transparent backgrounds adapt to Streamlit's light and dark themes.

5. **COT default off.** Phase 9 showed neutral/negative impact at h=5. The sidebar default reflects the best-performing configuration (Phase 8 baseline).

---

*Research tooling only — not investment advice.*
