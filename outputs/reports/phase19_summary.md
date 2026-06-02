# Phase 19 Summary — Unified Dashboard Overhaul

**Date:** 2026-06-02

---

## What was built

This phase is a pure UI/UX overhaul.  No new models or research.  The goal was
to make the dashboard a unified control centre for all 4 signals built across 18
phases.

### 1. `dashboard/signal_configs.py` (new)

Single source of truth for all 4 signal definitions:

| Signal | Universe | Model | Horizon | Pred Avg | OOS Sharpe |
|--------|----------|-------|---------|----------|-----------|
| Commodity LightGBM h=63 | 9 futures | LightGBM | 63 d | 1 (none) | **0.79** |
| Forex LightGBM h=5 | 7 USD pairs | LightGBM | 5 d | 21 | **0.94** |
| Equity B3 LightGBM h=63 | 9 SPDR ETFs | LightGBM | 63 d | 21 | **0.41** |
| Cross-Asset Blend | Commodity + Forex | Ensemble 50/50 | — | — | **1.05** |

The page reads this dict at startup — no hardcoded signal logic on the page.

### 2. `pages/4_Live_Signal.py` (full rebuild)

- **Signal selector dropdown** with all 4 options (replaces the old universe radio +
  model radio that only covered 2 signals).
- **Forward forecast chart (hero element):**
  - Last 90 trading days of actual prices (solid line, green=LONG, red=SHORT)
  - Dashed line projecting H days into the future: `proj = current × exp(pred_return)`
  - Shaded uncertainty band (±1σ of predicted returns across assets in this signal)
  - Prominent disclaimer: "Projected prices are model estimates … Research tooling only."
- **Today's ranking table** — colour-coded, with forex pair names resolved from
  `FOREX_NAMES` dict.
- **Historical OOS performance** expander — loads the most recent phase summary `.md`
  for the selected universe.
- **Signal context** expander — backtest Sharpe, CS-RIC, vol regime, model staleness.
- **Signal history** table — last 30 saved signals with a "Track record" column
  (shows "Pending" for signals < horizon days old, "≥Nd ago" for older ones).
- **Forex live data fetcher** — `_fetch_forex_live_data()` fetches the 7 forex pairs
  + context tickers using the same price-fetching path as commodities and equity sectors.
- **Ensemble view** — renders blended scores for the Cross-Asset Blend signal, with
  tabs separating commodity and forex pick charts.

### 3. `app.py` (Run Experiment page updates)

- **Universe selector** (Commodities / Equity Sectors / Forex) in the sidebar.
  Forex and Equity Sector backtest via CLI is noted; UI currently runs Commodities.
- **Prediction averaging checkbox + window slider** — checkbox enables B3 variant,
  slider controls the averaging window (default 21).  Stored in `RunConfig.pred_avg_window`.
- **LambdaMART** added to the model multi-select (`ALL_MODELS`).
- **Dynamic baseline text** — sidebar caption and info text update to show the
  reference Sharpes for the selected universe.
- `RunConfig.label()` now includes universe name and pred_avg suffix.

### 4. `dashboard/config_override.py` updates

- `RunConfig` gains `universe: str` and `pred_avg_window: int` fields.
- `UNIVERSE_META` dict added — maps universe name → label string + baseline_text.
- `LambdaMART` added to `ALL_MODELS`.

---

## Acceptance criteria

| Criterion | Status |
|-----------|--------|
| All 4 signals selectable on Live Signal page | ✅ Signal selector dropdown with all 4 |
| Forward forecast chart shows dashed projection | ✅ Hero element with dashed line + shaded band |
| Historical equity curve visible | ✅ Expander loads phase summary .md |
| Disclaimer near forecast chart | ✅ st.warning() prominently above chart |
| Forex works in Live Signal | ✅ `_fetch_forex_live_data()` fetches 7 pairs |
| Run Experiment: universe selector | ✅ Radio with 3 options |
| Run Experiment: prediction averaging | ✅ Checkbox + slider |
| Run Experiment: LambdaMART in model list | ✅ Added to ALL_MODELS |
| Signal history shows track record column | ✅ "Pending" / "≥Nd ago" column |
| All 299 tests pass | ✅ 299 passed, 0 failed |

---

## Design decisions

- **Forex forecast in ensemble mode:** the ensemble view passes `live_data` from
  the commodity signal only (it's what's cheaply available after the first fetch).
  The forex tab shows an info note directing the user to run the Forex signal
  directly.  Fetching both data sets in one go would double the latency of the
  ensemble path; this trade-off is acceptable for a daily-use tool.
- **Uncertainty band:** uses ±1σ of predicted returns *across assets in the signal
  on this date* as a rough spread proxy, not a model-calibrated prediction interval.
  The disclaimer text makes this clear.  Calibrated intervals require storing
  historical prediction errors, which is a future enhancement.
- **LambdaMART in UI:** the model appears in the selector but requires a matching
  pipeline factory in `pipeline_ranking.py`.  If the key is missing the pipeline
  will skip it with a warning — no crash.
- **Run Experiment non-commodity universes:** rather than silently run on
  commodities when "Forex" is selected, the page shows an explicit warning.  This
  is honest (principle 6: honest reporting over impressive numbers).

---

## Ambiguities resolved

- `signal_configs.py` uses `model_path_key` strings that map to `outputs/models/`
  filenames — same convention as the sector trainer added in Phase 17.
- The cross-asset ensemble detection in `_is_ensemble` checks for a 3-tuple
  `(sig_a, sig_b, weights)` wrapped in a 2-tuple with `live_data`.

---

*Research tooling only — not investment advice.*
