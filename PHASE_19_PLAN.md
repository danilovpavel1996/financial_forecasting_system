# PHASE_19_PLAN.md — Unified Dashboard Overhaul

> Read CLAUDE.md first. All non-negotiable principles still apply.
> This is a UI/UX phase — no new models or research. The goal is to make
> the dashboard a complete, unified control center for all signals.

---

## 1. Problem

The dashboard was built incrementally across 18 phases. It has gaps:
- Live Signal page only supports Commodities + Equity Sectors, not Forex
- No way to view the Cross-Asset (Commodity + Forex) blend
- No historical equity curve on the Live Signal page
- No forward forecast visualization ("what does the model predict for
  the next H days?")
- Run Experiment page doesn't know about forex or prediction averaging
- Stats and reference Sharpes are inconsistent across pages

---

## 2. What the operator wants to see

### On the Live Signal page (primary daily-use page):

**Signal selector** — choose any of these 4 signals:
1. Commodity LightGBM h=63 (Sharpe 0.79)
2. Forex LightGBM PredAvg21 h=5 (Sharpe 0.94)
3. Equity Sectors B3 LightGBM h=63 (Sharpe 0.41)
4. Cross-Asset Blend: Commodity + Forex (Sharpe 1.05)

For EACH selected signal, show:
- **Today's ranking table** (existing — color-coded LONG/SHORT/FLAT)
- **Historical equity curve** (backtest OOS performance up to the
  end of training data, showing how this signal performed)
- **Forward forecast chart** — for each LONG/SHORT pick, show:
  - Last 90 days of actual prices (existing)
  - PLUS a projected line extending H days into the future based on
    the model's predicted return
  - The projection is: `future_price = current_price × exp(pred_return)`
  - Show this as a DASHED line extending from today's close
  - Include a shaded band for ±1 standard deviation of historical
    prediction error (to show uncertainty honestly)

### On the Run Experiment page:

**Universe selector** — Commodities / Equity Sectors / Forex
**Prediction averaging** — checkbox + window slider (for B3 variant)
**LambdaMART** — add to model selector
Update the reference baseline text for each universe.

---

## 3. Implementation

### 3a. Unified signal config

Create a `dashboard/signal_configs.py` that defines all 4 signals:

```python
SIGNALS = {
    "Commodity LightGBM h=63": {
        "universe": "commodities",
        "model": "LightGBM",
        "horizon": 63,
        "pred_avg_window": 1,
        "use_cot": False,
        "backtest_sharpe": 0.79,
        "backtest_cs_ric": 0.055,
        "description": "Quarterly commodity futures ranking",
    },
    "Forex LightGBM h=5": {
        "universe": "forex",
        "model": "LightGBM",
        "horizon": 5,
        "pred_avg_window": 21,
        "use_cot": False,
        "backtest_sharpe": 0.94,
        "backtest_cs_ric": 0.060,
        "description": "Weekly forex pair ranking with prediction smoothing",
    },
    "Equity B3 LightGBM h=63": {
        "universe": "equity_sectors",
        "model": "LightGBM",
        "horizon": 63,
        "pred_avg_window": 21,
        "use_cot": False,
        "backtest_sharpe": 0.41,
        "backtest_cs_ric": 0.100,
        "description": "Quarterly sector rotation with prediction smoothing",
    },
    "Cross-Asset Blend": {
        "type": "ensemble",
        "components": ["Commodity LightGBM h=63", "Forex LightGBM h=5"],
        "weights": [0.50, 0.50],
        "backtest_sharpe": 1.05,
        "description": "50/50 commodity + forex, cross-asset diversification",
    },
}
```

This single config drives the entire Live Signal page — no hardcoded
signal logic scattered across the dashboard.

### 3b. Historical equity curve on Live Signal page

After showing today's ranking, add a section "Historical Performance"
that shows the OOS equity curve from the most recent backtest run.

Implementation: when the user clicks "Refresh Signal", also run a
quick backtest (or load cached results from `outputs/reports/`) and
display the equity curve inline. If a cached backtest report exists
for the selected signal configuration, load it; otherwise offer a
"Run Backtest" button.

### 3c. Forward forecast visualization

For each LONG and SHORT pick, extend the 90-day price chart with a
forecast line:

```python
# Current close price
current_price = prices[ticker].iloc[-1]

# Model's predicted forward return
pred_return = signal.rankings[ticker].predicted_return

# Projected price at t+horizon
projected_price = current_price * np.exp(pred_return)

# Interpolate linearly from today to t+horizon for the chart
forecast_dates = pd.bdate_range(today, periods=horizon+1)[1:]
forecast_prices = np.linspace(current_price, projected_price, len(forecast_dates))
```

Show the forecast as:
- **Dashed line** from today's close to the projected price
- **Green dashed** for LONG picks (expecting price to rise)
- **Red dashed** for SHORT picks (expecting price to fall)
- **Shaded band** showing ±1σ of historical prediction errors
  (computed from the backtest: std of (predicted - realized) returns)

### 3d. Run Experiment page updates

Add to the sidebar:
- **Universe selector:** Commodities / Equity Sectors / Forex
  (each runs its respective pipeline)
- **Prediction averaging:** checkbox + window input (default 21)
- **LambdaMART:** add to model multi-select
- Update the reference baseline text dynamically based on universe

### 3e. Signal history enhancement

The existing Signal History section shows past JSON signals. Enhance:
- Show a small table of signals per day, per universe
- A "track record" column: for signals old enough (> horizon days ago),
  show whether the LONG picks actually outperformed the SHORT picks
  (realized P&L of the signal)

---

## 4. Execution order

1. Create `dashboard/signal_configs.py` with all 4 signal definitions
2. Rebuild `pages/4_Live_Signal.py`:
   - Signal selector dropdown (4 options)
   - Today's ranking table (per signal)
   - Forward forecast chart (dashed projection + uncertainty band)
   - Historical equity curve section
   - Signal context (backtest Sharpe, CS-RIC, vol regime)
3. Update `app.py` (Run Experiment):
   - Universe selector
   - Prediction averaging controls
   - LambdaMART in model list
4. Update signal history with track-record column
5. Test all pages load without errors
6. Write phase19_summary.md

---

## 5. Definition of done

- [ ] All 4 signals selectable on Live Signal page
- [ ] Forward forecast chart shows dashed projection for each pick
- [ ] Historical equity curve visible for each signal
- [ ] Run Experiment supports all 3 universes + prediction averaging
- [ ] Signal history shows realized P&L for past signals (where available)
- [ ] All existing tests pass (299)
- [ ] `phase19_summary.md` with screenshots described

---

## 6. Design guidance

- Keep the page clean — don't show everything at once. Use tabs or
  expanders for historical vs forecast vs signal details.
- The forecast chart is the hero element — it should be prominent
  and immediately legible.
- Use consistent color coding: green=LONG, red=SHORT, gray=FLAT
  across ALL pages.
- The dashed forecast line should be clearly distinguishable from
  actual price history (dashed, slightly transparent).
- Show the disclaimer prominently near the forecast chart:
  "Projected prices are model estimates, not predictions.
  Historical accuracy: [CS-RIC]. Research tooling only."

---

*Research tooling only — not investment advice.*
