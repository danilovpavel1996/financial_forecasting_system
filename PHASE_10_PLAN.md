# PHASE_10_PLAN.md — Volatility Forecasting & Position Sizing

> Read CLAUDE.md first. All non-negotiable principles still apply.
> This adds volatility-based position sizing to the cross-sectional ranking
> strategy from Phase 8 (the best result: MeanReversion Sharpe 0.63 at h=5).

---

## 1. Why volatility forecasting

The Phase 8 ranking strategy uses fixed ±1 positions: long top-2, short
bottom-2, equal weight, every day. This ignores a critical piece of
information: **how noisy the market is right now.**

When commodity vol is low, the ranking signal's signal-to-noise ratio is
higher — the same CS-RIC of 0.03 translates into more reliable P&L.
When vol is high, noise dominates and positions should shrink.

**Volatility targeting** scales the portfolio so that its predicted
annualized volatility is approximately constant (e.g., 10%). This means:
- In calm markets: leverage up (positions > 1) → capture more of the signal
- In volatile markets: scale down (positions < 1) → protect capital

This is the single most widely used risk management technique in
institutional systematic trading. It doesn't change the signal — it
changes how much capital you allocate to it based on predicted risk.

The key fact that makes this tractable: **volatility is genuinely
forecastable.** Unlike returns (mostly noise), vol clusters and
mean-reverts. A simple GARCH(1,1) or EWMA produces reliable 1-day-ahead
vol forecasts with correlation > 0.5 to realized vol.

---

## 2. What to build

### 2a. Volatility forecaster (`src/models/vol_forecast.py`)

Implement two vol forecasting models:

**EWMA (Exponentially Weighted Moving Average):**
```
sigma²_t = lambda * sigma²_{t-1} + (1 - lambda) * r²_{t-1}
```
where lambda = 0.94 (RiskMetrics standard). This is the simplest
approach and often hard to beat. No fitting required — just a
recursive formula applied to past returns.

**GARCH(1,1):**
```
sigma²_t = omega + alpha * r²_{t-1} + beta * sigma²_{t-1}
```
Fit using the `arch` library (already in requirements.txt).
Fitting must be walk-forward: fit on training data, produce forecasts
for test data. Never fit on future data.

Both models produce a 1-day-ahead conditional volatility forecast
for each commodity. For the h=5 ranking strategy, use the AVERAGE of
the 5 daily vol forecasts over the next rebalancing period, or simply
use today's vol forecast as a proxy for the week ahead (vol is
persistent enough that this is reasonable).

### 2b. Portfolio vol targeting (`src/eval/vol_targeting.py`)

Given:
- `positions`: the raw ±1 long/short positions from the ranking strategy
- `vol_forecast`: predicted annualized vol of the long-short portfolio
- `target_vol`: desired annualized vol (default: 10%)

Compute:
```
scaled_positions = positions × (target_vol / vol_forecast)
```

Clip to a maximum leverage of 2.0 (don't lever up more than 2x even if
vol is very low — prevents extreme positions during abnormally calm
periods that precede crashes).

The `vol_forecast` for the portfolio is estimated from the recent
realized vol of the long-short P&L series (trailing 21-day rolling
std, annualized). This is a simple approach that works because
portfolio vol is even more persistent than individual asset vol.

**Alternative (more sophisticated):** forecast each asset's vol
individually, then estimate portfolio vol from the individual forecasts
plus a correlation estimate. This is better but significantly more
complex. Start with the portfolio-level approach.

### 2c. Integration into ranking backtester

Modify `RankingBacktester` (or create a wrapper) to support two modes:
1. `vol_target=None` (default) — existing behavior, fixed ±1 positions
2. `vol_target=0.10` — apply vol targeting as described above

The vol-targeted mode should:
- Compute trailing 21-day realized vol of the L/S P&L at each date
- Scale positions by (target_vol / realized_vol_annualized)
- Clip leverage to [0, 2.0]
- Report the REALIZED vol of the strategy (should be close to target)
- Report turnover (will be higher than fixed due to position resizing)

### 2d. Standalone vol forecast evaluation

Before using vol forecasts for sizing, verify they're actually good.
Build a simple evaluation in `scripts/eval_vol.py`:
- For each of the 9 commodities, compute:
  - EWMA 1-day-ahead vol forecast
  - GARCH(1,1) 1-day-ahead vol forecast (walk-forward fitted)
  - Realized vol (|return| or squared return as proxy)
- Report correlation between predicted and realized vol
- A good EWMA should achieve corr > 0.4; GARCH > 0.5

This validates the vol forecasts are meaningful before plugging them
into position sizing.

---

## 3. Experiment design

### Run 1: Phase 8 baseline (no COT, no vol targeting)

Re-run the Phase 8 configuration (no COT features) at h=5 to get the
clean baseline. Set config back to:
- Remove or disable the cftc section (or pass cot_raw=None)
- horizon=5, embargo=5

### Run 2: Phase 8 + vol targeting

Same as Run 1 but with vol_target=0.10. Compare:
- Sharpe (should improve if vol targeting works)
- Realized annual vol (should be close to 10%)
- Max drawdown (should improve — positions shrink during stress)
- Turnover (will increase slightly from position resizing)

### Run 3: Sensitivity analysis

Test target_vol = [0.05, 0.10, 0.15, 0.20] and report the Sharpe
for each. This shows whether the result is robust to the choice of
target vol or whether one specific value is cherry-picked.

---

## 4. Definition of done

- [ ] EWMA and GARCH vol forecasters implemented and tested.
- [ ] Vol forecast evaluation shows corr > 0.4 with realized vol
      for at least the EWMA model across the 9 commodities.
- [ ] Vol targeting module clips leverage to [0, 2.0].
- [ ] Ranking backtester supports vol_target parameter.
- [ ] Comparison table: baseline vs vol-targeted at h=5 (MeanReversion
      and LightGBM both shown).
- [ ] Sensitivity table: Sharpe across 4 target_vol values.
- [ ] Realized vol of vol-targeted strategy is within ±3% of target.
- [ ] All existing tests still pass + new vol tests.
- [ ] `phase10_summary.md` with honest comparison.

---

## 5. Config changes

Add to `config/config.yaml`:

```yaml
vol_targeting:
  enabled: true
  target_vol: 0.10          # annualized target volatility
  max_leverage: 2.0         # position clip
  lookback_days: 21         # trailing window for realized vol estimate
  method: "ewma"            # "ewma" or "garch"
```

---

## 6. Important: revert to Phase 8 baseline for comparison

Before running any vol-targeting experiment, revert the config to the
Phase 8 state (horizon=5, embargo=5, no COT features or COT disabled).
The Phase 8 MeanReversion Sharpe of 0.63 is the number to beat.

To disable COT without removing the config section, the pipeline should
check for a flag like `cftc.enabled: false`, or simply pass
`cot_raw=None` when the vol-targeting experiment runs.

---

*Research tooling only — not investment advice.*
