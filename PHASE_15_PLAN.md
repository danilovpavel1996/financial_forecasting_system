# PHASE_15_PLAN.md — Multi-Horizon Ensemble

> Read CLAUDE.md first. All non-negotiable principles still apply.
> This combines the h=5 MeanReversion signal (short-term tactical) with
> the h=63 LightGBM signal (long-term fundamental) into a single blended
> strategy.

---

## 1. Why a multi-horizon ensemble

The project has found two distinct signals at different timescales:

- **h=5 MeanReversion (Sharpe 0.40):** buy the most oversold commodity
  this week, short the most overbought. Fast-reacting, but noisy and
  lower Sharpe after extending to 2005.
- **h=63 LightGBM (Sharpe 0.79):** rank commodities by fundamental
  factors (basis momentum, yield curve, macro) for the next quarter.
  Slow-moving, higher Sharpe, but only rebalances every ~3 months.

These two signals are largely **uncorrelated** — one reacts to short-term
price dislocations, the other to structural factors. Combining uncorrelated
signals improves Sharpe proportionally: if two signals have Sharpe S1 and
S2 with correlation ρ ≈ 0, the combined Sharpe ≈ √(S1² + S2²).

The theoretical combined Sharpe ≈ √(0.40² + 0.79²) ≈ 0.89.

This won't be achieved exactly (transaction costs, imperfect independence),
but the principle is sound: multi-timeframe diversification is free lunch
in the same way multi-asset diversification is.

---

## 2. How the ensemble works

At each trading date `t`, the ensemble:

1. **Compute the h=5 signal:** rank all 9 commodities by negative 5-day
   trailing return (MeanReversion). Assign positions: top-2 = +1, bottom-2
   = -1, rest = 0. Normalize by gross notional (÷4).

2. **Compute the h=63 signal:** rank all 9 commodities using the trained
   LightGBM model on current features. Assign positions: top-2 = +1,
   bottom-2 = -1, rest = 0. Normalize by gross notional (÷4).

3. **Blend positions:** for each commodity:
   ```
   position_ensemble = w_short * position_h5 + w_long * position_h63
   ```
   Default weights: `w_short = 0.35`, `w_long = 0.65` (proportional to
   their Sharpe ratios: 0.40/(0.40+0.79) ≈ 0.34).

4. **Normalize:** scale the blended positions so the gross exposure
   matches the single-strategy case (sum of |positions| = 1.0).

### What happens in practice

- **Both signals agree:** commodity gets a full +1 or -1 position.
  Highest-conviction trades.
- **Signals disagree:** positions partially cancel, resulting in a
  reduced allocation. The model is less certain.
- **One signal is neutral, the other has a view:** intermediate position.

This naturally creates a conviction-weighted portfolio.

---

## 3. What to build

### 3a. Ensemble signal generator (`src/models/ensemble.py`)

```python
@dataclass
class MultiHorizonEnsemble:
    """Combines a short-horizon and long-horizon ranking signal.

    Both sub-models produce per-asset positions at each date.
    The ensemble blends them with configurable weights.
    """
    short_model: Any           # e.g., MeanReversionRanker
    long_model: Any            # e.g., LightGBMModel (trained)
    short_horizon: int = 5
    long_horizon: int = 63
    w_short: float = 0.35
    w_long: float = 0.65
```

The ensemble needs to:
- At each date, generate predictions from BOTH sub-models
- The short model uses features computed with h=5 target in mind
  (but the features themselves are the same — only the target changes)
- The long model uses the saved LightGBM trained on h=63 data
- Blend the resulting position arrays

### 3b. Ensemble backtester (`src/eval/ensemble_backtester.py`)

This is the key new piece. Unlike the ranking backtester (which runs ONE
model), the ensemble backtester must:

1. Build TWO pooled datasets: one with h=5 targets, one with h=63 targets
   (features are the same; only the forward return target differs).
2. At each walk-forward fold:
   - Train the short model on h=5 data
   - Train the long model on h=63 data
   - For each test date: generate predictions from both, blend positions,
     compute P&L
3. The P&L horizon for the ENSEMBLE should be the SHORT horizon (h=5),
   because positions are updated at the short-horizon frequency. The
   long model's predictions change slowly (same ranking for weeks), but
   the portfolio is rebalanced at the short cadence.

**Alternative simpler approach:** don't retrain in the backtester.
Instead, run the h=5 and h=63 backtests SEPARATELY (we already have
these results), then blend their DAILY P&L series:

```python
pnl_ensemble = w_short * pnl_h5 + w_long * pnl_h63
```

This is mathematically equivalent if the two strategies don't interact
(positions are independent). It's much simpler to implement and avoids
the complexity of running two models simultaneously.

**Use the simple approach first.** It requires no new backtester — just
a post-processing step on existing results.

### 3c. Ensemble analysis script (`scripts/run_ensemble.py`)

```bash
python scripts/run_ensemble.py
```

This script:
1. Runs the h=5 MeanReversion backtest (no COT)
2. Runs the h=63 LightGBM backtest (no COT)
3. Aligns their daily P&L series to the common date range
4. Computes the blended P&L at multiple weight combinations
5. Reports: Sharpe, ann_ret, max_dd, turnover for each blend
6. Also reports the correlation between the two P&L series
   (if < 0.3, diversification benefit is real)

Weight combinations to test:
- 100/0 (pure h=5)
- 75/25
- 50/50
- 35/65 (Sharpe-proportional — theoretical optimum)
- 25/75
- 0/100 (pure h=63)

### 3d. Live signal integration

Update `src/live/signal.py` and `pages/4_Live_Signal.py` to support an
ensemble mode:
- Show BOTH the h=5 and h=63 rankings side by side
- Show the blended position for each commodity
- Color code: green = both agree long, red = both agree short,
  yellow = signals disagree (lower conviction)

### 3e. Dashboard page update

Add an "Ensemble" option to the Model selector in the Live Signal page.
When selected, it runs both models and shows the blended view.

---

## 4. Execution order

1. Build `scripts/run_ensemble.py` — the P&L blending analysis
2. Run it and check the correlation between h=5 and h=63 P&L series
3. Report the Sharpe for each weight combination
4. If the ensemble beats both individual strategies: update live signal
5. Add ensemble view to the Live Signal dashboard page
6. Write `phase15_summary.md` with the weight sweep table

---

## 5. Definition of done

- [ ] Correlation between h=5 and h=63 daily P&L reported.
- [ ] Weight sweep table: Sharpe / ann_ret / max_dd for 6 weight
      combinations.
- [ ] If ensemble Sharpe > 0.79 (Phase 14 best): update live signal.
- [ ] Live Signal page shows ensemble view with both rankings +
      blended positions.
- [ ] All existing tests pass.
- [ ] `phase15_summary.md` with the full analysis.

---

## 6. What success looks like

- **Target:** ensemble Sharpe ≈ 0.85–0.90 (diversification benefit
  from combining uncorrelated signals).
- **Minimum:** ensemble Sharpe > 0.79 (beats the best individual model).
- **Bonus:** max drawdown improves because the two signals hedge each
  other during different market regimes.
- **Null:** correlation between the two P&L series is > 0.5, meaning
  they're too similar for diversification to help much.

---

## 7. Key constraint: P&L series alignment

The h=5 and h=63 backtests produce OOS P&L series over different date
ranges (different embargo values affect the first test fold). The
ensemble analysis must use only the INTERSECTION of dates where both
strategies have OOS predictions. Log how many dates are dropped.

---

## 8. Evaluation: which horizon for Sharpe calculation?

The blended P&L is computed daily. For Sharpe annualization, use
periods_per_year = 252 (daily) with NO subsampling — the blended
series is already a mix of horizons and doesn't have the clean
overlapping-return structure that requires subsampling.

Actually, this needs care. The h=63 component contributes 63-day
returns daily (overlapping). The blended series inherits this
autocorrelation. For honest Sharpe calculation, use Newey-West
standard errors with lag = 63, or subsample the blended series
every 5 days (the short horizon) as a compromise.

The script should report BOTH the raw daily Sharpe and the
subsampled-every-5-days Sharpe, so the operator can see the
sensitivity.

---

*Research tooling only — not investment advice.*
