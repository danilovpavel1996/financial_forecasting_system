# PHASE_18_PLAN.md — Equity Leg Upgrade + Forex Universe

> Read CLAUDE.md first. All non-negotiable principles still apply.
> Two parts: a quick fix (upgrade equity leg) then a new universe (forex).

---

## Part A: Upgrade Equity Leg in Cross-Asset Ensemble (5 minutes)

### Problem

Phase 17's cross-asset ensemble uses MeanReversion (Sharpe 0.35) for
the equity leg. But B3 PredAvg21 LightGBM (Sharpe 0.48) is now
available. Swapping it in should improve the combined Sharpe.

### What to do

Update `scripts/run_cross_asset_ensemble.py` to use B3 PredAvg21
LightGBM for the equity sector leg instead of MeanReversion:
- Run equity sector backtest with LightGBM h=63 + 21-day prediction
  averaging
- Blend with commodity LightGBM h=63 at the same weight sweep

Report the new combined Sharpe. If 50/50 exceeds 0.97, that's the
new best.

### Implementation

The B3 prediction averaging logic needs to be accessible as a model
variant, not just a standalone script. The simplest approach: add a
`pred_avg_window` parameter to `LightGBMModel` (or create a wrapper)
that applies rolling mean to predictions before ranking. Then pass
this model variant to the sector pipeline.

Alternatively, apply the averaging in the backtester itself — after
`model.predict()`, smooth the predictions over 21 days before ranking.
This is cleaner because it doesn't change the model class.

---

## Part B: Forex Universe

### Why forex

Three asset classes with low correlation = more diversification:
- Commodities: driven by supply/demand, carry, industrial cycles
- Equity sectors: driven by earnings, macro cycle, rate expectations
- Forex: driven by interest rate differentials, capital flows, PPP

If the forex signal is uncorrelated with both commodities and equity
sectors, a three-way ensemble could push Sharpe above 1.0.

### Universe: 7 major USD pairs

```yaml
forex:
  ranked_assets:
    - "EURUSD=X"    # Euro
    - "GBPUSD=X"    # British Pound
    - "USDJPY=X"    # Japanese Yen
    - "AUDUSD=X"    # Australian Dollar
    - "USDCAD=X"    # Canadian Dollar
    - "USDCHF=X"    # Swiss Franc
    - "NZDUSD=X"    # New Zealand Dollar
  context:
    - "SPY"
    - "TLT"
    - "UUP"
    - "^VIX"
  start_date: "2005-01-01"
```

yfinance provides daily forex data with `=X` suffix. EURUSD history
goes back to 2003.

**Quote convention note:** some pairs are XXX/USD (EURUSD, GBPUSD,
AUDUSD, NZDUSD — foreign currency priced in USD) and others are
USD/XXX (USDJPY, USDCAD, USDCHF — USD priced in foreign). For
cross-sectional ranking this is fine — the model learns each pair's
behavior through one-hot encoding. Don't invert any prices.

### Features

Reuse ALL existing features — they transfer well to forex:
- **Price features:** momentum, vol, dist-from-MA — directly applicable
- **Basis momentum:** captures trending behavior in FX (documented)
- **Yield curve slope:** directly drives USD strength / FX carry trades
- **Seasonality:** month effects exist in FX (quarter-end flows, etc.)
- **Cross-sectional relative:** which currency is trending/reverting
  more than peers — exactly the ranking question

**Skip:** COT (not configured for forex), carry_proxy (no ETF pairs)

### No late-close timing issue

Forex trades 24/5. yfinance records a "close" at 5pm ET. All pairs
use the same timestamp. No late-close lag needed.

### Pipeline

Mirror the equity sector approach:
- Add `forex` section to config.yaml
- Add `forex_tickers()`, `forex_context_tickers()` to universe.py
- Create `src/pipeline_ranking_forex.py` (mirrors sectors pipeline)
- Create `scripts/run_forex.py` and `scripts/run_sweep_forex.py`
- Add Forex to the Live Signal universe selector

### Sweep

9 configurations: 3 horizons [5, 21, 63] × 3 models [MeanReversion,
LightGBM, LambdaMART]. Apply B3 prediction averaging (21-day) to
LightGBM as well, since it worked for sectors.

### Three-way ensemble

After the forex sweep, build the final ensemble:
- Commodity LightGBM h=63 (Sharpe 0.79)
- Equity sector B3 LightGBM h=63 (Sharpe 0.48)
- Forex best model at best horizon

Blend with equal weights (33/33/33) and report:
- Pairwise correlations (3 pairs)
- Combined Sharpe

---

## Execution order

1. Part A: upgrade equity leg → re-run cross-asset ensemble
2. Add forex config + universe helpers
3. Fetch forex data, verify history
4. Build forex pipeline + sweep script
5. Run 9-config forex sweep (+ B3 variant for LightGBM)
6. Build three-way ensemble script
7. Report all results + update live signal if improved
8. Write phase18_summary.md

---

## Definition of done

- [ ] Equity leg upgraded to B3 LightGBM; improved ensemble Sharpe reported
- [ ] 7 forex pairs fetched, history verified
- [ ] Forex sweep completed (9+ configs)
- [ ] Three-way pairwise correlations reported
- [ ] Three-way ensemble Sharpe reported
- [ ] If three-way > two-way: update live signal with 3-universe selector
- [ ] All existing tests pass
- [ ] `phase18_summary.md` with all results

---

*Research tooling only — not investment advice.*
