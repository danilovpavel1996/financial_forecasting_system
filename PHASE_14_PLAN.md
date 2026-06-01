# PHASE_14_PLAN.md — New Features + Extended History

> Read CLAUDE.md first. All non-negotiable principles still apply.
> This adds three new feature categories and extends the data history
> to increase both signal quality and statistical reliability.

---

## 1. Problem statement

Phase 13 showed the current feature set (price returns, macro, COT) has
hit its ceiling. MeanReversion h=5 (Sharpe 0.63) remains unbeaten.
LambdaMART showed promising CS-RIC at h=63 (0.083) but only ~56
independent test periods make the Sharpe estimate unreliable.

Two interventions:
A. **New features** that carry genuinely different information
B. **More data** to increase statistical power at longer horizons

---

## 2. New feature category 1: Carry / term-structure proxy

### Why carry matters

In commodity futures, **carry** (the roll yield from contango/backwardation)
is one of the most robust documented return predictors. Commodities in
backwardation (front < spot) tend to outperform those in contango (front >
spot), because holding futures in backwardation earns a positive roll yield.

For cross-sectional ranking, this is powerful: if gold is in backwardation
and silver is in contango, the carry signal says long gold / short silver —
independent of momentum or mean-reversion.

### The data challenge

Ideally we'd compare front-month vs second-month futures prices. But
yfinance doesn't provide clean second-month continuous contracts. So we
use two practical proxies:

### Proxy A: ETF / futures spread (metals only)

For metals with liquid ETFs tracking spot price:
- `GLD` tracks gold spot → `carry_proxy = log(GC=F / GLD)`
- `SLV` tracks silver spot → `carry_proxy = log(SI=F / SLV)`

Positive = contango (futures above spot = negative carry for holders).
Negative = backwardation (futures below spot = positive carry).

This is a clean proxy because the ETF price ≈ spot and the futures price
reflects the term structure premium.

**Limitation:** only works for GC=F and SI=F (we have GLD and SLV).
Other commodities (PL=F, PA=F, CL=F, NG=F, HG=F, ZC=F, ZS=F) don't
have clean spot-tracking ETFs in our data.

### Proxy B: Implied roll yield from the continuous contract (all commodities)

The continuous front-month contract (`GC=F`, `CL=F`, etc.) from yfinance
rolls from one expiry to the next periodically. On roll dates, there's a
price gap between the expiring and new front contract.

**Approximation:** compute the **trailing 21-day return minus the
log-return of the raw close-to-close.** Actually, simpler: since
yfinance's `auto_adjust=True` adjusts for splits but NOT for futures
rolls, the daily return of the continuous contract includes both the
actual price move AND the roll gap. We can't cleanly separate them.

**Better approximation for all commodities:** use the **basis momentum**
signal instead of pure carry. Basis momentum = trailing 12-month return
of the futures contract minus the trailing 1-month return. In academic
literature (Koijen, Moskowitz, Pedersen, Vrugt 2018), this captures
term-structure information because the long-term return embeds cumulative
roll yield while the short-term return is dominated by spot moves.

```python
basis_momentum = ret_252d - ret_21d
```

This is computable from the existing price data (no new data source needed)
and captures term-structure information indirectly. It's been shown to be
a significant cross-sectional predictor of commodity futures returns.

### Features to build

For each commodity:
1. **`carry_proxy`** (GC=F and SI=F only): `log(futures / ETF)`
   - Also compute `carry_proxy_chg_21d`: 21-day change in carry proxy
2. **`basis_momentum`** (all 9): `ret_252d - ret_21d`
   - This is the primary term-structure feature for all commodities
3. **Cross-sectional relative versions:**
   - `rel_basis_momentum` = asset's basis_momentum minus basket average
   - `rel_carry_proxy` (where available) minus average

---

## 3. New feature category 2: Seasonality

### Why seasonality matters

Commodity returns have well-documented seasonal patterns:
- Natural gas: higher prices in winter (heating demand)
- Corn/soybeans: price pressure around planting (April–May) and
  harvest (September–October)
- Gold: stronger in Q1 and Q4 (jewelry demand, central bank buying)

These are calendar-based — no new data source needed.

### Features to build

1. **`month_sin`** = sin(2π × month / 12)
2. **`month_cos`** = cos(2π × month / 12)

Using sine/cosine encoding (not one-hot) because:
- Only 2 features instead of 11 dummy variables (less overfitting)
- Captures the cyclical nature (December is close to January)
- Works with tree models (LightGBM) and linear models

These features are the SAME for all assets on the same date, so they
don't help with cross-sectional ranking directly. But they help the
model learn "in this season, mean-reversion is stronger/weaker" —
acting as an interaction term.

---

## 4. New feature category 3: Yield curve slope

### Why yield curve matters

The yield curve slope (long rates minus short rates) signals:
- Monetary policy expectations (steep = easing expected, flat = tightening)
- Risk appetite (steep = risk-on, inverted = recession risk)
- Dollar direction (which moves gold inversely)

We already have `DGS10` (10-year Treasury yield) from FRED. Adding
`DGS2` (2-year Treasury yield) gives us the slope.

### Features to build

1. **`yield_curve_slope`** = DGS10 − DGS2
   - Positive = normal curve; negative = inverted
2. **`yield_curve_slope_chg_21d`** = 21-day change in slope
   - Steepening vs flattening signal
3. **`yield_curve_slope_zscore`** = z-score of slope over trailing
   252-day window (computed on train data only — same as all features)

Add `DGS2` to the FRED series list in config.yaml.

---

## 5. Extended history: 2005–2024

### Why more data helps

At h=63, Phase 13 had ~56 non-overlapping test periods. The Sharpe
standard error was 1/√56 ≈ 0.13, making 0.27 indistinguishable from zero.
Extending to 2005 adds ~5 more years = ~1,250 more trading days = ~20 more
quarterly periods, reducing the standard error to ~0.11.

More importantly, 2005–2009 includes the financial crisis — a critical
stress test for any commodity strategy.

### Config change

```yaml
dates:
  start: "2005-01-01"    # was 2010-01-01
  end: "2024-12-31"
```

### Data availability check

Claude Code must verify that all 9 commodities have data back to 2005:
- GC=F, SI=F, CL=F, NG=F, HG=F: likely yes (major CME/COMEX contracts)
- PL=F, PA=F: possibly shorter history
- ZC=F, ZS=F: likely yes (CBOT)
- DGS2 from FRED: yes (available since 1976)

If any commodity doesn't have data back to 2005, log it and use whatever
is available. The pooled dataset intersection will handle it.

COT (Disaggregated) reports started in 2006. So COT features will have
warm-up starting from ~2007 (2006 + 52-week percentile lookback).

---

## 6. What to build

### 6a. New FRED series (`config/config.yaml`)

Add `DGS2` to the FRED series list. Update `scripts/fetch_data.py` to
fetch the extended date range.

### 6b. Term structure features (`src/features/carry_features.py`)

New file:
- `basis_momentum(close, ret_252d, ret_21d)` for all 9 commodities
- `carry_proxy(futures_close, etf_close)` for GC=F and SI=F only
- Cross-sectional relative versions

### 6c. Seasonality features (`src/features/seasonal_features.py`)

New file:
- `month_sin`, `month_cos` from the date index
- Trivial to implement, no data source needed

### 6d. Yield curve features (extend `src/features/macro_features.py`)

Add `yield_curve_slope`, `yield_curve_slope_chg_21d` to the existing
macro features module. Uses DGS10 and DGS2 (both already in FRED).

### 6e. Integration into pooled dataset

Update `pooled_dataset.py` to include all new features.

### 6f. Extended date range

Update `config/config.yaml` to start from 2005. Re-fetch all data.

### 6g. Sweep with new features

Re-run the Phase 13 sweep (3 horizons × 2 COT configs × 3 models = 18
runs) with the new features and extended history. Compare to Phase 13.

---

## 7. Execution order

1. Add DGS2 to config, extend dates to 2005, re-fetch all data
2. Build carry_features.py (basis momentum + carry proxy)
3. Build seasonal_features.py (month sin/cos)
4. Extend macro_features.py (yield curve slope)
5. Integrate all new features into pooled_dataset.py
6. Shift-and-compare leakage tests for all new features
7. Run the 18-configuration sweep
8. Compare to Phase 13 sweep matrix
9. If any config beats Sharpe 0.63, update the live signal
10. Write phase14_summary.md with both sweep matrices side by side

---

## 8. Definition of done

- [ ] Data extended to 2005 (or earliest available per commodity).
- [ ] DGS2 fetched and yield curve slope features built.
- [ ] Basis momentum computed for all 9 commodities.
- [ ] Carry proxy computed for GC=F and SI=F.
- [ ] Seasonality features (month sin/cos) built.
- [ ] All new features pass shift-and-compare leakage tests.
- [ ] 18-run sweep completed with new features + extended history.
- [ ] Phase 13 vs Phase 14 comparison matrix in the summary.
- [ ] If improvement found, live signal updated.
- [ ] All existing tests pass + new feature tests.
- [ ] `phase14_summary.md` with both matrices and honest verdict.

---

## 9. What success looks like

- **Best case:** new features + more data push LambdaMART or LightGBM
  past Sharpe 0.63 at h=5 or h=21. Update live signal.
- **Partial win:** CS-RIC improves at h=63 with more reliable Sharpe
  estimate (more data). Or a new feature (basis momentum) shows
  significant cross-sectional power even if overall Sharpe is similar.
- **Null result:** features and data don't help. This would suggest the
  signal ceiling is structural (commodity markets are too efficient for
  free-data signals) and motivate either paid data or a different
  asset class.

---

*Research tooling only — not investment advice.*
