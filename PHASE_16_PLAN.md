# PHASE_16_PLAN.md — Equity Sector ETF Ranking

> Read CLAUDE.md first. All non-negotiable principles still apply.
> This adds a PARALLEL equity sector ranking system alongside the
> existing commodity system. Same infrastructure, different universe.

---

## 1. Why equity sectors

The commodity ranking system has Sharpe ~0.79 (h=63 LightGBM) driven
by term-structure and macro features across 9 assets. Equity sector
ETFs offer three structural advantages:

1. **Larger cross-section:** 11 sectors vs 9 commodities. More assets
   = more reliable per-date ranking = higher CS-RIC stability.
2. **Sector rotation is well-documented:** different sectors lead at
   different points in the business cycle (early cycle: financials,
   industrials; late cycle: energy, staples). The yield curve slope
   and macro features should be even MORE useful here than in
   commodities because sector rotation is explicitly driven by macro.
3. **Longer history:** most SPDR sector ETFs have data since 1998 —
   25+ years including two major recessions (2001, 2008), giving
   far more independent test periods at h=63.

---

## 2. Universe

### Ranked assets: 11 SPDR sector ETFs

```yaml
equity_sectors:
  - "XLB"    # Materials
  - "XLC"    # Communication Services (launched 2018 — shorter history)
  - "XLE"    # Energy
  - "XLF"    # Financials
  - "XLI"    # Industrials
  - "XLK"    # Technology
  - "XLP"    # Consumer Staples
  - "XLRE"   # Real Estate (launched 2015 — shorter history)
  - "XLU"    # Utilities
  - "XLV"    # Health Care
  - "XLY"    # Consumer Discretionary
```

**Data coverage note:** XLC (2018) and XLRE (2015) have shorter history.
Options:
- Option A: Include them, accept shorter common period
- Option B: Exclude them, use the 9 original sectors (since 1998)
- **Recommendation: Option B for backtesting** (9 sectors, 1999–2024),
  then add XLC and XLRE for the live signal only.

The 9 long-history sectors are: XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY.

### Context tickers (shared with commodities)

`SPY`, `TLT`, `UUP`, `^VIX` — already in the system.
Add `IEF` (7-10 year Treasury ETF) if not already present — useful for
the rates/equity relationship.

### Macro features — same FRED series

`DFII10`, `T10YIE`, `DGS10`, `DGS2`, `DTWEXBGS`, `VIXCLS` — all directly
relevant to sector rotation. The yield curve slope (DGS10 - DGS2) is
especially important: financials outperform when the curve steepens,
utilities outperform when it flattens.

---

## 3. Architecture: parallel universe, shared infrastructure

### 3a. Config structure

Add an `equity_sectors` section alongside the existing `ranked_assets`:

```yaml
universes:
  commodities:
    ranked_assets:
      metals: [GC=F, SI=F, PL=F, PA=F]
      energy: [CL=F, NG=F]
      industrial: [HG=F]
      agricultural: [ZC=F, ZS=F]
    context: [GLD, SLV, GDX]
    use_cot: false
    use_carry: true

  equity_sectors:
    ranked_assets: [XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY]
    context: [SPY, TLT, UUP, "^VIX"]
    use_cot: false
    use_carry: false    # no commodity carry for equities
    start_date: "1999-01-01"   # most SPDR sectors launched late 1998
```

**Important: do NOT restructure the existing commodity config.** Add the
equity section as a NEW parallel config. The existing commodity pipeline
must continue to work unchanged.

Simpler approach if restructuring config is too disruptive: add the
equity tickers to a flat list in config.yaml under a new key, and
build a separate pipeline script.

### 3b. What to reuse (everything important)

| Component | Reusable? | Notes |
|-----------|-----------|-------|
| `src/data/prices.py` | ✅ fully | yfinance works for ETFs identically |
| `src/data/macro.py` | ✅ fully | same FRED series |
| `src/features/price_features.py` | ✅ fully | returns, vol, momentum, dist-from-MA |
| `src/features/macro_features.py` | ✅ fully | yield curve, VIX — even more relevant |
| `src/features/carry_features.py` | ⚠️ partial | basis_momentum works; carry_proxy N/A |
| `src/features/seasonal_features.py` | ✅ fully | month sin/cos |
| `src/features/cross_features.py` | ✅ fully | relative momentum, vol, dist-from-MA |
| `src/features/cot_features.py` | ❌ skip | no CFTC data for equities |
| `src/features/pooled_dataset.py` | ✅ fully | already handles N assets generically |
| `src/eval/rank_backtester.py` | ✅ fully | asset-agnostic |
| `src/models/` | ✅ fully | all models are asset-agnostic |
| `scripts/run_ranking.py` | ⚠️ adapt | needs universe selector |
| `scripts/run_sweep.py` | ⚠️ adapt | needs universe selector |

### 3c. New code needed (minimal)

1. **`scripts/run_sectors.py`** — CLI for the equity sector ranking
   pipeline. Mirrors `run_ranking.py` but uses the equity universe.

2. **Update `src/data/universe.py`** — add `sector_tickers(cfg)` helper
   that returns the equity sector list from config.

3. **Update `src/features/pooled_dataset.py`** — the existing builder
   should work if passed the right tickers and `cot_raw=None`,
   `carry_proxy` disabled. May need a flag to skip commodity-specific
   features (carry_proxy, COT). Or simply handle them gracefully
   when tickers don't match (they already return 0 for non-carry
   tickers).

4. **Update Live Signal** — add a universe selector (Commodities /
   Equity Sectors) to the dashboard.

### 3d. Late-close timing: NOT an issue

Unlike commodities (where futures close at 2:30pm and ETFs at 4pm),
ALL sector ETFs close at 4pm ET. No late-close lag needed. The
`late_close_tickers` list should be empty for the equity universe.

---

## 4. Features for equity sectors

### Features that transfer directly from commodities
- Lagged returns (1/5/10/21d)
- Rolling volatility (5/21d)
- Rolling momentum (10/21d)
- Distance from moving average (20/60d)
- Basis momentum (ret_252d - ret_21d) — captures sector momentum
  persistence, analogous to commodity term structure
- Cross-sectional relative versions of all the above
- Seasonality (month_sin, month_cos)
- Yield curve slope + changes
- VIX level + changes
- Real yield (DFII10) level + changes
- Dollar index (DXY) changes

### Features to SKIP for equities
- COT positioning (commodity-specific)
- Carry proxy (commodity futures-specific)
- Gold/silver/platinum ratios (commodity-specific)

### Potential new features (optional, not required for Phase 16)
- SPY-relative return (sector return minus market return) — the classic
  sector rotation signal. Already captured by rel_momentum but could
  be made explicit.

---

## 5. Experiment design

### 5a. Data fetch

Fetch all 9 sector ETFs + context tickers. Verify history back to
1999 for the 9 core sectors.

### 5b. Sweep (same as Phase 14)

Run the 18-configuration sweep on the EQUITY universe:
- Horizons: [5, 21, 63]
- COT: [no-COT only — no COT for equities]
- Models: [MeanReversion, LightGBM, LambdaMART]
= 9 total runs (not 18 — no COT variant)

Report the same matrix format as Phase 14.

### 5c. Compare to commodities

Side-by-side: commodity sweep matrix vs equity sector sweep matrix.
Does the equity universe produce higher Sharpe at any horizon?

### 5d. Multi-asset ensemble (bonus if time permits)

If BOTH commodities and equity sectors produce positive signals at
h=63, test a combined ensemble:
- 50% commodity signal + 50% equity sector signal
- These should be even LESS correlated than the multi-horizon
  commodity ensemble (different asset classes entirely)

---

## 6. Execution order

1. Add equity sector tickers to config.yaml
2. Update universe.py with sector_tickers() helper
3. Fetch all sector ETF data, verify history back to 1999
4. Build scripts/run_sectors.py (mirrors run_ranking.py for equities)
5. Run the 9-configuration sweep on equity sectors
6. Report sweep matrix + compare to commodity results
7. If results are positive, add equity sectors to the Live Signal page
8. Write phase16_summary.md with both sweep matrices

---

## 7. Definition of done

- [ ] 9 sector ETFs fetched, history verified (log start dates).
- [ ] Equity sector ranking pipeline runs end-to-end.
- [ ] 9-run sweep completed (3 horizons × 3 models).
- [ ] Sweep matrix reported alongside commodity matrix for comparison.
- [ ] If any equity config beats commodity best (Sharpe 0.79):
      update live signal with equity universe option.
- [ ] Live Signal page has universe selector if equity results are
      positive.
- [ ] All existing tests pass + any new tests.
- [ ] `phase16_summary.md` with honest comparison.

---

## 8. What success looks like

- **Best case:** equity sector ranking at h=63 produces Sharpe > 0.79.
  With 25+ years of data and 9 sectors, the longer history gives more
  reliable estimates. Sector rotation driven by macro features should
  be a strong cross-sectional signal.
- **Good case:** equity Sharpe is positive (0.4–0.8) and uncorrelated
  with commodity signal, enabling a cross-asset ensemble with combined
  Sharpe > 1.0.
- **Null case:** equity sectors don't rank well with these features.
  This would suggest the features are commodity-specific and equities
  need different inputs (earnings, valuation, flows).

---

*Research tooling only — not investment advice.*
