# PHASE_8_PLAN.md — Expand Universe to Broad Commodity Basket

> Read CLAUDE.md first. All non-negotiable principles still apply.
> This extends the Phase 7 cross-sectional ranking system.

---

## 1. Why expand

Phase 7's cross-sectional ranking showed weak mean-reversion across 4 metals
(CS-RIC = 0.061, stability = 0.514 at 5-day horizon). Two structural problems
limited it:

1. **n=4 is too small for reliable ranking.** Spearman correlation over 4 items
   has almost no statistical power. With n=10, the same signal becomes far more
   detectable per-date, improving CS-RIC stability.
2. **4 metals are driven by similar macro factors** (real yields, USD, risk
   appetite). Cross-sectional dispersion is low. Adding commodities with
   genuinely different fundamentals (energy, industrial, agricultural) creates
   more relative mispricings for the model to exploit.

---

## 2. New universe

### Ranked assets (the cross-section — predict relative returns)

Keep the 4 metals plus add energy, industrial metals, and agricultural:

```yaml
ranked_assets:
  metals:
    - "GC=F"    # Gold
    - "SI=F"    # Silver
    - "PL=F"    # Platinum
    - "PA=F"    # Palladium
  energy:
    - "CL=F"    # WTI Crude Oil
    - "NG=F"    # Natural Gas
  industrial:
    - "HG=F"    # Copper
  agricultural:
    - "ZC=F"    # Corn
    - "ZS=F"    # Soybeans
```

That gives 9 ranked assets. If any ticker has insufficient data (< 2000 rows
back to 2010), drop it and note why in the summary. Natural gas (NG=F) is
known to be volatile and may have data quirks — handle gracefully.

### Context-only tickers (features, not ranked — unchanged)

`GLD`, `SLV`, `GDX`, `SPY`, `TLT`, `UUP`, `^VIX` — same as before, with
the late-close lag already applied.

### FRED macro series — unchanged

`DFII10`, `T10YIE`, `DGS10`, `DTWEXBGS`, `VIXCLS`.

---

## 3. What to change

### 3a. `config/config.yaml`

Replace the universe section. The ranked assets should be in a NEW key
`ranked_assets` (a flat list or grouped dict), separate from the existing
`metals` / `etfs` / `macro_context` which serve as context-only tickers.
Keep backward compatibility: the existing single-asset pipeline
(`run_backtest.py`) should still work with `--ticker GC=F`.

### 3b. `src/data/universe.py`

Add a `ranked_tickers(cfg)` helper that returns the flat list of ranked
assets from the new config key. The existing `price_tickers(cfg)` should
return the UNION of ranked + context tickers (for data fetching), and a new
helper should return ranked-only (for the ranking pipeline).

### 3c. `scripts/fetch_data.py`

Must fetch ALL tickers (ranked + context). No other changes needed — the
price loader is already ticker-agnostic.

### 3d. `src/features/cross_features.py`

MUST be generalised to N assets, not hardcoded for 4 metals. The basket
averages (for relative momentum, relative vol, etc.) should be computed
over all N ranked assets. Cross-ratios (gold/silver etc.) can remain as
they are — they're specific feature pairs, not basket-wide.

### 3e. `src/features/pooled_dataset.py`

Must handle N assets. Verify it already does — if there are any hardcoded
references to 4 metals or the metals list, generalise them.

### 3f. `src/eval/rank_backtester.py`

Must handle N assets in the long-short construction. With 9 assets:
- Long top-2, short bottom-2 (instead of top-1/bottom-1 with 4 assets).
  Or make it configurable: `n_long` and `n_short` as parameters.
- With more assets, the spread between top and bottom is naturally wider,
  so the strategy has more room for signal to express itself.

### 3g. Tests

- Existing cross-feature leakage tests must pass with the expanded basket.
- Pooled dataset fold-integrity test must pass with 9 assets (all 9 at the
  same date in the same fold).
- Add a test that verifies `ranked_tickers(cfg)` returns exactly the
  configured list.

---

## 4. What to run

After the code changes, run:

```bash
# Fetch the new tickers
python scripts/fetch_data.py

# Run ranking at both horizons
python scripts/run_ranking.py --horizon 1
python scripts/run_ranking.py --horizon 5
```

---

## 5. Definition of done

- [ ] All new tickers fetched and cached (log any with < 2000 rows).
- [ ] Cross-features work with 9 assets (relative features sum to zero
      across the full basket at every date).
- [ ] Pooled dataset has 9 assets per date, fold-integrity test passes.
- [ ] Ranking backtester uses configurable n_long / n_short (default: 2/2
      for 9 assets).
- [ ] Comparison tables at horizon=1 and horizon=5 with all baselines +
      models.
- [ ] `phase8_summary.md` compares the 9-asset results to the Phase 7
      4-asset results and states whether CS-RIC stability improved.
- [ ] All tests pass (existing + new).

---

## 6. What to watch for

- **Data coverage mismatches.** Agricultural futures may have different
  trading calendars (CBOT vs COMEX). The pooled dataset builder must handle
  dates where not all 9 assets have data — either drop those dates or
  forward-fill within a strict limit. Log the decision.
- **Palladium and natural gas liquidity.** PA=F and NG=F are thinner
  markets. If their data has excessive gaps, note it but don't drop them
  silently.
- **The n_long/n_short choice matters.** With 9 assets, top-2/bottom-2
  is reasonable. Top-1/bottom-1 concentrates too much risk in a single
  commodity. Report both if practical.

---

*Research tooling only — not investment advice.*
