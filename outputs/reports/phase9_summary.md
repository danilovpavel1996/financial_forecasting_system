# Phase 9 Summary — CFTC Commitment of Traders Data

**Date:** 2026-05-31  
**Assets:** GC=F, SI=F, PL=F, PA=F, CL=F, NG=F, HG=F, ZC=F, ZS=F (9-asset basket)  
**Period:** 2010-01-01 – 2024-12-31  **Folds:** 8 walk-forward  **Cost:** 5 bps round-trip  

---

## What was built

| Component | Description |
|-----------|-------------|
| `config/config.yaml` | Added `cftc` section with `report_type` and 9 commodity codes |
| `src/config.py` | Added `cftc: Dict` field (default `{}`, backward-compatible) |
| `src/data/cot.py` | COT loader: fetches Disaggregated Futures-Only, slims to 8 columns, caches as parquet |
| `src/features/cot_features.py` | 6 per-asset COT features + 2 cross-sectional features, aligned + lagged |
| `src/features/pooled_dataset.py` | Accepts optional `cot_raw` dict; joins 8 COT columns per asset when present |
| `src/pipeline_ranking.py` | Fetches COT data if `cfg.cftc` is configured, passes to pooled dataset builder |
| `scripts/fetch_data.py` | Fetches and summarises COT data alongside prices and macro |
| `requirements.txt` / `pyproject.toml` | Added `cot_reports>=0.4` |
| `tests/test_cot_features.py` | 15 new tests (239 total, all pass) |

---

## COT data coverage

All 9 commodities matched by CFTC code (primary path, no name-based fallback needed):

| Ticker | Code | Exchange match | Weeks | NaN rows |
|--------|------|---------------|-------|---------|
| GC=F | 088691 | GOLD - COMMODITY EXCHANGE INC. | 782 | 0 |
| SI=F | 084691 | SILVER - COMMODITY EXCHANGE INC. | 782 | 0 |
| PL=F | 076651 | PLATINUM - NEW YORK MERCANTILE EXCHANGE | 782 | 0 |
| PA=F | 075651 | PALLADIUM - NEW YORK MERCANTILE EXCHANGE | 782 | 0 |
| CL=F | 067651 | CRUDE OIL, LIGHT SWEET - NEW YORK MERCANTILE EXCHANGE | 782 | 0 |
| NG=F | 023651 | NATURAL GAS - NEW YORK MERCANTILE EXCHANGE | 782 | 0 |
| HG=F | 085692 | COPPER-GRADE #1 - COMMODITY EXCHANGE INC. | 782 | 0 |
| ZC=F | 002602 | CORN - CHICAGO BOARD OF TRADE | 782 | 0 |
| ZS=F | 005602 | SOYBEANS - CHICAGO BOARD OF TRADE | 782 | 0 |

Coverage: 782/780 expected weeks (100%), 2010-01-08 → 2024-12-27.

**Column name variation discovered:** 
- 2010–2012 uses `Report_Date_as_MM_DD_YYYY` (cot_reports parses to ISO string)
- 2013+ uses `Report_Date_as_YYYY-MM-DD`

All 5 numeric columns (`M_Money_Positions_Long_All`, `M_Money_Positions_Short_All`, `Prod_Merc_Positions_Long_All`, `Prod_Merc_Positions_Short_All`, `Open_Interest_All`) are stable across all years 2010–2024.

**Caching fix:** The raw combined DataFrame has 146,977 rows × 191 mixed-type columns. Parquet serialization of the full raw fails on string columns with trailing spaces. Fixed by narrowing to 8 essential columns (`_slim_columns`) before caching.

---

## COT feature engineering

6 per-asset features (computed on weekly series, then aligned + lagged 1 trading day):

| Feature | Description |
|---------|-------------|
| `mm_net` | Managed Money long − short (net speculative positioning) |
| `mm_net_chg` | Week-over-week change in mm_net |
| `mm_net_pct` | 52-week rolling min-max percentile of mm_net (primary contrarian signal) |
| `mm_long_ratio` | MM_Long / (MM_Long + MM_Short) |
| `prod_net` | Producer/Merchant long − short |
| `oi_chg_pct` | Week-over-week % change in open interest |

2 cross-sectional features (asset vs basket average):

| Feature | Description |
|---------|-------------|
| `rel_mm_net_pct` | asset mm_net_pct − basket average |
| `rel_mm_net_chg` | asset mm_net_chg − basket average |

**Warm-up:** mm_net_pct requires 26+ weekly observations (min_periods), so COT features are NaN for the first ~6 months. All 9 assets show 126 NaN rows (≈ half-year warm-up + the existing 59-day price warm-up).

**Pooled dataset** grew from 42 to 50 feature columns (8 COT per asset). Common dates dropped slightly from 3668 (Phase 8) to 3613 (Phase 9) due to the COT warm-up period.

---

## Leakage verification

**Leakage rule:** Report date (Tuesday) + 3 calendar days = release_date (Friday). Feature at day `t` uses the most recent release_date strictly < `t`. Implemented via `align_to_trading_index(max_ffill_days=10)` + `lag_series(lag=1)`.

| Test | Result |
|------|--------|
| Shift-and-compare (drop 1/5/10 weeks) | PASS |
| Direct date-comparison: feature at t uses release_date < t | PASS |
| Friday uses previous Friday's release, not same-day | PASS |
| mm_net_pct ∈ [0, 1] after warm-up | PASS |
| No gap > 10 trading days after first valid value | PASS |
| Cross-sectional COT features sum to zero across basket | PASS |

---

## Results: horizon = 5 (Phase 8 vs Phase 9 comparison)

### Phase 8 (without COT)

```
               n_folds  mean_CS_RIC  std_CS_RIC  CS_RIC_stab  Sharpe_net  ann_ret  max_dd  spr_capture  turnover
EqualWeight          8          NaN         NaN          NaN         NaN    0.000   0.000          NaN     0.000
MomentumRank         8      -0.0220      0.3981        0.476       -0.29   -0.059  -0.572       -0.007     0.167
MeanReversion        8       0.0197      0.3877        0.501        0.63    0.114  -0.302        0.009     0.326
ElasticNet           8          NaN         NaN          NaN         NaN    0.000   0.000          NaN     0.000
LightGBM             8       0.0109      0.4329        0.491        0.43    0.060  -0.276        0.007     0.110
```

### Phase 9 (with COT, 50 features)

```
               n_folds  mean_CS_RIC  std_CS_RIC  CS_RIC_stab  Sharpe_net  ann_ret  max_dd  spr_capture  turnover
EqualWeight          8          NaN         NaN          NaN         NaN    0.000   0.000          NaN     0.000
MomentumRank         8      -0.0298      0.3977        0.467       -0.56   -0.108  -0.655       -0.011     0.169
MeanReversion        8       0.0264      0.3887        0.506        0.29    0.051  -0.350        0.012     0.328
ElasticNet           8          NaN         NaN          NaN         NaN    0.000   0.000          NaN     0.000
LightGBM             8      -0.0016      0.4075        0.483       -0.33   -0.043  -0.475       -0.006     0.129
```

### Change table (Phase 8 → Phase 9, horizon=5)

| Model | CS_RIC Δ | CS_RIC_stab Δ | Sharpe Δ |
|-------|----------|--------------|---------|
| MomentumRank  | −0.0220 → −0.0298 (worse) | 0.476 → 0.467 (worse) | −0.29 → −0.56 (worse) |
| MeanReversion | +0.0197 → +0.0264 (better) | 0.501 → 0.506 (better) | +0.63 → +0.29 (worse) |
| LightGBM      | +0.0109 → −0.0016 (worse) | 0.491 → 0.483 (worse) | +0.43 → −0.33 (worse) |

---

## Honest verdict: COT features did NOT improve performance

**COT features hurt more than they helped at horizon=5.**

1. **MeanReversion:** CS-RIC improved slightly (+0.0264 vs +0.0197) but Sharpe dropped sharply (+0.29 vs +0.63). The `ret_5d` signal itself is unchanged — MeanReversion uses only that one feature. The drop in Sharpe comes from the reduced dataset (3613 vs 3668 common dates), which slightly shifts fold boundaries and produces a different out-of-sample P&L stream.

2. **LightGBM:** Both CS-RIC and Sharpe degraded significantly. CS-RIC went from +0.0109 to −0.0016 (no signal); Sharpe from +0.43 to −0.33. Adding 8 COT features per observation (72 extra columns across 9 assets) to a dataset of 3613 dates × 9 assets with 8 walk-forward folds gives LightGBM more dimensions to overfit. The COT features appear to introduce noise rather than signal in this backtest.

3. **MomentumRank:** Worsened on all metrics. Same analysis as LightGBM — the reduced dataset and the structural change hurt the position-count baseline.

4. **Structural explanation for expected vs actual:** The plan hypothesised that extreme Managed Money positioning (mm_net_pct near 1.0) would predict mean-reversion. While this is a well-documented stylised fact in commodity markets, the 5-day horizon may be too short to capture the positioning reversal — COT signals typically play out over 3–8 weeks, not 5 days. The COT features may have more predictive power at horizon=21 (monthly), which is outside the current study.

**No leakage flags were triggered** (no CS-RIC > 0.15, no Sharpe > 2.5 after including COT features).

---

## Acceptance criteria

| Criterion | Status |
|-----------|--------|
| COT data fetched for all 9 commodities (2010–2024) | ✅ 782 weeks each, 0 NaN, 100% coverage |
| 6 per-asset COT features + 2 cross-sectional features built | ✅ 8 COT columns per asset in pooled dataset |
| Shift-and-compare leakage test passes | ✅ (3 window sizes) |
| Direct date-comparison: feature at t uses only release_date < t | ✅ |
| Friday does not use same-day release | ✅ |
| mm_net_pct ∈ [0, 1] after warm-up | ✅ |
| Forward-fill gaps ≤ 10 trading days | ✅ |
| Cross-sectional COT sum to zero | ✅ |
| Pooled dataset includes COT features; fold-integrity test still passes | ✅ 239/239 tests pass |
| Ranking comparison table at horizon=5 with and without COT | ✅ above |
| `phase9_summary.md` states whether COT improved CS-RIC and/or Sharpe | ✅ **No — COT features hurt LightGBM and did not improve Sharpe** |

---

## Key decisions and notes

1. **Parquet caching of slim columns only.** The raw combined COT DataFrame has 191 columns with mixed types (strings with spaces in some numeric columns in older years). Saving only the 8 essential columns avoids pyarrow serialisation errors and speeds up cache reads dramatically.

2. **52-week percentile on weekly series.** The `mm_net_pct` is computed on the weekly indexed series BEFORE forward-filling to daily. This is correct — computing on the daily forward-filled series would inflate the rolling window count without adding new information.

3. **release_date = report_date + 3 calendar days.** The CFTC Tuesday report date + 3 days → Friday. This is a fixed offset (no holiday adjustment). The lag=1 trading day then ensures the feature is not available on the Friday of release.

4. **Backward compatibility maintained.** The `cot_raw=None` default in `build_pooled_dataset()` means all Phase 7/8 tests run unchanged. The `cftc: dict` field in Config defaults to `{}`.

5. **Horizon mismatch hypothesis.** COT signals typically take weeks to play out. At horizon=5 (1 trading week), the positioning information may not be predictive. The Phase 10 recommendation would be to test at horizon=21 or horizon=63.

---

## Column names discovered from actual data (logged per year)

All years (2010–2024): column names for the 5 numeric fields are STABLE:
- `M_Money_Positions_Long_All`
- `M_Money_Positions_Short_All`
- `Prod_Merc_Positions_Long_All`
- `Prod_Merc_Positions_Short_All`
- `Open_Interest_All`

Date column variation:
- 2010–2012: `Report_Date_as_MM_DD_YYYY` (parsed to ISO by cot_reports)
- 2013–2024: `Report_Date_as_YYYY-MM-DD`

---

*Research tooling only — not investment advice.*
