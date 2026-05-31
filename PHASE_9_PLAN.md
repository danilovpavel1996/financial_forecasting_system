# PHASE_9_PLAN.md — CFTC Commitment of Traders Data

> Read CLAUDE.md first. All non-negotiable principles still apply.
> This adds weekly CFTC positioning data as features to the cross-sectional
> ranking system built in Phases 7–8.

---

## 1. Why CFTC data

Phases 7–8 found weak mean-reversion in cross-sectional commodity ranking.
The current features are entirely price-derived (returns, momentum, vol,
ratios). They capture *what prices did* but not *who is positioned how*.

CFTC Commitment of Traders (COT) data shows the aggregate positions of:
- **Managed Money** (hedge funds, CTAs) — the speculative side
- **Producer/Merchant** (commercials/hedgers) — the fundamental side
- **Swap Dealers** — intermediaries

The classic signal: when Managed Money is extremely long a commodity
(crowded trade), it tends to mean-revert as positions unwind. This is
the *structural explanation* for the mean-reversion effect Phase 7 found,
but measured directly from positioning data instead of inferred from prices.

This is the single most informative non-price feature for commodity futures.

---

## 2. Data source

### Report type: Disaggregated Futures-Only

The **Disaggregated** report splits traders into 4 categories (vs only 2
in the Legacy report): Producer/Merchant, Swap Dealers, Managed Money,
Other Reportables. This is the right report for commodities.

### Timing and publication lag (CRITICAL for leakage)

- Positions are reported **as of each Tuesday**.
- The report is **released on Friday at 3:30pm ET**.
- Therefore: on any given trading day, the most recent AVAILABLE report
  is from the **previous Friday**, covering positions as of the **Tuesday
  before that Friday**.

**Lag rule:** in the feature pipeline, the COT feature at date `t` must
use the most recent report whose RELEASE DATE (Friday) is strictly < `t`.
In practice this means:
- Monday through Thursday: use the report released the PREVIOUS Friday
- Friday: use the report released the PREVIOUS Friday (NOT today's —
  it comes out at 3:30pm, after most trading)

Implementation: store each report row with its **release_date** (the
Friday), then lag by 1 trading day from release_date, same as FRED
macro features. This ensures no look-ahead.

### Python library: `cot_reports`

```bash
pip install cot_reports
```

```python
import cot_reports as cot

# Fetch Disaggregated Futures-Only for a specific year
df = cot.cot_year(year=2024, cot_report_type="disaggregated_fut")
```

The library downloads the official CFTC zip files. No API key needed.
Cache the downloaded data to `data/raw/cot/` as parquet, same pattern
as prices and FRED.

### CFTC contract codes for our 9 commodities

Filter the downloaded data by `CFTC_Contract_Market_Code`:

```yaml
cftc_codes:
  "GC=F": "088691"    # Gold - COMEX
  "SI=F": "084691"    # Silver - COMEX
  "PL=F": "076651"    # Platinum - NYMEX
  "PA=F": "075651"    # Palladium - NYMEX
  "CL=F": "067651"    # Crude Oil WTI - NYMEX
  "NG=F": "023651"    # Natural Gas - NYMEX
  "HG=F": "085692"    # Copper - COMEX
  "ZC=F": "002602"    # Corn - CBOT
  "ZS=F": "005602"    # Soybeans - CBOT
```

NOTE: these codes may need verification against the actual downloaded data.
The `Market_and_Exchange_Names` column is the fallback for matching. If a
code doesn't match, search by commodity name substring (e.g. "GOLD",
"SILVER", "CRUDE OIL"). Log the exact match used for each commodity.

---

## 3. Features to extract (per commodity, per week)

### Raw fields from the Disaggregated report

For each commodity, extract from the report:
- `MM_Long` = Managed Money long positions (column: `M_Money_Positions_Long_All`)
- `MM_Short` = Managed Money short positions (column: `M_Money_Positions_Short_All`)
- `Prod_Long` = Producer/Merchant long (column: `Prod_Merc_Positions_Long_All`)
- `Prod_Short` = Producer/Merchant short (column: `Prod_Merc_Positions_Short_All`)
- `OI` = Open Interest (column: `Open_Interest_All`)

NOTE: column names vary across CFTC report formats. The loader must discover
the correct column names from the downloaded data and log them. Do NOT
hardcode column names without verifying against the actual data. Common
alternatives include `M_Money_Positions_Long_ALL` (capitalisation varies)
and sometimes the column names use underscores vs spaces.

### Derived features (computed in feature engineering, not the loader)

All features below are computed from the raw fields above and are PER
COMMODITY (not cross-sectional). Cross-sectional relative versions
(e.g., "this commodity's MM positioning vs basket average") are built
in cross_features.py, same as relative momentum.

1. **`mm_net`** = MM_Long − MM_Short
   Net managed-money positioning. Positive = speculators are net long.

2. **`mm_net_chg`** = week-over-week change in mm_net
   Are speculators adding or cutting?

3. **`mm_net_pct`** = mm_net as percentile of its trailing 52-week range
   `(mm_net − min_52w) / (max_52w − min_52w)`
   Values near 1.0 = historically extreme long (crowded); near 0.0 =
   extreme short. THIS IS THE PRIMARY CONTRARIAN SIGNAL.

4. **`mm_long_ratio`** = MM_Long / (MM_Long + MM_Short)
   Directional conviction of managed money. 0.5 = neutral.

5. **`prod_net`** = Prod_Long − Prod_Short
   Commercial hedger positioning. Commercials are typically contrarian
   to speculators — when producers are very short, it often signals
   fundamental overvaluation.

6. **`oi_chg_pct`** = week-over-week % change in open interest
   Rising OI + rising price = new longs entering (trend confirmation).
   Rising OI + falling price = new shorts entering (trend exhaustion).

### Alignment to daily trading calendar

COT data is weekly. To use it in the daily feature pipeline:
1. Store the release_date (Friday) on each observation.
2. Forward-fill to daily trading days, same as FRED. Use
   `align_to_trading_index()` from macro_features.py.
3. Apply lag of 1 trading day from release_date (same as FRED).
4. The `mm_net_pct` 52-week percentile uses a trailing 52-week window
   of weekly observations (NOT daily — compute on the weekly series
   BEFORE forward-filling to daily).

---

## 4. What to build

### 4a. COT data loader (`src/data/cot.py`)

- Fetch Disaggregated Futures-Only reports for the configured date range
  using `cot_reports` library.
- Filter to the 9 configured commodities by CFTC code (with name-based
  fallback).
- Extract the raw fields listed in §3.
- Store release_date (the Friday of publication).
- Cache to `data/raw/cot/` as parquet, keyed by year range.
- Handle missing weeks, holidays, and years with no data gracefully.
- Add to `scripts/fetch_data.py` so one command fetches everything.

### 4b. COT feature engineering (`src/features/cot_features.py`)

- Compute the 6 derived features from §3 for each commodity.
- The 52-week percentile (`mm_net_pct`) is computed on the WEEKLY series
  before forward-filling.
- Align to daily trading calendar via `align_to_trading_index()`.
- Lag by 1 trading day from release_date (same as FRED).
- All features must pass the shift-and-compare leakage test.

### 4c. Integration into pooled dataset

Update `src/features/pooled_dataset.py` to include COT features alongside
price and macro features. The COT features are PER-ASSET (each commodity
gets its own positioning data), so they naturally slot into the per-asset
feature block.

Also add CROSS-SECTIONAL COT features to `cross_features.py`:
- `rel_mm_net_pct` = asset's mm_net_pct minus basket average
  "Is this commodity more/less crowded than the average?"
- `rel_mm_net_chg` = asset's mm_net_chg minus basket average
  "Are speculators adding to this commodity more/less than peers?"

### 4d. Tests

- Leakage test: shift-and-compare on all COT features (mandatory).
- Verify that the COT feature at date `t` uses only report data with
  release_date < t (a direct date-comparison test).
- Verify mm_net_pct is in [0, 1] after the warm-up period.
- Verify the forward-fill doesn't introduce gaps > 10 trading days
  (would indicate missing weekly data).

### 4e. Run ranking at horizon=5

Re-run `scripts/run_ranking.py --horizon 5` with the expanded feature
set and compare to Phase 8 results. The key question: does MeanReversion
or LightGBM CS-RIC and Sharpe improve with COT features?

---

## 5. Config changes

Add to `config/config.yaml`:

```yaml
cftc:
  report_type: "disaggregated_fut"
  codes:
    "GC=F": "088691"
    "SI=F": "084691"
    "PL=F": "076651"
    "PA=F": "075651"
    "CL=F": "067651"
    "NG=F": "023651"
    "HG=F": "085692"
    "ZC=F": "002602"
    "ZS=F": "005602"
```

Add `cot_reports` to `requirements.txt` and `pyproject.toml`.

---

## 6. Definition of done

- [ ] COT data fetched and cached for all 9 commodities (2010–2024).
- [ ] 6 per-asset COT features + 2 cross-sectional COT features built.
- [ ] Shift-and-compare leakage test passes on all COT features.
- [ ] Direct date-comparison test: feature at `t` uses only release_date < t.
- [ ] mm_net_pct in [0, 1] after warm-up.
- [ ] Pooled dataset includes COT features; fold-integrity test still passes.
- [ ] Ranking comparison table at horizon=5 with and without COT features.
- [ ] `phase9_summary.md` states whether COT features improved CS-RIC
      and/or Sharpe vs Phase 8 baseline.
- [ ] All tests pass.

---

## 7. What to watch for

- **Column name variations.** CFTC report formats have changed over the
  years. The loader must handle this gracefully — discover column names
  from the data, not assume them. Log the columns found.
- **Missing commodities in early years.** Some commodities may not appear
  in the Disaggregated report before 2010 (the format started in 2009).
  Handle gracefully; log coverage per commodity.
- **The mm_net_pct 52-week lookback.** This consumes the first year of
  data as warm-up. With data from 2010, features won't be valid until
  2011. That's fine — the price features already consume ~60 days.
- **Forward-fill limit.** Set max_ffill_days=10 for weekly COT data
  (max gap = 2 missed weekly reports). Log any gaps beyond this.

---

*Research tooling only — not investment advice.*
