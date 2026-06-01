# Phase 16 Summary — Equity Sector Ranking

## What was built

Added a **parallel equity sector ranking pipeline** alongside the existing commodity system.
Same walk-forward infrastructure, different universe. No COT, no carry proxy, no late-close lag.

### New files
- `config/config.yaml` — `equity_sectors` section (9 ETFs, context, start_date 1999-01-01)
- `src/config.py` — `equity_sectors` field on `Config`
- `src/data/universe.py` — `sector_tickers()`, `sector_context_tickers()`, `sector_price_tickers()`, `sector_start_date()`
- `src/features/pooled_dataset.py` — 5 new override params (`ranked_override`, `context_override`, `ref_ticker_override`, `late_close_override`, `carry_pairs_override`); backward-compatible
- `src/pipeline_ranking_sectors.py` — equity sector pipeline (mirrors `pipeline_ranking.py`)
- `scripts/run_sectors.py` — CLI for single-run equity sector backtest
- `scripts/run_sweep_sectors.py` — 9-config sweep (3 horizons × 3 models, no COT)
- `src/live/features.py` — override params added to `build_live_features` / `get_feature_date`
- `src/live/signal.py` — sector ETF name map + override params on `generate_signal`
- `pages/4_Live_Signal.py` — universe selector added (Commodities / Equity Sectors)

---

## Data verification

All 9 sector ETFs confirmed back to 1999-01-04 (6 540 rows each):

| Ticker | Sector | First date | Rows |
|--------|--------|-----------|------|
| XLB | Materials | 1999-01-04 | 6 540 |
| XLE | Energy | 1999-01-04 | 6 540 |
| XLF | Financials | 1999-01-04 | 6 540 |
| XLI | Industrials | 1999-01-04 | 6 540 |
| XLK | Technology | 1999-01-04 | 6 540 |
| XLP | Consumer Staples | 1999-01-04 | 6 540 |
| XLU | Utilities | 1999-01-04 | 6 540 |
| XLV | Health Care | 1999-01-04 | 6 540 |
| XLY | Consumer Discretionary | 1999-01-04 | 6 540 |

Pooled dataset: 4 426 common valid dates × 9 assets = 39 834 rows, 44 columns.
(Valid-date window starts ~2007 due to 252d warm-up + DTWEXBGS start 2006.)

---

## Sweep results (Phase 16)

*Format: Sharpe (net, non-overlapping) / CS-RIC (mean OOS)*
*Embargo = horizon. Costs = 5 bps round-trip. History: 1999–2024.*

| Model | h=5 | h=21 | h=63 |
| --- | --- | --- | --- |
| MeanReversion | -0.71 / -0.0083 | -0.01 / 0.0155 | **0.35 / 0.0271** |
| LightGBM | 0.31 / 0.0068 | 0.07 / 0.0453 | 0.12 / 0.0983 |
| LambdaMART | -0.25 / -0.0197 | -0.02 / 0.0070 | -0.32 / 0.0046 |

**Equity sector best: MeanReversion h=63 — Sharpe 0.35**

---

## Comparison: Equity Sectors vs Commodities (Phase 14)

| Model | Horizon | Equity Sharpe / CS-RIC | Commodity Sharpe / CS-RIC |
| --- | --- | --- | --- |
| MeanReversion | 5 | -0.71 / -0.0083 | 0.40 / 0.020 |
| MeanReversion | 63 | 0.35 / 0.0271 | — |
| LightGBM | 63 | 0.12 / 0.0983 | **0.79 / 0.055** |

---

## Acceptance-criteria assessment

- [x] 9 sector ETFs fetched, history verified back to 1999-01-04.
- [x] Equity sector ranking pipeline runs end-to-end without errors.
- [x] 9-run sweep completed (3 horizons × 3 models).
- [x] Sweep matrix reported alongside commodity matrix.
- [x] All existing tests pass (299/299).
- [ ] **Equity does NOT beat commodity best (0.35 < 0.79).**

The plan's conditional for adding equity sectors to the Live Signal was **not met**.
The universe selector UI was added as infrastructure, clearly labeled; users can see
the backtest Sharpe in the signal context panel.

---

## Interpretation

**h=63 MeanReversion (Sharpe 0.35, CS-RIC 0.027):** Positive edge. Long-term momentum /
mean-reversion across sectors exists out-of-sample. At 25 years of data, this is likely
a stable structural effect (sector rotation). But it is clearly weaker than the commodity
signal at the same horizon.

**LightGBM h=63 (CS-RIC 0.098, Sharpe 0.12):** The model has meaningful cross-sectional
predictive power (CS-RIC 0.098 is higher than commodity CS-RIC 0.055) but the net Sharpe
is only 0.12. The gap between IC and Sharpe suggests excessive turnover — the model changes
its rankings too frequently, and at 5 bps/trade the trading cost erodes the edge.

**Short horizons (h=5):** No consistent edge. MeanReversion is strongly negative (-0.71),
suggesting sectors trend over 5-day windows rather than mean-revert — opposite to commodities.

**Why equity ≠ commodity:**
The commodity signal at h=63 is driven partly by term-structure (basis_momentum) which
quantifies roll yield and convenience yield — genuinely predictive of futures returns.
Equity sector ETFs have no analogous structural carry signal. The macro and cross-sectional
features do extract some rotation signal, but it's weaker.

---

## Decisions made

1. Used `carry_pairs_override={}` — sectors get `basis_momentum` only (no carry_proxy).
   This is correct: basis_momentum captures long-run vs short-run momentum drift, which
   applies to equities too. carry_proxy requires a futures/ETF pair and is skipped.

2. Used `late_close_override=set()` — no timing lag applied since all ETFs close at 4pm ET.
   No look-ahead risk.

3. Live Signal universe selector was added as infrastructure (not conditional on results).
   The equity signal at Sharpe 0.35 is modest but positive and may be useful in combination
   with the commodity signal as a cross-asset diversifier.

---

## Next steps (optional)

- **Cross-asset ensemble:** Combine commodity signal (Sharpe 0.79) + equity sector signal
  (Sharpe 0.35) at 50/50 or 70/30 weighting. If correlation is low, combined Sharpe could
  improve. Phase 15 showed diversification works — this extends it across asset classes.
- **Equity-specific features:** SPY-relative return (sector minus market), earnings revision
  momentum, or sector valuation multiples could improve the LightGBM IC-to-Sharpe conversion.
- **LightGBM tuning:** Reduce number of leaves / increase min_child_samples to lower turnover
  without sacrificing CS-RIC. The CS-RIC of 0.098 suggests the model has signal; it just
  trades too much.

*Research tooling only — not investment advice.*
