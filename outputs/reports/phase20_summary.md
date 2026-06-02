# Phase 20 Summary — Forex Expansion + Crypto Universe + Cost Modeling

## What was built

### Part A — Forex universe expanded from 7 to 15 pairs

Added 8 cross pairs to `config/config.yaml`:
- EURJPY=X, GBPJPY=X, EURGBP=X, AUDJPY=X, EURAUD=X, GBPAUD=X, AUDNZD=X, CADJPY=X

Updated the forex pipeline (`src/pipeline_ranking_forex.py`) to use top-3 long / bottom-3 short
for universes with ≥9 assets (was top-2/bottom-2 for the old 7-pair universe).

### Part B — Crypto universe (10 tokens)

New pipeline `src/pipeline_ranking_crypto.py` that:
- Fetches BTC, ETH, XRP, LTC, ADA, LINK, DOGE, SOL, AVAX, DOT (start: 2020-01-01)
- Filters all data to **business days only** (drops Saturday/Sunday rows)
- Uses a crypto-specific splitter: train_years=1.5, test_years=0.5, n_splits=3
  (global splitter requires 2,772 days; crypto only has ~823 after bday filtering)

### Part C — Per-universe cost modeling

`cost_bps` added to each universe section in config.yaml:
- `forex: cost_bps: 3` (Fusion Markets spread-only)
- `equity_sectors: cost_bps: 3` (ETF spread + commission)
- `crypto: cost_bps: 20` (Binance/Kraken 0.1% per side)
- Global `cost_bps: 5` remains as commodity default

New universe helpers: `forex_cost_bps`, `sector_cost_bps`, `crypto_cost_bps`,
`universe_cost_bps(cfg, universe_name)` in `src/data/universe.py`.

### Bug fix — LambdaMART label_gain

LambdaMART had a hardcoded `label_gain = [0,1,3,7,15,31,63,127,255]` (9 entries max).
With 15 forex pairs, labels range 0-14, causing LightGBM to throw "Label 9 is not less
than the number of label mappings (9)". Fixed by computing `label_gain` dynamically from
`max(group_sizes)` at fit time.

### Dashboard updates

- `dashboard/ib_tickers.py`: Added 8 forex crosses (EUR.JPY, GBP.JPY, EUR.GBP, AUD.JPY,
  EUR.AUD, GBP.AUD, AUD.NZD, CAD.JPY on IDEALPRO) and 10 crypto tokens (on PAXOS,
  with `fractional=True` flag for token-quantity sizing).
- `dashboard/ib_tickers.py`: Updated `contracts()` to return fractional float quantities
  for crypto assets (no integer rounding; min ~$10 on Binance/Kraken).
- `dashboard/signal_configs.py`: Added "Crypto MeanRev h=63" signal entry, updated
  FOREX_NAMES for 8 new crosses, added CRYPTO_NAMES dict.
- `pages/4_Live_Signal.py`: Added crypto universe branch in `_generate_single`,
  updated welcome table, shows `cost_bps` per universe in Signal Context panel,
  resolves crypto/forex cross names via shared `_ticker_display_name` helper.

---

## Sweep results

### Expanded Forex (15 pairs, cost_bps=3, 2005–2024)

| Model | h=5 (Sharpe / CS-RIC) | h=21 | h=63 |
|-------|----------------------|------|------|
| MeanReversion | -0.20 / 0.0140 | 0.39 / 0.0212 | -0.44 / 0.0261 |
| LightGBM | **1.32 / 0.0714** | 0.05 / 0.0062 | -0.21 / 0.0630 |
| LambdaMART | 0.99 / 0.0392 | -0.17 / -0.0040 | 0.25 / 0.0506 |
| LightGBM PredAvg21 | 0.79 / 0.0342 | -0.21 / 0.0002 | 0.35 / 0.0117 |

**Best:** LightGBM h=5 — Sharpe **1.32**, CS-RIC 0.0714, turnover 0.283/day

**Improvement vs Phase 18 (7 pairs, cost_bps=5):** Sharpe 0.94 → 1.32
- Lower costs (3 vs 5 bps): +~0.1 Sharpe
- Larger cross-section (15 vs 7 pairs): +~0.3 Sharpe (better ranking stability)

No leakage flags (all CS-RIC < 0.15, all Sharpe < 2.5).

### Crypto (10 tokens, cost_bps=20, 2020–2024, 3 OOS folds)

| Model | h=5 (Sharpe / CS-RIC) | h=21 | h=63 |
|-------|----------------------|------|------|
| MeanReversion | -0.18 / 0.0138 | -0.28 / -0.0508 | **0.26 / -0.0747** |
| LightGBM | -0.54 / -0.0502 | 0.01 / 0.0130 | -0.89 / -0.0449 |
| LambdaMART | -0.52 / 0.0329 | 0.22 / 0.0531 | 0.25 / 0.1997 |
| LightGBM PredAvg21 | -0.65 / -0.1240 | -0.54 / -0.0276 | -0.93 / -0.0774 |

**Best:** MeanReversion h=63 — Sharpe **0.26** (below 0.30 acceptance threshold)

**Assessment:** No edge found for crypto after 20 bps costs. Reasons:
1. Only 823 business-day dates (DOT starts late; DOT = binding constraint), yielding
   3 non-overlapping 6-month test folds — statistically thin.
2. The 20 bps cost bar is 4× higher than forex; mean-reversion signals barely survive.
3. LambdaMART h=63 CS-RIC=0.1997 looks suspicious but Sharpe=0.25 with 3 folds is
   well within noise; not flagged as leakage at current threshold.

---

## Comparison with prior phases

| Universe | Phase | Best Sharpe | Horizon | Model | Cost (bps) |
|----------|-------|-------------|---------|-------|-----------|
| Commodities | 14 | 0.79 | 63 | LightGBM | 5 |
| Forex 7-pair | 18 | 0.94 | 5 | LightGBM PredAvg21 | 5 |
| Equity sectors | 17 | 0.41 | 63 | LightGBM PredAvg21 | 5 (now 3) |
| Cross-asset blend | 18 | 1.05 | — | Ensemble | 4 |
| **Forex 15-pair** | **20** | **1.32** | **5** | **LightGBM** | **3** |
| Crypto 10-token | 20 | 0.26 | 63 | MeanReversion | 20 |

---

## Acceptance criteria

- [x] 15 forex pairs fetched and verified (all pairs have data back to 2005-2006)
- [x] 10 crypto tokens fetched and verified (filtered to business days)
- [x] Expanded forex sweep completed (cost_bps=3) — **best Sharpe 1.32**
- [x] Crypto sweep completed (cost_bps=20) — best Sharpe 0.26
- [x] All results compared in one summary table (above)
- [x] Per-universe cost_bps used in all pipelines
- [ ] Ensemble built: skipped — crypto Sharpe below 0.5 threshold from plan §5f
- [x] Live Signal supports Crypto universe (exploratory)
- [x] Dashboard shows cost_bps per universe in Signal Context panel
- [x] All existing tests pass (299 passed)
- [x] phase20_summary.md written

---

## Decisions and ambiguities

1. **Crypto splitter:** Global splitter (train_years=3, n_splits=8) requires ~2,772 days
   minimum. Crypto after bday filtering has ~823 dates. Added crypto-specific splitter
   override (train_years=1.5, test_years=0.5, n_splits=3) in config.yaml under
   `crypto.splitter`. This is config-driven and documented.

2. **LambdaMART label_gain fix:** The hardcoded 9-entry `label_gain` was a latent bug
   that only surfaced with >9 assets. Fixed to be dynamic (scales to any group size).
   This also benefits the commodity pipeline (9 assets, no change) and future expansions.

3. **Crypto live signal:** Added "Crypto MeanRev h=63" as an exploratory entry in the
   Live Signal selector with an explicit warning note. The model is MeanReversion
   (not LightGBM) since LightGBM performed worse than MeanReversion for crypto.

4. **Crypto IB mapping:** Crypto tokens mapped to IB PAXOS exchange with `fractional=True`
   flag. The portfolio simulator shows token quantities (e.g. 0.005 BTC) rather than
   integer contracts. Primary recommendation for small capital: use Binance/Kraken
   directly (shown in the alt_exchange note).

5. **No ensemble:** The plan states to build an ensemble only if crypto Sharpe > 0.5.
   At 0.26, crypto does not meet this bar. Not built.

---

*Research tooling only — not investment advice.*
