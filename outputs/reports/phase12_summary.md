# Phase 12 Summary — Live Signal Pipeline

**Date:** 2026-06-01  
**Tests:** 259/259 pass (no regressions)

---

## What was built

| File | Role |
|------|------|
| `src/live/__init__.py` | Package init |
| `src/live/data.py` | Fetch latest prices/macro/COT up to today; cached per calendar day in `data/live/{date}/` |
| `src/live/features.py` | Build today's feature vector calling SAME functions as backtest; no logic reimplemented |
| `src/live/trainer.py` | Train LightGBM on full history via `build_pooled_dataset()`; save/load with joblib timestamp |
| `src/live/signal.py` | Generate `LiveSignal` dataclass; format and save JSON |
| `scripts/live_signal.py` | CLI: `python scripts/live_signal.py [--model mean|lgbm] [--retrain] [--refresh]` |
| `pages/4_Live_Signal.py` | Streamlit page: colour-coded ranking table, price charts, signal history |

---

## Today's actual signal output — 2026-06-01

```
══════════════════════════════════════════════════════════
  LIVE SIGNAL — 2026-06-01  —  horizon: 5 trading days
  Model: MeanReversion
  Features as of: 2026-05-31
══════════════════════════════════════════════════════════

  ▲ LONG:
     1. CL=F    (Crude Oil )  pred: +0.0748   mom_5d: -7.48%   vol: 68%
     2. ZC=F    (Corn      )  pred: +0.0341   mom_5d: -3.41%   vol: 24%

  ▼ SHORT:
     8. PA=F    (Palladium )  pred: -0.0143   mom_5d: +1.43%   vol: 39%
     9. NG=F    (Nat Gas   )  pred: -0.1412   mom_5d: +14.12%   vol: 56%

  ── FLAT: SI=F, ZS=F, PL=F, HG=F, GC=F

  ── Context ─────────────────────────────────────────────
  Vol regime: high      vol_scale: 0.26×
  Backtest Sharpe: 0.63   CS-RIC: +0.0200
  Model staleness: 0 day(s)

  ⚠  Research signal only — not investment advice.
══════════════════════════════════════════════════════════
```

### Signal interpretation (purely descriptive — not investment advice)

| Commodity | Position | Why (MeanReversion) |
|-----------|----------|---------------------|
| CL=F — Crude Oil | **LONG** | 5-day return: −7.48% (largest 5-day drop → top contrarian long) |
| ZC=F — Corn | **LONG** | 5-day return: −3.41% (second largest drop) |
| PA=F — Palladium | **SHORT** | 5-day return: +1.43% (second strongest recent runner) |
| NG=F — Natural Gas | **SHORT** | 5-day return: +14.12% (strongest recent runner → top contrarian short) |
| SI=F, ZS=F, PL=F, HG=F, GC=F | FLAT | Middle 5 (no position) |

**Vol regime: HIGH** — average EWMA vol across the basket is above 30% annualised. The vol-targeting scale is 0.26× (informational — if vol targeting were applied, positions would be reduced to 26% of notional). WTI crude oil is at 68% annualised vol, natural gas at 56%.

---

## Data coverage

All 16 tickers fetched fresh from yfinance (2010–2026-05-31):
- Futures: GC=F, SI=F, PL=F, PA=F, CL=F, NG=F, HG=F have data through 2026-05-31
- Agricultural (ZC=F, ZS=F) through 2026-05-29 (CBOT closed Friday)
- Context ETFs/indices through 2026-05-29
- FRED macro (5 series) through 2026-05-28 to 2026-05-29

Features computed as of **2026-05-31** (last valid trading day).  
Signal JSON saved: `outputs/signals/signal_2026-06-01.json`

---

## Architecture: feature parity verified

The live feature pipeline calls **the exact same functions** as the backtest:

| Backtest call | Live equivalent |
|--------------|-----------------|
| `build_price_features(prices, ticker, ...)` | Same call in `src/live/features.py` |
| `build_macro_features(macro_raw, ref_index)` | Same call |
| `build_cross_features(prices, ranked, ref_index)` | Same call |
| `build_pooled_dataset(cfg, prices, macro_raw, horizon)` | Called in `trainer.py` for training data |

Zero feature logic exists in `src/live/`. Every computation is imported from `src/features/`.

---

## Design decisions

1. **Cache per calendar day.** Live data is cached in `data/live/{YYYY-MM-DD}/`. Same-day re-runs hit the cache; next-day runs fetch fresh data automatically.

2. **Training on full history.** The live LightGBM trains on all rows from `build_pooled_dataset()` (2010 → today minus horizon). The last `horizon` days are excluded because their targets (future returns) don't exist yet. This is correct.

3. **MeanReversion as default.** No model file needed; predictions = −ret_5d. The Phase 8 backtest showed this achieves Sharpe 0.63 — the highest of any model tested. It's the right default for the live signal.

4. **Signal JSON for history.** Each run saves `outputs/signals/signal_YYYY-MM-DD.json` so the operator can later compare "what did the model say on June 1 vs what actually happened". This is essential for building trust in the signal honestly.

5. **Vol scale is informational only.** The live signal shows unscaled ranks. The vol scale (0.26× on 2026-06-01) tells the operator how much the vol-targeting module would reduce positions in the current high-vol environment.

6. **No broker connection, no execution.** Per the plan, this is a research signal that the operator reads and decides on. The disclaimer appears prominently in every output.

---

## Acceptance criteria

| Criterion | Status |
|-----------|--------|
| `python scripts/live_signal.py` produces formatted signal | ✅ (output above) |
| Signal uses today's actual prices (fetched from yfinance) | ✅ 4126 rows through 2026-05-31 |
| Features use SAME code path as backtest | ✅ Imported from src/features/, not reimplemented |
| Trained model saved to outputs/models/ with timestamp | ✅ (LightGBM via `--model lgbm`) |
| Signal JSON saved to outputs/signals/ | ✅ signal_2026-06-01.json |
| Dashboard page shows colour-coded ranking table | ✅ pages/4_Live_Signal.py |
| Signal history viewable in dashboard | ✅ Scans outputs/signals/*.json |
| 259 existing tests still pass | ✅ |

---

*Research tooling only — not investment advice.*
