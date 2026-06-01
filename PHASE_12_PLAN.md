# PHASE_12_PLAN.md — Live Signal Pipeline

> Read CLAUDE.md first. All non-negotiable principles still apply.
> This adds a forward-looking signal generator that produces TODAY's
> commodity ranking and position recommendation based on real-time data.

---

## 1. What this does

The backtest system (Phases 1–10) answers: "would this strategy have worked
historically?" The live signal pipeline answers: "what does the model say
to do RIGHT NOW?"

Concretely, running the live signal produces:
- Today's feature values for all 9 commodities
- Model predictions (forward return estimates) for the next `horizon` days
- A ranking: which commodities to go long, which to short
- Confidence context: how the current signal compares to historical norms
- A position sheet: "Long ZC=F + SI=F, Short NG=F + PA=F" (example)

**This is still research tooling — not a trading bot.** It does not
execute trades, connect to a broker, or manage real money. It produces
a signal that the operator reads and decides what to do with.

---

## 2. Architecture

### 2a. Live data fetcher (`src/live/data.py`)

Fetch the LATEST prices up to TODAY for all tickers:
- Use yfinance with `end=today+1` (yfinance end is exclusive)
- For FRED macro: fetch up to today
- For COT: fetch the most recent year (COT data updates weekly)
- Cache with a short TTL (1 day) so re-runs within the same day
  don't re-fetch, but tomorrow's run gets fresh data

The key difference from the backtest fetcher: the backtest uses a FIXED
date range (2010–2024). The live fetcher extends to TODAY.

### 2b. Live feature builder (`src/live/features.py`)

Compute today's features for all 9 commodities using the SAME feature
functions from src/features/. This is critical — the live features must
use exactly the same code path as the backtest features. No separate
implementation.

The output is a single-row feature vector per commodity (the most
recent valid date), ready for model prediction.

### 2c. Model trainer (`src/live/trainer.py`)

Train the ranking models on ALL available historical data (not
walk-forward — for live prediction we use the full history as training).

Models to train:
- MeanReversion (no training needed — it's a heuristic)
- LightGBM (train on full pooled historical dataset)

Save the trained model to `outputs/models/` with a timestamp so the
operator can see when it was last retrained.

**Retraining schedule:** the model should be retrained periodically
(e.g., weekly or monthly). The live signal script checks if the saved
model is stale (> N days old) and retrains automatically if so. Default
staleness threshold: 7 days.

### 2d. Signal generator (`src/live/signal.py`)

The core function: given today's features and a trained model, produce:

```python
@dataclass
class LiveSignal:
    date: datetime.date              # signal date (today)
    horizon: int                     # forecast horizon in trading days
    model_name: str                  # which model produced this
    model_trained_on: datetime.date  # when the model was last trained
    rankings: list[AssetRanking]     # ordered best → worst
    positions: dict[str, float]      # ticker → position size (+1, -1, 0)
    confidence: SignalConfidence     # context metrics

@dataclass
class AssetRanking:
    ticker: str
    predicted_return: float          # model's point estimate
    rank: int                        # 1 = best, N = worst
    position: str                    # "LONG", "SHORT", or "FLAT"
    recent_momentum_5d: float        # trailing 5d return (context)
    current_vol: float               # EWMA vol (context)

@dataclass
class SignalConfidence:
    backtest_sharpe: float           # from the most recent backtest
    backtest_cs_ric: float           # average CS-RIC from backtest
    days_since_retrain: int          # staleness of the model
    current_vol_regime: str          # "low" / "normal" / "high"
    vol_scale: float                 # vol-targeting position scale
```

### 2e. CLI script (`scripts/live_signal.py`)

```bash
python scripts/live_signal.py [--horizon 5] [--retrain]
```

Prints a formatted signal report to stdout:

```
══════════════════════════════════════════════════════
  LIVE SIGNAL — 2026-06-01 — horizon: 5 trading days
  Model: MeanReversion (trained: 2026-05-31)
══════════════════════════════════════════════════════

  LONG:
    1. ZC=F  (Corn)       pred: +1.2%   mom_5d: -2.1%  vol: 18%
    2. SI=F  (Silver)     pred: +0.8%   mom_5d: -1.5%  vol: 22%

  SHORT:
    3. NG=F  (Nat Gas)    pred: -0.3%   mom_5d: +3.2%  vol: 45%
    4. PA=F  (Palladium)  pred: -0.9%   mom_5d: +1.8%  vol: 28%

  FLAT:
    5–9. GC=F, PL=F, CL=F, HG=F, ZS=F

  ── Context ──────────────────────────────────────────
  Vol regime: normal (scale: 1.05x)
  Backtest Sharpe: 0.63 | CS-RIC: 0.020
  Model staleness: 1 day

  ⚠ Research signal only — not investment advice.
══════════════════════════════════════════════════════
```

Also saves this to `outputs/signals/signal_YYYY-MM-DD.json` for history.

### 2f. Dashboard integration (`pages/4_Live_Signal.py`)

A new Streamlit page that:
- Shows today's signal (the same output as the CLI, but visual)
- A table of the 9 commodities ranked best → worst with color coding
  (green for LONG, red for SHORT, gray for FLAT)
- Trailing price charts for the LONG and SHORT picks (last 60 days)
- Vol regime indicator
- Signal history: table of past signals from outputs/signals/
- A "Refresh Signal" button that re-fetches data and regenerates
- A "Retrain Model" button that forces retraining

---

## 3. What NOT to build

- **No broker connection.** No order execution, no position management.
- **No real-time streaming.** Signal updates on button click, not live.
- **No portfolio tracking.** No P&L tracking of actual trades.
- **No alerts/notifications.** The operator checks the dashboard.

---

## 4. File structure

```
src/live/
├── __init__.py
├── data.py           # fetch latest prices/macro/COT up to today
├── features.py       # compute today's feature vector (reuses src/features/)
├── trainer.py        # train on full history, save to outputs/models/
└── signal.py         # generate LiveSignal from features + model

scripts/
└── live_signal.py    # CLI: print today's signal

pages/
└── 4_Live_Signal.py  # Streamlit page

outputs/
├── models/           # saved trained models (timestamped)
└── signals/          # signal history (JSON, one per day)
```

---

## 5. Execution order

1. `src/live/data.py` — live data fetcher with today's date
2. `src/live/features.py` — compute today's features
3. `src/live/trainer.py` — train on full history, save model
4. `src/live/signal.py` — generate signal dataclass
5. `scripts/live_signal.py` — CLI output + JSON save
6. `pages/4_Live_Signal.py` — dashboard page
7. End-to-end test: run the CLI and verify it produces a valid signal

---

## 6. Critical constraints

### 6a. Feature parity

The live features MUST use the exact same functions as the backtest:
- `src/features/price_features.py` → `build_price_features()`
- `src/features/macro_features.py` → `build_macro_features()`
- `src/features/cross_features.py` → `build_cross_features()`
- `src/features/cot_features.py` → `build_cot_features()`

Do NOT reimplement any feature logic in src/live/. Import and call the
existing functions. If a feature function needs to be adapted for live
use (e.g., it requires a full date range), create a thin wrapper that
calls the original.

### 6b. No future data in live mode

The live feature at today must use ONLY data available today:
- Prices: up to yesterday's close (today's close isn't known yet if
  markets are open, but yfinance returns the last available close)
- FRED: lagged 1 trading day (same as backtest)
- COT: most recent Friday release, lagged 1 day (same as backtest)

### 6c. Model training uses full history

Unlike the walk-forward backtest (which never sees the test set during
training), the live model trains on ALL available data — including what
was "test" in the backtest. This is correct: for a live forecast, you
want to use all information available up to today.

The backtest Sharpe (0.63) is still the out-of-sample estimate of the
strategy's quality. The live model just has more training data.

### 6d. Signal history for tracking

Each signal is saved as JSON in outputs/signals/ so the operator can
later check: "what did the model say on June 1? Did it turn out right?"
This is essential for building trust (or losing it honestly).

---

## 7. Config additions

Add to `config/config.yaml`:

```yaml
live:
  retrain_staleness_days: 7      # retrain model if older than this
  default_horizon: 5             # default forecast horizon
  signal_output_dir: "outputs/signals"
```

---

## 8. Definition of done

- [ ] `python scripts/live_signal.py` produces a formatted signal
      for today with all 9 commodities ranked.
- [ ] The signal uses today's actual prices (fetched from yfinance).
- [ ] Features use the SAME code path as the backtest (imported, not
      reimplemented).
- [ ] A trained model is saved to outputs/models/ with timestamp.
- [ ] Signal JSON saved to outputs/signals/.
- [ ] Dashboard page shows today's signal with color-coded table.
- [ ] Signal history viewable in the dashboard.
- [ ] All existing 259 tests still pass.
- [ ] `phase12_summary.md` with today's actual signal output and STOP.

---

*Research tooling only — not investment advice.*
