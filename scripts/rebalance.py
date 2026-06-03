"""Forex portfolio rebalancer for Fusion Markets MT5.

Generates today's Forex LightGBM h=5 signal (15 pairs), compares it
to the last saved signal, and prints numbered MT5 instructions.

Usage
-----
    .venv/bin/python scripts/rebalance.py          # normal run
    .venv/bin/python scripts/rebalance.py --force  # bypass safety checks

No arguments needed. Run once per week (every 5 trading days).
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.CRITICAL,  # suppress all noise for non-technical users
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)

# ── Constants ──────────────────────────────────────────────────────────────────

LOT_SIZE   = 0.01   # Fusion Markets minimum lot
HORIZON    = 5      # trading days (Forex LightGBM h=5)
MODEL_KEY  = "live_lgbm_forex_latest"
SIGNAL_DIR = Path(__file__).resolve().parent.parent / "outputs" / "signals"

# yfinance → MT5 ticker (strip "=X")
def _to_mt5(yf_ticker: str) -> str:
    return yf_ticker.replace("=X", "")

# Human-readable pair name
_PAIR_NAMES: dict[str, str] = {
    "EURUSD": "EUR/USD", "GBPUSD": "GBP/USD", "USDJPY": "USD/JPY",
    "AUDUSD": "AUD/USD", "USDCAD": "USD/CAD", "USDCHF": "USD/CHF",
    "NZDUSD": "NZD/USD", "EURJPY": "EUR/JPY", "GBPJPY": "GBP/JPY",
    "EURGBP": "EUR/GBP", "AUDJPY": "AUD/JPY", "EURAUD": "EUR/AUD",
    "GBPAUD": "GBP/AUD", "AUDNZD": "AUD/NZD", "CADJPY": "CAD/JPY",
}


# ── Signal generation ──────────────────────────────────────────────────────────

def _generate_forex_signal():
    """Fetch live data, load/train model, return LiveSignal."""
    import numpy as np
    import joblib

    from src.config import load_config
    from src.data import universe
    from src.data.prices import fetch_all_tickers
    from src.live.data import LiveData
    from src.live.signal import generate_signal
    from src.live.trainer import TrainedModel
    from src.features.pooled_dataset import build_pooled_dataset, feature_cols
    from src.models.gbm import LightGBMModel

    cfg = load_config()

    today      = datetime.date.today()
    end_prices = (today + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    end_date   = today.strftime("%Y-%m-%d")
    start_date = universe.forex_start_date(cfg)

    live_cache = cfg.paths.data_raw.parent / "live" / end_date
    live_cache.mkdir(parents=True, exist_ok=True)

    print("  Fetching live forex prices…")
    all_tkrs = universe.forex_price_tickers(cfg)
    prices   = fetch_all_tickers(all_tkrs, start_date, end_prices, live_cache,
                                 force_refresh=False)

    live_data = LiveData(
        prices=prices, macro_raw={}, cot_raw={},
        as_of=today, start=start_date, end=end_date,
    )

    ranked_tkrs  = universe.forex_tickers(cfg)
    context_tkrs = universe.forex_context_tickers(cfg)
    ref_ticker   = ranked_tkrs[0]

    # ── Load or train model ───────────────────────────────────────────────────
    model_path   = cfg.paths.outputs_models / f"{MODEL_KEY}.pkl"
    trained_model = None
    needs_train  = not model_path.exists()

    if not needs_train:
        try:
            payload = joblib.load(model_path)
            from src.live.trainer import TrainedModel as TM
            tm = TM(
                model=payload["model"], feature_cols=payload["feature_cols"],
                trained_at=payload["trained_at"], horizon=payload["horizon"],
                n_rows=payload["n_rows"], path=model_path,
            )
            if (datetime.datetime.now() - tm.trained_at).days > 7:
                needs_train = True
            else:
                trained_model = tm
        except Exception:
            needs_train = True

    if needs_train:
        print("  Training LightGBM on full forex history (first run or stale model)…")
        pooled = build_pooled_dataset(
            cfg, live_data.prices, live_data.macro_raw,
            horizon=HORIZON, cot_raw=None,
            ranked_override=ranked_tkrs,
            context_override=context_tkrs,
            ref_ticker_override=ref_ticker,
            late_close_override=set(),
            carry_pairs_override={},
        )
        fcols = feature_cols(pooled)
        X = pooled[fcols].values.astype(float)
        y = pooled["target"].values.astype(float)
        ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1)

        model = LightGBMModel(random_state=cfg.random_seed)
        model.fit(X[ok], y[ok])
        ts = datetime.datetime.now()
        payload = {
            "model": model, "feature_cols": fcols,
            "trained_at": ts, "horizon": HORIZON, "n_rows": int(ok.sum()),
        }
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(payload, model_path)
        trained_model = TrainedModel(
            model=model, feature_cols=fcols, trained_at=ts,
            horizon=HORIZON, n_rows=int(ok.sum()), path=model_path,
        )
        print(f"  Model trained on {ok.sum()} rows and cached.")

    print("  Generating signal…")
    sig = generate_signal(
        cfg=cfg,
        live_data=live_data,
        trained_model=trained_model,
        horizon=HORIZON,
        model_name="LightGBM",
        use_cot=False,
        backtest_sharpe=1.32,
        backtest_cs_ric=0.071,
        vol_target=0.10,
        ranked_override=ranked_tkrs,
        context_override=context_tkrs,
        late_close_override=set(),
        ref_ticker_override=ref_ticker,
    )
    return sig


# ── Signal persistence ─────────────────────────────────────────────────────────

def _signal_path(date: datetime.date) -> Path:
    return SIGNAL_DIR / f"signal_forex_{date.strftime('%Y-%m-%d')}.json"


def _save_forex_signal(sig) -> Path:
    """Save forex signal as JSON; returns the path written.

    If a file for today already exists (e.g. --force re-run), it is NOT
    overwritten — the original open date is preserved for review_date math.
    """
    import numpy as np

    SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
    path = _signal_path(sig.date)
    if path.exists():
        return path   # keep original; don't shift the review date
    payload = {
        "date": sig.date.isoformat(),
        "feature_date": sig.feature_date.isoformat(),
        "horizon": sig.horizon,
        "model_name": sig.model_name,
        "model_trained_on": sig.model_trained_on.isoformat() if sig.model_trained_on else None,
        "rankings": [
            {
                "ticker": r.ticker,
                "mt5": _to_mt5(r.ticker),
                "position": r.position,
                "predicted_return": round(r.predicted_return, 6),
                "rank": r.rank,
            }
            for r in sig.rankings
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _load_previous_signal() -> dict | None:
    """Return the most recent saved forex signal JSON, or None if none exists."""
    files = sorted(SIGNAL_DIR.glob("signal_forex_*.json"), reverse=True)
    if not files:
        return None
    try:
        return json.loads(files[0].read_text(encoding="utf-8"))
    except Exception:
        return None


# ── Diff & print ───────────────────────────────────────────────────────────────

def _active_positions(signal_data: dict) -> dict[str, str]:
    """Return {mt5_ticker: position} for LONG/SHORT entries only."""
    result = {}
    for r in signal_data.get("rankings", []):
        pos = r.get("position", "FLAT")
        if pos in ("LONG", "SHORT"):
            result[r["mt5"]] = pos
    return result


def _next_rebalance(today: datetime.date) -> datetime.date:
    import pandas as pd
    dates = pd.bdate_range(pd.Timestamp(today), periods=HORIZON + 1)
    return dates[-1].date()


def _print_instructions(sig, prev_data: dict | None) -> None:
    today      = sig.date
    border     = "═" * 51
    thin       = "─" * 51

    # Build current positions: {mt5: position}
    new_positions: dict[str, str] = {}
    for r in sig.rankings:
        if r.position in ("LONG", "SHORT"):
            new_positions[_to_mt5(r.ticker)] = r.position

    # Build previous positions
    old_positions: dict[str, str] = _active_positions(prev_data) if prev_data else {}

    to_close = {
        t: old_positions[t]
        for t in old_positions
        if t not in new_positions or old_positions[t] != new_positions[t]
    }
    to_open = {
        t: new_positions[t]
        for t in new_positions
        if t not in old_positions or old_positions[t] != new_positions[t]
    }

    next_reb = _next_rebalance(today)

    print()
    print(border)
    print(f"  REBALANCE INSTRUCTIONS — {today}")
    print(f"  Signal: Forex LightGBM h=5  |  {len(new_positions)} active positions")
    print(border)

    if not to_close and not to_open:
        print()
        print("  No changes needed — keep current positions open.")
        print()
        _print_hold_summary(old_positions, new_positions)
        print(thin)
        print(f"  Next rebalance: {next_reb}")
        print(border)
        return

    # STEP 1: CLOSE
    step = 1
    if to_close:
        print()
        print(f"  STEP {step}: CLOSE these positions in MT5:")
        for mt5, pos in sorted(to_close.items()):
            side   = "BUY" if pos == "LONG" else "SELL"
            name   = _PAIR_NAMES.get(mt5, mt5)
            print(f"    ❌  Close {mt5:<8}  (currently {side} {LOT_SIZE:.2f} lot)  — {name}")
        step += 1
    else:
        print()
        print(f"  STEP {step}: Nothing to close — all existing positions stay.")
        step += 1

    # STEP 2: OPEN
    print()
    if to_open:
        print(f"  STEP {step}: OPEN these new positions in MT5:")
        for mt5, pos in sorted(to_open.items()):
            side = "BUY" if pos == "LONG" else "SELL"
            name = _PAIR_NAMES.get(mt5, mt5)
            print(f"    ✅  {side:<4} {LOT_SIZE:.2f} lot of {mt5:<8}  — {name}")
        step += 1
    else:
        print(f"  STEP {step}: No new positions to open.")
        step += 1

    # STEP 3: Verify
    total_open = len(new_positions)
    print()
    print(f"  STEP {step}: Verify in MT5 → Trade tab:")
    print(f"    You should see exactly {total_open} open position(s):")
    for mt5, pos in sorted(new_positions.items()):
        side = "BUY" if pos == "LONG" else "SELL"
        name = _PAIR_NAMES.get(mt5, mt5)
        print(f"      • {side:<4} {mt5:<8}  — {name}")

    print()
    print(thin)
    _print_hold_summary(old_positions, new_positions)
    print()
    print(f"  Next rebalance: {next_reb}")
    print()
    print("  ⚠  Research signal only — not investment advice.")
    print(border)


def _print_hold_summary(old: dict[str, str], new: dict[str, str]) -> None:
    unchanged = {t: new[t] for t in new if t in old and old[t] == new[t]}
    if unchanged:
        sides = ", ".join(
            f"{t} ({'BUY' if p == 'LONG' else 'SELL'})"
            for t, p in sorted(unchanged.items())
        )
        print(f"  Unchanged (keep open): {sides}")


# ── Safety checks ─────────────────────────────────────────────────────────────

def _day_name(d: datetime.date) -> str:
    return d.strftime("%A %B %-d")


def _safety_checks(today: datetime.date) -> None:
    """Exit early if it's not time to rebalance yet."""
    import pandas as pd

    # Check 1: already ran today
    if _signal_path(today).exists():
        next_reb = _next_rebalance(today)
        print(f"⚠️  Signal already generated today ({today}). No action needed.")
        print(f"   Next rebalance: {next_reb}  ({_day_name(next_reb)})")
        sys.exit(0)

    # Check 2: too early (previous signal's review date hasn't arrived)
    prev = _load_previous_signal()
    if prev:
        sig_date = datetime.date.fromisoformat(prev["date"])
        next_reb = _next_rebalance(sig_date)
        if today < next_reb:
            print(f"⚠️  Too early to rebalance. Current positions are still active.")
            print(f"   Current positions opened: {sig_date}")
            print(f"   Next rebalance date:      {next_reb}  ({_day_name(next_reb)})")
            print(f"   Come back on {_day_name(next_reb)}.")
            sys.exit(0)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Forex MT5 rebalancer")
    parser.add_argument("--force", action="store_true",
                        help="Bypass safety checks (for testing)")
    args = parser.parse_args()

    today = datetime.date.today()

    print()
    print("Forex Portfolio Rebalancer")
    print("══════════════════════════")
    print(f"  Date: {today}")
    print()

    if not args.force:
        _safety_checks(today)

    # 1. Load previous signal BEFORE generating new one
    prev_data = _load_previous_signal()
    if prev_data:
        print(f"  Previous signal: {prev_data['date']}")
    else:
        print("  No previous forex signal found — treating all positions as FLAT (first run).")

    print()

    # 2. Generate today's signal
    try:
        sig = _generate_forex_signal()
    except Exception as exc:
        print(f"\n  ERROR generating signal: {exc}")
        print("  Check your internet connection and try again.")
        sys.exit(1)

    # 3. Print instructions
    _print_instructions(sig, prev_data)

    # 4. Save new signal (after printing so the previous one is still the reference)
    path = _save_forex_signal(sig)
    print()
    print(f"  Signal saved → {path.relative_to(Path(__file__).resolve().parent.parent)}")
    print()


if __name__ == "__main__":
    main()
