"""CLI: generate and print today's live commodity ranking signal.

Usage
-----
    python scripts/live_signal.py [--horizon 5] [--model mean|lgbm]
                                  [--retrain] [--no-save] [--refresh]

Options
-------
--horizon N      Forecast horizon in trading days (default: 5).
--model mean     Use MeanReversion heuristic (default, no model file needed).
--model lgbm     Use LightGBM; retrains if model is stale or missing.
--retrain        Force model retraining even if model is fresh.
--no-save        Don't write signal JSON to outputs/signals/.
--refresh        Force re-fetch of all data (bypass same-day cache).

Output
------
Formatted signal table printed to stdout.
Signal saved to outputs/signals/signal_YYYY-MM-DD.json (unless --no-save).
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("live_signal")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate today's live commodity signal")
    parser.add_argument("--horizon", type=int, default=None,
                        help="Forecast horizon in trading days")
    parser.add_argument("--model", choices=["mean", "lgbm"], default="mean",
                        help="mean = MeanReversion, lgbm = LightGBM")
    parser.add_argument("--retrain", action="store_true",
                        help="Force model retraining")
    parser.add_argument("--no-save", action="store_true",
                        help="Skip saving signal JSON")
    parser.add_argument("--refresh", action="store_true",
                        help="Force re-fetch of data (bypass cache)")
    args = parser.parse_args()

    from src.config import load_config
    from src.live.data import fetch_live_data
    from src.live.signal import format_signal, generate_signal, save_signal
    from src.live.trainer import load_latest_model, model_is_stale, train_live_model

    cfg = load_config()
    live_cfg = cfg.cftc  # re-use for live settings lookup
    live_settings = getattr(cfg, '__dict__', {}).get('live_settings', {})

    horizon = args.horizon or 5
    staleness_days = 7

    # ── Fetch live data ───────────────────────────────────────────────────────
    logger.info("Fetching live data up to today …")
    live_data = fetch_live_data(cfg, force_refresh=args.refresh, include_cot=False)
    logger.info("Live data as of %s — prices for %d tickers", live_data.as_of, len(live_data.prices))

    # ── Load or train model ───────────────────────────────────────────────────
    trained_model = None
    model_name = "MeanReversion"

    if args.model == "lgbm":
        model_name = "LightGBM"
        existing = load_latest_model(cfg)
        needs_train = args.retrain or model_is_stale(existing, staleness_days)

        if needs_train:
            logger.info("Training LightGBM on full history …")
            trained_model = train_live_model(cfg, live_data, horizon=horizon, use_cot=False)
        else:
            trained_model = existing
            logger.info(
                "Using cached model from %s (%d training rows)",
                trained_model.trained_at.strftime("%Y-%m-%d %H:%M"),
                trained_model.n_rows,
            )

    # ── Generate signal ───────────────────────────────────────────────────────
    logger.info("Generating %s signal (horizon=%d) …", model_name, horizon)
    signal = generate_signal(
        cfg=cfg,
        live_data=live_data,
        trained_model=trained_model,
        horizon=horizon,
        model_name=model_name,
        use_cot=False,
        backtest_sharpe=0.63,
        backtest_cs_ric=0.020,
        vol_target=0.10,
    )

    # ── Print formatted output ────────────────────────────────────────────────
    print()
    print(format_signal(signal))
    print()

    # ── Save JSON ─────────────────────────────────────────────────────────────
    if not args.no_save:
        signal_dir = cfg.paths.outputs_reports.parent / "signals"
        path = save_signal(signal, signal_dir)
        logger.info("Signal saved → %s", path)


if __name__ == "__main__":
    main()
