"""Run the walk-forward backtest and produce a timestamped report.

Usage
-----
    python scripts/run_backtest.py [--refresh] [--ticker GC=F] [--rolling-window 63]

Outputs (all under outputs/)
-----------------------------
    reports/backtest_{ticker}_{date}_{hash8}.md
    figures/equity_{ticker}_{date}_{hash8}.png
    figures/rolling_ic_{ticker}_{date}_{hash8}.png

The {hash8} tag identifies the exact config used, so reports from different
dates with the same config are directly comparable.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.config import load_config
from src.eval.backtester import comparison_table
from src.pipeline import run_pipeline
from src.reporting import build_verdict, config_hash, write_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run walk-forward backtest and write report")
    parser.add_argument("--refresh", action="store_true", help="Force re-fetch and rebuild features")
    parser.add_argument("--ticker", default="GC=F", help="Target ticker (default: GC=F)")
    parser.add_argument("--rolling-window", type=int, default=63,
                        help="Rolling IC window in trading days (default: 63)")
    args = parser.parse_args()

    cfg = load_config()
    cfg_h = config_hash(cfg)
    logger.info("Config hash: %s", cfg_h)
    logger.info("Running pipeline for %s ...", args.ticker)

    results, oos_index = run_pipeline(
        cfg, target_ticker=args.ticker, force_refresh=args.refresh
    )

    # ── Console summary ───────────────────────────────────────────────────────
    tbl = comparison_table(results)
    print(f"\n=== Comparison Table  (config={cfg_h}) ===")
    print(tbl.to_string())

    print("\n=== Verdict ===")
    print(build_verdict(results, cfg, args.ticker))
    print()

    # ── Leakage checks ────────────────────────────────────────────────────────
    import numpy as np
    for name, r in results.items():
        if np.isfinite(r.pooled_ic) and abs(r.pooled_ic) > 0.05:
            logger.warning("LEAKAGE FLAG: %s pooled_IC=%.4f — investigate.", name, r.pooled_ic)
        if np.isfinite(r.strategy.sharpe) and r.strategy.sharpe > 2.5:
            logger.warning("LEAKAGE FLAG: %s Sharpe=%.2f — investigate.", name, r.strategy.sharpe)

    # ── Write report ──────────────────────────────────────────────────────────
    report_path = write_report(
        results=results,
        oos_index=oos_index,
        cfg=cfg,
        ticker=args.ticker,
        output_dir=cfg.paths.outputs_reports,
        figures_dir=cfg.paths.outputs_figures,
        rolling_ic_window=args.rolling_window,
    )
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
