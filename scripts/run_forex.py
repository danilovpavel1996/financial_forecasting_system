"""Run the forex cross-sectional ranking backtest.

Mirrors scripts/run_sectors.py but uses the forex universe
(EURUSD=X, GBPUSD=X, USDJPY=X, AUDUSD=X, USDCAD=X, USDCHF=X, NZDUSD=X)
configured under forex in config.yaml.  No COT, no carry proxy, no late-close lag.

Usage
-----
    python scripts/run_forex.py --horizon 63
    python scripts/run_forex.py --horizon 5
    python scripts/run_forex.py --horizon 21 --refresh
    python scripts/run_forex.py --horizon 63 --pred-avg 21
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

import numpy as np

from src.config import load_config
from src.eval.rank_backtester import ranking_comparison_table
from src.pipeline_ranking_forex import run_forex_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_CSRIC_LEAKAGE = 0.15
_SHARPE_LEAKAGE = 2.5


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-sectional forex ranking backtest"
    )
    parser.add_argument("--horizon", type=int, default=63,
                        help="Forecast horizon in trading days (default: 63)")
    parser.add_argument("--refresh", action="store_true",
                        help="Force re-fetch and rebuild data")
    parser.add_argument("--pred-avg", type=int, default=1,
                        help="Prediction averaging window (default: 1 = off; use 21 for B3 variant)")
    args = parser.parse_args()

    cfg = load_config()
    logger.info(
        "Running forex ranking pipeline — horizon=%d, pred_avg_window=%d",
        args.horizon, args.pred_avg,
    )

    results = run_forex_pipeline(
        cfg,
        horizon=args.horizon,
        force_refresh=args.refresh,
        pred_avg_window=args.pred_avg,
    )

    tbl = ranking_comparison_table(results)
    label = f"horizon={args.horizon} (forex)"
    if args.pred_avg > 1:
        label += f", pred_avg={args.pred_avg}"
    print(f"\n=== Forex Ranking Comparison Table  ({label}) ===")
    print(tbl.to_string())
    print()

    for name, r in results.items():
        if np.isfinite(r.mean_cs_ric) and abs(r.mean_cs_ric) > _CSRIC_LEAKAGE:
            logger.warning(
                "LEAKAGE FLAG: %s mean_CS_RIC=%.4f — investigate before trusting.",
                name, r.mean_cs_ric,
            )
        if np.isfinite(r.ls_sharpe) and r.ls_sharpe > _SHARPE_LEAKAGE:
            logger.warning(
                "LEAKAGE FLAG: %s Sharpe=%.2f — investigate.", name, r.ls_sharpe
            )


if __name__ == "__main__":
    main()
