"""Run the cross-sectional metals ranking backtest.

Usage
-----
    python scripts/run_ranking.py [--horizon 1] [--refresh]
    python scripts/run_ranking.py --horizon 5

Outputs
-------
Prints a comparison table to stdout.  Leakage flags mirror run_backtest.py.
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
from src.pipeline_ranking import run_ranking_pipeline

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
        description="Cross-sectional metals ranking backtest"
    )
    parser.add_argument("--horizon", type=int, default=1,
                        help="Forecast horizon in trading days (default: 1)")
    parser.add_argument("--refresh", action="store_true",
                        help="Force re-fetch and rebuild data")
    args = parser.parse_args()

    cfg = load_config()
    logger.info("Running ranking pipeline — horizon=%d", args.horizon)

    results = run_ranking_pipeline(
        cfg, horizon=args.horizon, force_refresh=args.refresh
    )

    tbl = ranking_comparison_table(results)
    print(f"\n=== Ranking Comparison Table  (horizon={args.horizon}) ===")
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
