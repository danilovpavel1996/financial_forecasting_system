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
    parser.add_argument("--no-cot", action="store_true",
                        help="Disable COT features (revert to Phase 8 baseline)")
    parser.add_argument("--vol-target", type=float, default=None,
                        help="Annualized target portfolio vol (e.g. 0.10 = 10%)")
    parser.add_argument("--max-leverage", type=float, default=2.0,
                        help="Maximum position scale factor (default: 2.0)")
    args = parser.parse_args()

    cfg = load_config()
    logger.info(
        "Running ranking pipeline — horizon=%d, cot=%s, vol_target=%s",
        args.horizon, not args.no_cot, args.vol_target,
    )

    results = run_ranking_pipeline(
        cfg,
        horizon=args.horizon,
        force_refresh=args.refresh,
        use_cot=not args.no_cot,
        vol_target=args.vol_target,
        max_leverage=args.max_leverage,
    )

    tbl = ranking_comparison_table(results)
    label = f"horizon={args.horizon}"
    if args.vol_target:
        label += f", vol_target={args.vol_target:.0%}"
    if args.no_cot:
        label += ", no_cot"
    print(f"\n=== Ranking Comparison Table  ({label}) ===")
    print(tbl.to_string())
    print()

    # Print realized vol for vol-targeted runs
    if args.vol_target:
        print("Realized ann_vol per model (target: {:.0%}):".format(args.vol_target))
        for name, r in results.items():
            if np.isfinite(r.ls_ann_vol):
                print(f"  {name}: {r.ls_ann_vol:.1%}")
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
