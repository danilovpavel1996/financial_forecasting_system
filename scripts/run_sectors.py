"""Run the equity sector cross-sectional ranking backtest.

Mirrors scripts/run_ranking.py but uses the equity sector universe
(XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY) configured under
equity_sectors in config.yaml.  No COT, no carry proxy.

Usage
-----
    python scripts/run_sectors.py --horizon 63
    python scripts/run_sectors.py --horizon 5
    python scripts/run_sectors.py --horizon 21 --refresh
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
from src.pipeline_ranking_sectors import run_sectors_pipeline

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
        description="Cross-sectional equity sector ranking backtest"
    )
    parser.add_argument("--horizon", type=int, default=63,
                        help="Forecast horizon in trading days (default: 63)")
    parser.add_argument("--refresh", action="store_true",
                        help="Force re-fetch and rebuild data")
    parser.add_argument("--vol-target", type=float, default=None,
                        help="Annualized target portfolio vol (e.g. 0.10 = 10%)")
    parser.add_argument("--max-leverage", type=float, default=2.0,
                        help="Maximum position scale factor (default: 2.0)")
    args = parser.parse_args()

    cfg = load_config()
    logger.info(
        "Running equity sector ranking pipeline — horizon=%d, vol_target=%s",
        args.horizon, args.vol_target,
    )

    results = run_sectors_pipeline(
        cfg,
        horizon=args.horizon,
        force_refresh=args.refresh,
        vol_target=args.vol_target,
        max_leverage=args.max_leverage,
    )

    tbl = ranking_comparison_table(results)
    label = f"horizon={args.horizon} (equity sectors)"
    if args.vol_target:
        label += f", vol_target={args.vol_target:.0%}"
    print(f"\n=== Equity Sector Ranking Comparison Table  ({label}) ===")
    print(tbl.to_string())
    print()

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
