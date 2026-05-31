"""CLI: pull and cache all raw price and macro data.

Usage:
    python scripts/fetch_data.py [--refresh]

    --refresh   Force re-download even if cache exists.

Second run (no --refresh) reads entirely from disk — no network required.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Make src importable when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()  # reads .env into os.environ

from src.config import load_config
from src.data import universe
from src.data.cot import fetch_all_cot
from src.data.macro import fetch_all_series
from src.data.prices import fetch_all_tickers

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("fetch_data")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch and cache all raw data.")
    p.add_argument("--refresh", action="store_true", help="Force re-download, ignore cache.")
    return p.parse_args()


def print_summary(
    prices: dict,
    macro: dict,
    cot: dict | None = None,
) -> None:
    print("\n" + "=" * 60)
    print("FETCH SUMMARY")
    print("=" * 60)

    print(f"\nPrice series ({len(prices)} tickers):")
    print(f"  {'Ticker':<15} {'Rows':>6}  {'Start':<12} {'End':<12}")
    print(f"  {'-'*15} {'-'*6}  {'-'*12} {'-'*12}")
    for ticker, df in sorted(prices.items()):
        print(
            f"  {ticker:<15} {len(df):>6}  "
            f"{str(df.index.min().date()):<12} {str(df.index.max().date()):<12}"
        )

    print(f"\nMacro series ({len(macro)} series):")
    print(f"  {'Series':<12} {'Rows':>6}  {'Non-NaN':>8}  {'Start':<12} {'End':<12}")
    print(f"  {'-'*12} {'-'*6}  {'-'*8}  {'-'*12} {'-'*12}")
    for sid, s in sorted(macro.items()):
        valid = s.dropna()
        print(
            f"  {sid:<12} {len(s):>6}  {len(valid):>8}  "
            f"{str(s.index.min().date()):<12} {str(s.index.max().date()):<12}"
        )

    if cot:
        print(f"\nCOT series ({len(cot)} tickers):")
        print(f"  {'Ticker':<8} {'Weeks':>6}  {'Start':<12} {'End':<12}")
        print(f"  {'-'*8} {'-'*6}  {'-'*12} {'-'*12}")
        for ticker, df in sorted(cot.items()):
            print(
                f"  {ticker:<8} {len(df):>6}  "
                f"{str(df.index.min().date()):<12} {str(df.index.max().date()):<12}"
            )

    print("\nAll data cached to data/raw/")
    print("=" * 60 + "\n")


def main() -> None:
    args = parse_args()
    cfg = load_config()

    start = cfg.dates["start"]
    end = cfg.dates["end"]
    cache_dir = cfg.paths.data_raw

    tickers = universe.price_tickers(cfg)
    series_ids = universe.fred_series(cfg)

    logger.info("Fetching %d price tickers: %s", len(tickers), tickers)
    prices = fetch_all_tickers(tickers, start, end, cache_dir, force_refresh=args.refresh)

    logger.info("Fetching %d FRED series: %s", len(series_ids), series_ids)
    api_key = os.environ.get("FRED_API_KEY", "").strip()
    if not api_key:
        logger.error(
            "FRED_API_KEY not found in environment. "
            "Create a .env file with FRED_API_KEY=<your_key> (see .env.example)."
        )
        sys.exit(1)

    macro = fetch_all_series(series_ids, start, end, cache_dir, api_key=api_key, force_refresh=args.refresh)

    cot_data = None
    cftc_cfg = cfg.cftc
    if cftc_cfg and cftc_cfg.get("codes"):
        logger.info("Fetching COT data for %d tickers ...", len(cftc_cfg["codes"]))
        try:
            cot_data = fetch_all_cot(
                codes=cftc_cfg["codes"],
                start=start,
                end=end,
                cache_dir=cache_dir,
                report_type=cftc_cfg.get("report_type", "disaggregated_fut"),
                force_refresh=args.refresh,
            )
        except Exception as exc:
            logger.error("COT fetch failed: %s", exc)

    print_summary(prices, macro, cot_data)


if __name__ == "__main__":
    main()
