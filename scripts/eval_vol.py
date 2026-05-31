"""Standalone volatility forecast evaluation.

For each of the 9 ranked commodities, compute:
  - EWMA 1-day-ahead vol forecast (lambda=0.94)
  - GARCH(1,1) 1-day-ahead vol forecast (walk-forward fit)
  - Trailing 21-day realized vol (benchmark proxy)

Reports Pearson correlation between each forecast and realized vol.
A good EWMA achieves corr > 0.40; GARCH typically > 0.50.

Usage
-----
    python scripts/eval_vol.py [--refresh]
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
import pandas as pd

from src.config import load_config
from src.data import universe
from src.data.prices import fetch_all_tickers
from src.models.vol_forecast import EWMAVolForecast, GARCHVolForecast, realized_vol, vol_forecast_corr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("eval_vol")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate volatility forecasts")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    cfg = load_config()
    ranked = universe.ranked_tickers(cfg)

    prices = fetch_all_tickers(
        ranked,
        cfg.dates["start"],
        cfg.dates["end"],
        cfg.paths.data_raw,
        force_refresh=args.refresh,
    )

    ewma_model  = EWMAVolForecast(lam=0.94)
    garch_model = GARCHVolForecast(min_train_days=250, refit_every=63)

    rows = []
    for ticker in ranked:
        if ticker not in prices:
            logger.warning("%s: no price data, skipping.", ticker)
            continue

        close = prices[ticker]["Close"].dropna()
        log_ret = np.log(close / close.shift(1)).dropna()

        ewma_fc  = ewma_model.fit_predict(log_ret)
        rv       = realized_vol(log_ret, window=21)

        corr_ewma = vol_forecast_corr(ewma_fc, log_ret, proxy_window=21)

        # GARCH is slow — compute it too for completeness
        logger.info("Fitting GARCH(1,1) for %s ...", ticker)
        garch_fc = garch_model.fit_predict(log_ret)
        corr_garch = vol_forecast_corr(garch_fc, log_ret, proxy_window=21)

        # Summary stats for realized vol
        rv_mean = float(rv.dropna().mean())
        rv_std  = float(rv.dropna().std())

        rows.append({
            "ticker":     ticker,
            "corr_ewma":  round(corr_ewma,  3) if np.isfinite(corr_ewma)  else np.nan,
            "corr_garch": round(corr_garch, 3) if np.isfinite(corr_garch) else np.nan,
            "rv_mean":    round(rv_mean, 3),
            "rv_std":     round(rv_std,  3),
            "n_obs":      int(log_ret.notna().sum()),
        })
        logger.info(
            "%s: corr_EWMA=%.3f  corr_GARCH=%.3f  realized_vol_mean=%.1f%%",
            ticker, corr_ewma, corr_garch, rv_mean * 100,
        )

    df = pd.DataFrame(rows).set_index("ticker")

    print("\n=== Volatility Forecast Evaluation ===")
    print(df.to_string())
    print()

    ewma_ok  = (df["corr_ewma"]  > 0.40).sum()
    garch_ok = (df["corr_garch"] > 0.50).sum()
    n = len(df)
    print(f"EWMA  corr > 0.40 : {ewma_ok}/{n} tickers")
    print(f"GARCH corr > 0.50 : {garch_ok}/{n} tickers")
    print()


if __name__ == "__main__":
    main()
