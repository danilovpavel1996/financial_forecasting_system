"""Unified signal definitions for the Live Signal page.

All 4 signals produced across 18 phases live here.  The page is driven
entirely by this config — no hardcoded signal logic elsewhere.
"""
from __future__ import annotations

from typing import Any

SIGNALS: dict[str, dict[str, Any]] = {
    "Commodity LightGBM h=63": {
        "type": "single",
        "universe": "commodities",
        "model": "LightGBM",
        "horizon": 63,
        "pred_avg_window": 1,
        "use_cot": False,
        "backtest_sharpe": 0.79,
        "backtest_cs_ric": 0.055,
        "description": "Quarterly commodity futures ranking — Phase 14 OOS 2005–2024",
        "model_path_key": "live_lgbm_latest",
    },
    "Forex LightGBM h=5": {
        "type": "single",
        "universe": "forex",
        "model": "LightGBM",
        "horizon": 5,
        "pred_avg_window": 21,
        "use_cot": False,
        "backtest_sharpe": 0.94,
        "backtest_cs_ric": 0.060,
        "description": "Weekly forex pair ranking with 21-day prediction smoothing — Phase 18",
        "model_path_key": "live_lgbm_forex_latest",
    },
    "Equity B3 LightGBM h=63": {
        "type": "single",
        "universe": "equity_sectors",
        "model": "LightGBM",
        "horizon": 63,
        "pred_avg_window": 21,
        "use_cot": False,
        "backtest_sharpe": 0.41,
        "backtest_cs_ric": 0.100,
        "description": "Quarterly sector rotation with 21-day prediction smoothing — Phase 17",
        "model_path_key": "live_lgbm_sectors_latest",
    },
    "Cross-Asset Blend": {
        "type": "ensemble",
        "components": ["Commodity LightGBM h=63", "Forex LightGBM h=5"],
        "weights": [0.50, 0.50],
        "backtest_sharpe": 1.05,
        "backtest_cs_ric": 0.057,
        "description": "50/50 commodity + forex — Phase 18 OOS Sharpe 1.05, ρ=−0.04",
    },
}

# Human-readable names for forex pairs (supplement to signal.py _NAMES)
FOREX_NAMES: dict[str, str] = {
    "EURUSD=X": "EUR/USD",
    "GBPUSD=X": "GBP/USD",
    "USDJPY=X": "USD/JPY",
    "AUDUSD=X": "AUD/USD",
    "USDCAD=X": "USD/CAD",
    "USDCHF=X": "USD/CHF",
    "NZDUSD=X": "NZD/USD",
}
