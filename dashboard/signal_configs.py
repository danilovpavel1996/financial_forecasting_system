"""Unified signal definitions for the Live Signal page.

All signals produced across 20 phases live here.  The page is driven
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
        "cost_bps": 5,
        "description": "Quarterly commodity futures ranking — Phase 14 OOS 2005–2024",
        "model_path_key": "live_lgbm_latest",
    },
    "Forex LightGBM h=5": {
        "type": "single",
        "universe": "forex",
        "model": "LightGBM",
        "horizon": 5,
        "pred_avg_window": 1,
        "use_cot": False,
        "backtest_sharpe": 1.32,
        "backtest_cs_ric": 0.071,
        "cost_bps": 3,
        "description": "Weekly forex ranking (15 pairs, top-3/bottom-3) — Phase 20 OOS 2005–2024",
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
        "cost_bps": 3,
        "description": "Quarterly sector rotation with 21-day prediction smoothing — Phase 17",
        "model_path_key": "live_lgbm_sectors_latest",
    },
    "Crypto MeanRev h=63": {
        "type": "single",
        "universe": "crypto",
        "model": "MeanReversion",
        "horizon": 63,
        "pred_avg_window": 1,
        "use_cot": False,
        "backtest_sharpe": 0.26,
        "backtest_cs_ric": -0.075,
        "cost_bps": 20,
        "description": "Quarterly crypto ranking — Phase 20 exploratory (3 OOS folds, 2020–2024, below 0.3 threshold)",
        "model_path_key": "live_lgbm_crypto_latest",
    },
    "Cross-Asset Blend": {
        "type": "ensemble",
        "components": ["Commodity LightGBM h=63", "Forex LightGBM h=5"],
        "weights": [0.50, 0.50],
        "backtest_sharpe": 1.05,
        "backtest_cs_ric": 0.057,
        "cost_bps": 4,  # blended: 50%×5 + 50%×3
        "description": "50/50 commodity + forex (15-pair Phase 20) — Phase 18 ensemble Sharpe 1.05",
    },
}

# Human-readable names for forex pairs
FOREX_NAMES: dict[str, str] = {
    # Majors
    "EURUSD=X": "EUR/USD",
    "GBPUSD=X": "GBP/USD",
    "USDJPY=X": "USD/JPY",
    "AUDUSD=X": "AUD/USD",
    "USDCAD=X": "USD/CAD",
    "USDCHF=X": "USD/CHF",
    "NZDUSD=X": "NZD/USD",
    # Crosses (Phase 20)
    "EURJPY=X": "EUR/JPY",
    "GBPJPY=X": "GBP/JPY",
    "EURGBP=X": "EUR/GBP",
    "AUDJPY=X": "AUD/JPY",
    "EURAUD=X": "EUR/AUD",
    "GBPAUD=X": "GBP/AUD",
    "AUDNZD=X": "AUD/NZD",
    "CADJPY=X": "CAD/JPY",
}

# Human-readable names for crypto tokens (Phase 20)
CRYPTO_NAMES: dict[str, str] = {
    "BTC-USD":  "Bitcoin",
    "ETH-USD":  "Ethereum",
    "XRP-USD":  "Ripple",
    "LTC-USD":  "Litecoin",
    "ADA-USD":  "Cardano",
    "LINK-USD": "Chainlink",
    "DOGE-USD": "Dogecoin",
    "SOL-USD":  "Solana",
    "AVAX-USD": "Avalanche",
    "DOT-USD":  "Polkadot",
}
