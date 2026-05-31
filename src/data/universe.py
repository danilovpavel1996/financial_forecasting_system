"""Asset and series registry — thin helpers over the Config universe."""
from __future__ import annotations

from typing import List

from src.config import Config


def price_tickers(cfg: Config) -> List[str]:
    """All price tickers (metals + ETFs + macro context) in config order."""
    tickers: List[str] = []
    for group in ("metals", "etfs", "macro_context"):
        tickers.extend(cfg.universe.get(group, []))
    return tickers


def fred_series(cfg: Config) -> List[str]:
    """FRED macro series identifiers."""
    return list(cfg.fred_series)


def metal_tickers(cfg: Config) -> List[str]:
    return list(cfg.universe.get("metals", []))


def etf_tickers(cfg: Config) -> List[str]:
    return list(cfg.universe.get("etfs", []))


def macro_context_tickers(cfg: Config) -> List[str]:
    return list(cfg.universe.get("macro_context", []))


def equity_context_tickers(cfg: Config) -> List[str]:
    """Equity-market tickers (ETFs + macro context) that close at 4pm ET.

    When the target is a futures contract (e.g. GC=F, ~2:30pm ET settlement),
    the same-day return of any equity-market ticker is partially forward-looking
    relative to the futures close.  Pass these to build_price_features as
    late_close_tickers so they receive an extra one-day lag.
    """
    tickers: List[str] = []
    for group in ("etfs", "macro_context"):
        tickers.extend(cfg.universe.get(group, []))
    return tickers
