"""Asset and series registry — thin helpers over the Config universe."""
from __future__ import annotations

from typing import List

from src.config import Config


def ranked_tickers(cfg: Config) -> List[str]:
    """Flat ordered list of ranked assets for the cross-sectional pipeline.

    Reads from universe.ranked_assets in config.yaml.  The groups (metals,
    energy, industrial, agricultural) are flattened in config order.
    Returns an empty list when ranked_assets is not configured (Phase < 8).
    """
    ranked = cfg.universe.get("ranked_assets", {})
    if isinstance(ranked, list):
        return list(ranked)
    tickers: List[str] = []
    for group_tickers in ranked.values():
        tickers.extend(group_tickers)
    return tickers


def price_tickers(cfg: Config) -> List[str]:
    """All price tickers for data fetching (ranked assets + context), deduplicated.

    Phase 8+: returns ranked_assets + etfs + macro_context.
    Phase <8: falls back to metals + etfs + macro_context (backward compat).
    """
    tickers: List[str] = []
    seen: set = set()

    # Phase 8+: ranked assets (superset of old metals)
    for t in ranked_tickers(cfg):
        if t not in seen:
            tickers.append(t)
            seen.add(t)

    # Backward compat: if ranked_assets not configured, use legacy metals list
    if not tickers:
        for t in cfg.universe.get("metals", []):
            if t not in seen:
                tickers.append(t)
                seen.add(t)

    # Context tickers (ETFs + macro) — common to all phases
    for group in ("etfs", "macro_context"):
        for t in cfg.universe.get(group, []):
            if t not in seen:
                tickers.append(t)
                seen.add(t)

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
