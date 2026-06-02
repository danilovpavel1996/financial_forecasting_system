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


def sector_tickers(cfg: Config) -> List[str]:
    """Flat list of equity sector ETF tickers from the equity_sectors config section."""
    return list(cfg.equity_sectors.get("ranked_assets", []))


def sector_context_tickers(cfg: Config) -> List[str]:
    """Context tickers for the equity sector universe."""
    return list(cfg.equity_sectors.get("context", []))


def sector_price_tickers(cfg: Config) -> List[str]:
    """All price tickers needed for the equity sector pipeline, deduplicated."""
    seen: set = set()
    tickers: List[str] = []
    for t in sector_tickers(cfg) + sector_context_tickers(cfg):
        if t not in seen:
            tickers.append(t)
            seen.add(t)
    return tickers


def sector_start_date(cfg: Config) -> str:
    """Historical start date for the equity sector universe."""
    return cfg.equity_sectors.get("start_date", cfg.dates["start"])


def forex_tickers(cfg: Config) -> List[str]:
    """Flat list of forex pair tickers from the forex config section."""
    return list(cfg.forex.get("ranked_assets", []))


def forex_context_tickers(cfg: Config) -> List[str]:
    """Context tickers for the forex universe."""
    return list(cfg.forex.get("context", []))


def forex_price_tickers(cfg: Config) -> List[str]:
    """All price tickers needed for the forex pipeline, deduplicated."""
    seen: set = set()
    tickers: List[str] = []
    for t in forex_tickers(cfg) + forex_context_tickers(cfg):
        if t not in seen:
            tickers.append(t)
            seen.add(t)
    return tickers


def forex_start_date(cfg: Config) -> str:
    """Historical start date for the forex universe."""
    return cfg.forex.get("start_date", cfg.dates["start"])


def forex_cost_bps(cfg: Config) -> float:
    """Round-trip cost in bps for the forex universe (default: global cost_bps)."""
    return float(cfg.forex.get("cost_bps", cfg.cost_bps))


def sector_cost_bps(cfg: Config) -> float:
    """Round-trip cost in bps for the equity sector universe."""
    return float(cfg.equity_sectors.get("cost_bps", cfg.cost_bps))


def crypto_tickers(cfg: Config) -> List[str]:
    """Flat list of crypto tickers from the crypto config section."""
    return list(cfg.crypto.get("ranked_assets", []))


def crypto_context_tickers(cfg: Config) -> List[str]:
    """Context tickers for the crypto universe."""
    return list(cfg.crypto.get("context", []))


def crypto_price_tickers(cfg: Config) -> List[str]:
    """All price tickers needed for the crypto pipeline, deduplicated."""
    seen: set = set()
    tickers: List[str] = []
    for t in crypto_tickers(cfg) + crypto_context_tickers(cfg):
        if t not in seen:
            tickers.append(t)
            seen.add(t)
    return tickers


def crypto_start_date(cfg: Config) -> str:
    """Historical start date for the crypto universe."""
    return cfg.crypto.get("start_date", cfg.dates["start"])


def crypto_cost_bps(cfg: Config) -> float:
    """Round-trip cost in bps for the crypto universe (default: 20)."""
    return float(cfg.crypto.get("cost_bps", 20.0))


def universe_cost_bps(cfg: Config, universe_name: str) -> float:
    """Generic helper: return cost_bps for the named universe.

    universe_name one of: 'commodities', 'forex', 'equity_sectors', 'crypto'.
    Falls back to global cfg.cost_bps for unknown names.
    """
    if universe_name == "forex":
        return forex_cost_bps(cfg)
    if universe_name == "equity_sectors":
        return sector_cost_bps(cfg)
    if universe_name == "crypto":
        return crypto_cost_bps(cfg)
    return float(cfg.cost_bps)  # commodities or unknown


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
