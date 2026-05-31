"""Phase 0 smoke tests: config loads and validates correctly."""
import pytest
from src.config import load_config, Config, SplitterConfig


def test_config_loads():
    cfg = load_config()
    assert isinstance(cfg, Config)


def test_universe_has_tickers():
    cfg = load_config()
    tickers = cfg.all_price_tickers()
    assert len(tickers) > 0
    assert "GC=F" in tickers


def test_horizon_positive():
    cfg = load_config()
    assert cfg.horizon >= 1


def test_embargo_gte_horizon():
    """Core leakage-prevention invariant: embargo must be >= horizon."""
    cfg = load_config()
    assert cfg.splitter.embargo >= cfg.horizon, (
        f"embargo ({cfg.splitter.embargo}) must be >= horizon ({cfg.horizon})"
    )


def test_cost_bps_nonnegative():
    cfg = load_config()
    assert cfg.cost_bps >= 0


def test_fred_series_nonempty():
    cfg = load_config()
    assert len(cfg.fred_series) > 0


def test_paths_are_absolute():
    cfg = load_config()
    assert cfg.paths.data_raw.is_absolute()
    assert cfg.paths.outputs_reports.is_absolute()
