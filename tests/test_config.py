"""Phase 0 smoke tests: config loads and validates correctly."""
import pytest
from src.config import load_config, Config, SplitterConfig
from src.data.universe import ranked_tickers


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


def test_ranked_tickers_returns_configured_list():
    """ranked_tickers() must return exactly the 9 Phase-8 tickers in config order."""
    cfg = load_config()
    tickers = ranked_tickers(cfg)
    expected = ["GC=F", "SI=F", "PL=F", "PA=F", "CL=F", "NG=F", "HG=F", "ZC=F", "ZS=F"]
    assert tickers == expected, f"ranked_tickers mismatch:\n  got:      {tickers}\n  expected: {expected}"


def test_ranked_tickers_in_price_tickers():
    """All ranked tickers must be included in price_tickers() for data fetching."""
    from src.data.universe import price_tickers
    cfg = load_config()
    ranked = ranked_tickers(cfg)
    fetched = price_tickers(cfg)
    missing = [t for t in ranked if t not in fetched]
    assert not missing, f"ranked tickers not in price_tickers: {missing}"
