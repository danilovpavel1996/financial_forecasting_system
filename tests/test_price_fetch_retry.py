"""fetch_all_tickers must survive a transient data-provider hiccup.

On 2026-10-16 yfinance returned an empty frame for a single pair; the weekly
rebalance aborted and no trading happened that week, while the same request
succeeded minutes later. These tests pin the retry behaviour that prevents a
repeat, and the loud failure that must still happen when data is really gone.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import prices  # noqa: E402


def _frame() -> pd.DataFrame:
    return pd.DataFrame({"Close": [1.0]}, index=pd.to_datetime(["2026-10-16"]))


def test_transient_failure_is_retried_then_succeeds(monkeypatch, tmp_path):
    calls = {"n": 0}

    def flaky(ticker, start, end, cache_dir, force_refresh=False):
        calls["n"] += 1
        if calls["n"] < 3:                 # fails twice, then works
            raise RuntimeError("yfinance returned no data")
        return _frame()

    monkeypatch.setattr(prices, "fetch_ticker", flaky)
    out = prices.fetch_all_tickers(["USDCAD=X"], "2026-01-01", "2026-10-17",
                                   tmp_path, attempts=4, backoff=0.0)
    assert list(out) == ["USDCAD=X"]
    assert calls["n"] == 3


def test_persistent_failure_still_raises(monkeypatch, tmp_path):
    def always_fails(ticker, start, end, cache_dir, force_refresh=False):
        raise RuntimeError("yfinance returned no data")

    monkeypatch.setattr(prices, "fetch_ticker", always_fails)
    with pytest.raises(RuntimeError, match="after 3 attempts"):
        prices.fetch_all_tickers(["USDCAD=X"], "2026-01-01", "2026-10-17",
                                 tmp_path, attempts=3, backoff=0.0)


def test_one_bad_ticker_fails_the_whole_fetch(monkeypatch, tmp_path):
    """The universe must stay complete: 14 of 15 pairs is a different model."""
    def one_bad(ticker, start, end, cache_dir, force_refresh=False):
        if ticker == "USDCAD=X":
            raise RuntimeError("yfinance returned no data")
        return _frame()

    monkeypatch.setattr(prices, "fetch_ticker", one_bad)
    with pytest.raises(RuntimeError, match="USDCAD=X"):
        prices.fetch_all_tickers(["EURUSD=X", "USDCAD=X"], "2026-01-01",
                                 "2026-10-17", tmp_path, attempts=2,
                                 backoff=0.0)
