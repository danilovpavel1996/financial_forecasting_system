"""Tests for the pure reconciliation/guard logic in scripts/execute_mt5.py.

These run without metaapi-cloud-sdk installed — the SDK is only imported
inside the async runner.
"""
import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.execute_mt5 import (
    LOT,
    actual_book,
    guard_violations,
    plan_orders,
    target_book,
)


def _signal(date="2026-08-14", entries=None):
    entries = entries if entries is not None else [
        ("AUDJPY", "LONG"), ("EURJPY", "LONG"), ("USDJPY", "LONG"),
        ("EURUSD", "SHORT"), ("GBPUSD", "SHORT"), ("USDCHF", "SHORT"),
    ]
    return {
        "date": date,
        "rankings": [
            {"mt5": s, "position": p} for s, p in entries
        ] + [{"mt5": "EURGBP", "position": "FLAT"}],
    }


def _pos(symbol, side, volume=LOT):
    return {"symbol": symbol, "volume": volume, "id": f"id-{symbol}",
            "type": "POSITION_TYPE_BUY" if side == "buy" else "POSITION_TYPE_SELL"}


TODAY = datetime.date(2026, 8, 14)


def test_target_book_ignores_flat():
    book = target_book(_signal())
    assert book == {"AUDJPY": "buy", "EURJPY": "buy", "USDJPY": "buy",
                    "EURUSD": "sell", "GBPUSD": "sell", "USDCHF": "sell"}


def test_plan_noop_when_books_match():
    target = target_book(_signal())
    actual = dict(target)
    closes, opens = plan_orders(actual, target)
    assert closes == [] and opens == []


def test_plan_direction_flip_closes_then_reopens():
    # EURUSD held long, signal says short → must appear in BOTH lists
    target = {"EURUSD": "sell"}
    actual = {"EURUSD": "buy"}
    closes, opens = plan_orders(actual, target)
    assert closes == ["EURUSD"]
    assert opens == [("EURUSD", "sell")]


def test_plan_open_new_and_close_dropped():
    target = {"AUDJPY": "buy", "USDCHF": "sell"}
    actual = {"USDCHF": "sell", "CADJPY": "buy"}   # CADJPY dropped from signal
    closes, opens = plan_orders(actual, target)
    assert closes == ["CADJPY"]
    assert opens == [("AUDJPY", "buy")]
    # unchanged USDCHF is touched by neither list
    assert "USDCHF" not in closes
    assert all(s != "USDCHF" for s, _ in opens)


def test_actual_book_maps_metaapi_types():
    book = actual_book([_pos("EURUSD", "buy"), _pos("USDCHF", "sell")])
    assert book == {"EURUSD": "buy", "USDCHF": "sell"}


def test_guard_stale_signal():
    problems = guard_violations(_signal(date="2026-08-07"), TODAY, [], False)
    assert any("stale" in p for p in problems)
    assert guard_violations(_signal(date="2026-08-07"), TODAY, [], True) == []


def test_guard_position_cap():
    entries = [(s, "LONG") for s in
               ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF",
                "NZDUSD"]]                      # 7 > cap of 6
    problems = guard_violations(_signal(entries=entries), TODAY, [], False)
    assert any("hard cap" in p for p in problems)


def test_guard_foreign_symbol_on_account():
    problems = guard_violations(_signal(), TODAY, [_pos("XAUUSD", "buy")], False)
    assert any("foreign symbol" in p for p in problems)


def test_guard_offsize_volume():
    problems = guard_violations(_signal(), TODAY, [_pos("EURUSD", "buy", volume=0.5)], False)
    assert any("volume" in p for p in problems)


def test_guard_clean_account_passes():
    positions = [_pos("EURUSD", "buy"), _pos("USDCHF", "sell")]
    assert guard_violations(_signal(), TODAY, positions, False) == []
