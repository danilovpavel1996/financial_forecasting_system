"""Fetch MT5 trade history from MetaApi into a CSV for live_report.py.

Pulls all deals since the account's first day, pairs entry/exit deals by
position id, and writes ``data/live/mt5_history_metaapi.csv`` in the same
schema as the hand-transcribed history of the first (expired) demo account.
Still-open positions get an empty close_time, matching that schema.

The ``profit`` column is the raw MT5 profit; swap and commission are summed
into the ``note`` column so the report can quantify carry drag separately.

Usage:
    .venv/bin/python scripts/fetch_mt5_history.py              # fetch only
    .venv/bin/python scripts/fetch_mt5_history.py --undeploy   # + stop the
        MetaApi cloud terminal afterwards (stops hourly billing; positions
        remain open at the broker; the executor redeploys automatically)
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import datetime
import json
import os
from pathlib import Path

ROOT     = Path(__file__).resolve().parent.parent
OUT_CSV  = ROOT / "data" / "live" / "mt5_history_metaapi.csv"
SNAPSHOT = ROOT / "data" / "live" / "account_snapshot.json"
ACCOUNT_START = datetime.datetime(2026, 8, 14)   # DEMO_002 creation day


def deals_to_trades(deals: list[dict]) -> list[dict]:
    """Pair DEAL_ENTRY_IN / DEAL_ENTRY_OUT deals into closed/open trades."""
    by_pos: dict[str, dict] = {}
    for d in deals:
        if d.get("type") not in ("DEAL_TYPE_BUY", "DEAL_TYPE_SELL"):
            continue                                # skip balance/credit deals
        pid = d.get("positionId")
        if pid is None:
            continue
        t = by_pos.setdefault(pid, {
            "symbol": d.get("symbol", ""), "volume": d.get("volume", ""),
            "open_time": "", "open_price": "", "side": "",
            "close_time": "", "close_price": "", "profit": None,
            "swap": 0.0, "commission": 0.0,
        })
        t["swap"] += d.get("swap", 0.0) or 0.0
        t["commission"] += d.get("commission", 0.0) or 0.0
        if d.get("entryType") == "DEAL_ENTRY_IN":
            t["open_time"]  = d["time"].strftime("%Y-%m-%d %H:%M:%S")
            t["open_price"] = d.get("price", "")
            t["side"] = "buy" if d["type"] == "DEAL_TYPE_BUY" else "sell"
        elif d.get("entryType") == "DEAL_ENTRY_OUT":
            t["close_time"]  = d["time"].strftime("%Y-%m-%d %H:%M:%S")
            t["close_price"] = d.get("price", "")
            t["profit"] = (t["profit"] or 0.0) + (d.get("profit", 0.0) or 0.0)
    trades = sorted(by_pos.values(), key=lambda t: t["open_time"])
    for t in trades:
        extras = f"swap={t.pop('swap'):.2f};commission={t.pop('commission'):.2f}"
        t["note"] = ("still_open;" if not t["close_time"] else "") + extras
        t["symbol"] = t["symbol"].lower()
        if t["profit"] is not None:
            t["profit"] = round(t["profit"], 2)
        else:
            t["profit"] = ""
    return trades


def write_csv(trades: list[dict]) -> None:
    cols = ["open_time", "symbol", "side", "volume", "open_price",
            "close_time", "close_price", "profit", "note"]
    with open(OUT_CSV, "w", newline="") as f:
        f.write("# MT5 history for demo login 438689 (DEMO_002), fetched from "
                "MetaApi.\n# Regenerate with scripts/fetch_mt5_history.py — "
                "do not edit by hand.\n")
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for t in trades:
            w.writerow({c: t.get(c, "") for c in cols})


async def run(args: argparse.Namespace) -> None:
    from dotenv import load_dotenv
    load_dotenv()
    token      = os.environ.get("METAAPI_TOKEN")
    account_id = os.environ.get("METAAPI_ACCOUNT_ID")
    if not token or not account_id:
        raise SystemExit("METAAPI_TOKEN / METAAPI_ACCOUNT_ID missing from .env.")

    from metaapi_cloud_sdk import MetaApi

    api     = MetaApi(token)
    account = await api.metatrader_account_api.get_account(account_id)
    if account.state != "DEPLOYED":
        print("  Deploying account…")
        await account.deploy()
    await account.wait_connected()
    connection = account.get_rpc_connection()
    await connection.connect()
    await connection.wait_synchronized()

    end = datetime.datetime.now() + datetime.timedelta(days=1)
    res = await connection.get_deals_by_time_range(ACCOUNT_START, end)
    trades = deals_to_trades(res.get("deals", []))
    write_csv(trades)

    # Snapshot balance/equity/floating P&L so the dashboard can show the real
    # account total without deploying the terminal on every page load.
    info = await connection.get_account_information()
    positions = await connection.get_positions()
    SNAPSHOT.write_text(json.dumps({
        "fetched_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "login": info.get("login"),
        "balance": info.get("balance"),
        "equity": info.get("equity"),
        "margin": info.get("margin"),
        "floating_pnl": round(sum(p.get("profit", 0.0) for p in positions), 2),
        "positions": [{
            "symbol": p["symbol"],
            "side": "buy" if p["type"] == "POSITION_TYPE_BUY" else "sell",
            "volume": p.get("volume"),
            "openPrice": p.get("openPrice"),
            "currentPrice": p.get("currentPrice"),
            "profit": p.get("profit"),
            "swap": p.get("swap"),
        } for p in positions],
    }, indent=2), encoding="utf-8")
    print(f"  Account: balance {info.get('balance'):.2f}, equity "
          f"{info.get('equity'):.2f}, floating "
          f"{sum(p.get('profit', 0.0) for p in positions):+.2f} USD")
    print(f"  Snapshot → {SNAPSHOT.relative_to(ROOT)}")

    closed = [t for t in trades if t["profit"] != ""]
    total  = sum(t["profit"] for t in closed)
    print(f"  {len(trades)} trades ({len(closed)} closed, "
          f"{len(trades) - len(closed)} open), closed profit {total:+.2f} USD")
    print(f"  Written → {OUT_CSV.relative_to(ROOT)}")

    if args.undeploy:
        print("  Undeploying account (stops MetaApi hourly billing; "
              "positions stay open at the broker)…")
        await account.undeploy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch MT5 history via MetaApi")
    parser.add_argument("--undeploy", action="store_true",
                        help="undeploy the MetaApi cloud terminal after fetching")
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
