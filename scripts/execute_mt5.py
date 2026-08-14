"""Execute the latest forex signal on MT5 via MetaApi (cloud bridge).

Reads the newest ``outputs/signals/signal_forex_<date>.json`` (produced by
``scripts/rebalance.py``), reconciles it against the ACTUAL open positions on
the MetaApi-connected account, and closes/opens 0.01-lot market positions so
the account matches the signal book.

Safety model (all violations abort before any order is sent):
- DRY RUN is the default; ``--live`` is required to send real orders.
- Reconciliation uses the account's actual positions as ground truth, never
  the previous signal file.
- Refuses to run twice for the same signal date (execution receipt in
  ``outputs/executions/``) unless ``--force``.
- Refuses a signal not generated today unless ``--allow-stale``.
- Hard caps: at most MAX_POSITIONS positions, LOT lots each, only symbols in
  the 15-pair universe. Foreign symbols or off-size volumes on the account
  mean a human intervened → abort and ask the human.
- Any order failure stops execution immediately (no silent partial books).
- Optional Telegram notification after every run (set TELEGRAM_BOT_TOKEN and
  TELEGRAM_CHAT_ID in .env); silent no-op if unset.

Environment (.env): METAAPI_TOKEN, METAAPI_ACCOUNT_ID.

Usage:
    .venv/bin/python scripts/execute_mt5.py             # dry run
    .venv/bin/python scripts/execute_mt5.py --live      # send orders

Research tooling on a demo account — not investment advice.
"""
from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROOT       = Path(__file__).resolve().parent.parent
SIGNAL_DIR = ROOT / "outputs" / "signals"
EXEC_DIR   = ROOT / "outputs" / "executions"

LOT           = 0.01
MAX_POSITIONS = 6
UNIVERSE = {
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD",
    "EURJPY", "GBPJPY", "EURGBP", "AUDJPY", "EURAUD", "GBPAUD", "AUDNZD",
    "CADJPY",
}


# ── Pure planning logic (unit-tested in tests/test_execute_mt5.py) ────────────

def target_book(signal: dict) -> dict[str, str]:
    """{symbol: 'buy'|'sell'} for the signal's LONG/SHORT entries."""
    book = {}
    for r in signal.get("rankings", []):
        if r.get("position") == "LONG":
            book[r["mt5"]] = "buy"
        elif r.get("position") == "SHORT":
            book[r["mt5"]] = "sell"
    return book


def actual_book(positions: list[dict]) -> dict[str, str]:
    """{symbol: 'buy'|'sell'} from MetaApi position dicts."""
    return {
        p["symbol"]: ("buy" if p["type"] == "POSITION_TYPE_BUY" else "sell")
        for p in positions
    }


def plan_orders(actual: dict[str, str], target: dict[str, str],
                ) -> tuple[list[str], list[tuple[str, str]]]:
    """Return (symbols_to_close, [(symbol, side)] to open).

    A direction flip appears in both lists (close first, then reopen).
    """
    closes = sorted(
        s for s in actual if s not in target or target[s] != actual[s]
    )
    opens = sorted(
        (s, side) for s, side in target.items()
        if s not in actual or actual[s] != side
    )
    return closes, opens


def guard_violations(signal: dict, today: datetime.date,
                     positions: list[dict], allow_stale: bool) -> list[str]:
    """List of human-readable reasons NOT to trade. Empty list == safe."""
    problems = []
    sig_date = datetime.date.fromisoformat(signal["date"])
    if sig_date != today and not allow_stale:
        problems.append(
            f"signal is dated {sig_date}, today is {today} — stale signal "
            "(re-run rebalance.py, or pass --allow-stale)"
        )
    target = target_book(signal)
    if len(target) > MAX_POSITIONS:
        problems.append(
            f"signal wants {len(target)} positions, hard cap is {MAX_POSITIONS}"
        )
    bad_syms = sorted(t for t in target if t not in UNIVERSE)
    if bad_syms:
        problems.append(f"signal contains symbols outside universe: {bad_syms}")
    for p in positions:
        if p["symbol"] not in UNIVERSE:
            problems.append(
                f"account holds foreign symbol {p['symbol']} — a human "
                "intervened; refusing to touch this account"
            )
        if abs(p.get("volume", LOT) - LOT) > 1e-9:
            problems.append(
                f"account position {p['symbol']} has volume {p.get('volume')} "
                f"≠ {LOT} — a human intervened; refusing to trade"
            )
    return problems


# ── Notification ──────────────────────────────────────────────────────────────

def notify(text: str) -> None:
    """Telegram message if configured, else no-op. Never raises."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat  = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat:
        return
    try:
        data = urllib.parse.urlencode({"chat_id": chat, "text": text}).encode()
        urllib.request.urlopen(
            f"https://api.telegram.org/bot{token}/sendMessage", data, timeout=10
        )
    except Exception as exc:  # notification failure must not kill execution
        print(f"  (Telegram notification failed: {exc})")


# ── Signal / receipt IO ───────────────────────────────────────────────────────

def load_latest_signal() -> dict:
    files = sorted(SIGNAL_DIR.glob("signal_forex_*.json"))
    if not files:
        sys.exit("No signal files found — run scripts/rebalance.py first.")
    return json.loads(files[-1].read_text(encoding="utf-8"))


def receipt_path(sig_date: str) -> Path:
    return EXEC_DIR / f"execution_{sig_date}.json"


# ── MetaApi execution ─────────────────────────────────────────────────────────

async def run(args: argparse.Namespace) -> None:
    from dotenv import load_dotenv
    load_dotenv()

    signal = load_latest_signal()
    today  = datetime.date.today()
    target = target_book(signal)
    mode   = "LIVE" if args.live else "DRY RUN"

    print(f"\nMT5 Executor — {mode}")
    print("═" * 40)
    print(f"  Signal: {signal['date']}  ({len(target)} target positions)")

    rp = receipt_path(signal["date"])
    if rp.exists() and not args.force:
        sys.exit(
            f"⚠️  Already executed signal {signal['date']} "
            f"(receipt: {rp.relative_to(ROOT)}). Use --force to override."
        )

    token      = os.environ.get("METAAPI_TOKEN")
    account_id = os.environ.get("METAAPI_ACCOUNT_ID")
    if not token or not account_id:
        sys.exit("METAAPI_TOKEN / METAAPI_ACCOUNT_ID missing from .env.")

    from metaapi_cloud_sdk import MetaApi

    api     = MetaApi(token)
    account = await api.metatrader_account_api.get_account(account_id)
    if account.state != "DEPLOYED":
        print("  Deploying account…")
        await account.deploy()
    print("  Waiting for broker connection…")
    await account.wait_connected()
    connection = account.get_rpc_connection()
    await connection.connect()
    await connection.wait_synchronized()

    positions = await connection.get_positions()
    actual    = actual_book(positions)
    print(f"  Account positions: {actual or '(none)'}")

    problems = guard_violations(signal, today, positions, args.allow_stale)
    if problems:
        msg = "MT5 executor ABORTED:\n- " + "\n- ".join(problems)
        print(f"\n⛔ {msg}")
        notify(msg)
        sys.exit(2)

    closes, opens = plan_orders(actual, target)
    print(f"\n  Plan: close {closes or 'nothing'}, "
          f"open {[f'{side} {s}' for s, side in opens] or 'nothing'}")

    results: list[dict] = []
    if not args.live:
        print("\n  DRY RUN — no orders sent. Re-run with --live to execute.")
        notify(
            f"MT5 dry run {signal['date']}: would close {closes or 'nothing'}, "
            f"open {[f'{side} {s}' for s, side in opens] or 'nothing'}."
        )
        await _maybe_undeploy(account, args)
        return

    pos_by_symbol = {p["symbol"]: p for p in positions}
    try:
        for sym in closes:
            print(f"  Closing {sym}…")
            r = await connection.close_position(pos_by_symbol[sym]["id"])
            results.append({"action": "close", "symbol": sym, "result": str(r)})
        for sym, side in opens:
            print(f"  Opening {side} {LOT} {sym}…")
            opts = {"comment": f"reb-{signal['date']}"}
            if side == "buy":
                r = await connection.create_market_buy_order(sym, LOT, options=opts)
            else:
                r = await connection.create_market_sell_order(sym, LOT, options=opts)
            results.append({"action": "open", "symbol": sym, "side": side,
                            "result": str(r)})
    except Exception as exc:
        msg = (f"MT5 executor FAILED mid-rebalance on signal {signal['date']}: "
               f"{exc}. Executed so far: {results}. "
               "Account book may be PARTIAL — check MT5 manually.")
        print(f"\n⛔ {msg}")
        notify(msg)
        _write_receipt(rp, signal, actual, closes, opens, results,
                       status="FAILED_PARTIAL", verified=None)
        sys.exit(2)

    # Post-trade verification against the account, not against our own plan.
    final = actual_book(await connection.get_positions())
    ok    = final == target
    _write_receipt(rp, signal, actual, closes, opens, results,
                   status="OK" if ok else "MISMATCH", verified=final)

    if ok:
        msg = (f"MT5 rebalance {signal['date']} done: "
               f"closed {len(closes)}, opened {len(opens)}, "
               f"book = {final}")
        print(f"\n✅ {msg}")
    else:
        msg = (f"MT5 rebalance {signal['date']}: orders sent but final book "
               f"{final} ≠ target {target} — check MT5 manually!")
        print(f"\n⚠️  {msg}")
    notify(msg)
    print(f"  Receipt → {rp.relative_to(ROOT)}")
    # On success only — after a partial failure we keep the terminal deployed
    # so a human can inspect/fix via API without waiting for a redeploy.
    await _maybe_undeploy(account, args)


async def _maybe_undeploy(account, args) -> None:
    if args.undeploy:
        print("  Undeploying MetaApi terminal (stops hourly billing; "
              "positions stay open at the broker)…")
        await account.undeploy()


def _write_receipt(path: Path, signal: dict, actual_before: dict,
                   closes: list, opens: list, results: list,
                   status: str, verified: dict | None) -> None:
    EXEC_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "executed_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "signal_date": signal["date"],
        "status": status,
        "book_before": actual_before,
        "planned_closes": closes,
        "planned_opens": opens,
        "order_results": results,
        "book_after_verified": verified,
    }, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute forex signal on MT5 via MetaApi")
    parser.add_argument("--live", action="store_true",
                        help="actually send orders (default: dry run)")
    parser.add_argument("--force", action="store_true",
                        help="re-run even if a receipt exists for this signal date")
    parser.add_argument("--allow-stale", action="store_true",
                        help="execute a signal not generated today")
    parser.add_argument("--undeploy", action="store_true",
                        help="undeploy the MetaApi cloud terminal after a "
                             "successful run (saves hourly billing)")
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
