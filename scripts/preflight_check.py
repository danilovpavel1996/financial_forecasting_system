"""Weekly health checks that turn into Railway push notifications.

Runs LAST in scripts/railway_cron.sh (after trading and artifact push, so a
warning never blocks the rebalance). Exits non-zero when human action is
needed — that marks the cron run FAILED in Railway, which pushes a
notification to the Railway mobile app. The log then says exactly what to do.

Checks:
  1. Fusion Markets demo age — reported, never fatal. Fusion does not
     enforce its own 30-day expiry (two accounts have outlived it), so a
     dead account is detected by the executor failing to connect, not by a
     calendar guess.
  2. MetaApi billing balance — queried from the billing API with the normal
     METAAPI_TOKEN. Warns when below BALANCE_WARN_USD (deployment stops
     working entirely at $0, which would kill the whole pipeline).

Usage:
    .venv/bin/python scripts/preflight_check.py
"""
from __future__ import annotations

import datetime
import json
import os
import sys
import urllib.request
from pathlib import Path

# MetaApi auto-deposit tops the balance up on its own, and its trigger sits
# below the $3 this check originally used — so $3 fired every Friday while
# nothing was wrong. The threshold now only trips when auto-deposit has
# genuinely failed (a declined card, a cancelled method), which still leaves
# roughly three weeks of runway at the observed ~$0.30/week burn.
BALANCE_WARN_USD     = 1.0
BILLING_URL = ("https://billing-api-v1.agiliumtrade.agiliumtrade.ai"
               "/users/current/balance")


def note_demo_age() -> list[str]:
    """Report the demo's nominal expiry, but never fail the run over it.

    This used to fail the run when DEMO_EXPIRES came within ten days, which
    produced a red notification every Friday for weeks while the account kept
    trading perfectly. Fusion has now ignored its own 30-day rule on two
    accounts in a row: 438689 was pronounced dead on 2026-09-13 and traded
    until 2026-10-02, and 471278 passed 2026-10-11 and was still live on
    2026-10-12. A date nobody enforces is not a signal, and crying wolf weekly
    teaches you to ignore the alerts that do matter.

    The real failure mode needs no guesswork: if the account ever dies, the
    executor cannot connect and the run fails loudly by itself.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.live.report import demo_expiry

    expires = demo_expiry()
    days = (expires - datetime.date.today()).days
    if days < 0:
        print(f"  Demo past its nominal expiry ({expires}, {-days} day(s) "
              "ago) but still trading — informational only.")
    else:
        print(f"  Demo nominal expiry {expires} ({days} day(s)).")
    return []


def check_metaapi_balance() -> list[str]:
    token = os.environ.get("METAAPI_TOKEN")
    if not token:
        return ["METAAPI_TOKEN missing — cannot check billing balance."]
    try:
        req = urllib.request.Request(BILLING_URL, headers={"auth-token": token})
        with urllib.request.urlopen(req, timeout=30) as r:
            balance = float(json.load(r)["amount"])
    except Exception as exc:
        # Don't page the human for a transient API blip; just log it.
        print(f"  MetaApi balance check skipped (API error: {exc}).")
        return []
    if balance < BALANCE_WARN_USD:
        return [f"MetaApi balance is ${balance:.2f} (< "
                f"${BALANCE_WARN_USD:.2f}). Auto-deposit should have topped "
                "this up, so check the payment method at app.metaapi.cloud → "
                "Billing. At $0 the account cannot deploy and the weekly "
                "rebalance stops entirely."]
    print(f"  MetaApi balance OK: ${balance:.2f}.")
    return []


def check_artifact_push() -> list[str]:
    """A run that trades but loses its records is a silent failure — page it.

    Only meaningful on Railway (ephemeral container); skipped locally, where
    artifacts are simply written to the working tree.
    """
    if not os.environ.get("RAILWAY_ENVIRONMENT_NAME"):
        print("  Artifact push check skipped (not running on Railway).")
        return []
    if not os.environ.get("GITHUB_TOKEN"):
        return ["GITHUB_TOKEN is not set — this week's signal JSON, execution "
                "receipt and trade history were LOST when the container "
                "exited. Trading itself was unaffected, but the evaluation "
                "record has a hole in it. Fix: create a fine-grained GitHub "
                "PAT (contents: read and write on "
                "danilovpavel1996/financial_forecasting_system) and add it as "
                "GITHUB_TOKEN in Railway → service → Variables."]
    print("  Artifact push OK: GITHUB_TOKEN is set.")
    return []


def check_trade_history() -> list[str]:
    """The report is only as good as the trade history behind it.

    The transcribed history of the expired demo account cannot be re-fetched
    from anywhere — if it is absent from the image, the weekly report silently
    understates P&L and reports every affected week as an execution failure.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.live.report import MT5_CSVS

    missing = [p.name for p in MT5_CSVS if not p.exists()]
    if missing:
        return [f"Trade history file(s) missing: {', '.join(missing)}. This "
                "week's report understates realized P&L and shows the affected "
                "weeks as execution failures. Check .dockerignore is not "
                "excluding data/live/*.csv from the image."]
    print("  Trade history OK: all history files present.")
    return []


def main() -> None:
    from dotenv import load_dotenv
    load_dotenv()

    print("\nPreflight checks")
    print("────────────────")
    warnings = (note_demo_age() + check_metaapi_balance()
                + check_artifact_push() + check_trade_history())
    if warnings:
        print("\n⚠️  ACTION NEEDED — failing this run so Railway notifies you:")
        for w in warnings:
            print(f"  • {w}")
        sys.exit(1)
    print("  All checks passed.")


if __name__ == "__main__":
    main()
