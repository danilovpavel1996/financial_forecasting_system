"""Weekly health checks that turn into Railway push notifications.

Runs LAST in scripts/railway_cron.sh (after trading and artifact push, so a
warning never blocks the rebalance). Exits non-zero when human action is
needed — that marks the cron run FAILED in Railway, which pushes a
notification to the Railway mobile app. The log then says exactly what to do.

Checks:
  1. Fusion Markets demo expiry — demo accounts die 30 days after creation.
     Expiry date comes from env var DEMO_EXPIRES (YYYY-MM-DD); update it in
     Railway → service → Variables each time a new demo is created.
     Warns when <= WARN_DAYS days remain.
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

DEFAULT_DEMO_EXPIRES = "2026-09-13"   # DEMO_002 (login 438689), created 2026-08-14
# Checked only weekly (Fridays), so the window must exceed one week or the
# warning could arrive days before expiry instead of a full run ahead.
WARN_DAYS            = 10
BALANCE_WARN_USD     = 3.0
BILLING_URL = ("https://billing-api-v1.agiliumtrade.agiliumtrade.ai"
               "/users/current/balance")


def check_demo_expiry() -> list[str]:
    expires = datetime.date.fromisoformat(
        os.environ.get("DEMO_EXPIRES", DEFAULT_DEMO_EXPIRES))
    days_left = (expires - datetime.date.today()).days
    if days_left < 0:
        return [f"Fusion demo EXPIRED {-days_left} day(s) ago ({expires}). "
                "Create a new demo in the Fusion hub, update the account "
                "login/password in MetaApi, and set DEMO_EXPIRES in Railway "
                "variables to the new expiry."]
    if days_left <= WARN_DAYS:
        return [f"Fusion demo expires in {days_left} day(s) ({expires}). "
                "Create a new demo account soon: Fusion hub → Create New "
                "Demo (MT5, USD, 2000), then update the MetaApi account "
                "credentials and the DEMO_EXPIRES Railway variable."]
    print(f"  Demo expiry OK: {days_left} days left (expires {expires}).")
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
        return [f"MetaApi balance is ${balance:.2f} (< ${BALANCE_WARN_USD:.0f}). "
                "Top up at app.metaapi.cloud → Billing → Deposit, or the "
                "account cannot deploy and the weekly rebalance will stop."]
    print(f"  MetaApi balance OK: ${balance:.2f}.")
    return []


def main() -> None:
    from dotenv import load_dotenv
    load_dotenv()

    print("\nPreflight checks")
    print("────────────────")
    warnings = check_demo_expiry() + check_metaapi_balance()
    if warnings:
        print("\n⚠️  ACTION NEEDED — failing this run so Railway notifies you:")
        for w in warnings:
            print(f"  • {w}")
        sys.exit(1)
    print("  All checks passed.")


if __name__ == "__main__":
    main()
