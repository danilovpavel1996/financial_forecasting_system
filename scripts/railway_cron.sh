#!/usr/bin/env bash
# Weekly rebalance job, run by Railway cron (Fridays 15:00 UTC — railway.json).
#
#   1. rebalance.py         — fetch prices, train, generate + save the signal
#   2. execute_mt5.py --live — reconcile & trade on MT5 via MetaApi
#   3. fetch_mt5_history.py --undeploy — refresh trade history, then stop the
#      MetaApi cloud terminal until next week (saves hourly billing)
#   4. push the new artifacts (signal JSON, execution receipt, history CSV)
#      back to GitHub, because Railway containers are ephemeral. Requires a
#      GITHUB_TOKEN env var (fine-grained PAT, contents:write on this repo);
#      without it the artifacts are lost when the container exits.
#
# set -e: any failure aborts the rest — in particular, a failed execution
# leaves the MetaApi terminal deployed for manual inspection, and the
# executor itself sends a Telegram alert if TELEGRAM_* vars are set.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

python scripts/rebalance.py
python scripts/execute_mt5.py --live
python scripts/fetch_mt5_history.py --undeploy

if [[ -n "${GITHUB_TOKEN:-}" ]]; then
  repo="github.com/danilovpavel1996/financial_forecasting_system.git"
  tmp=$(mktemp -d)
  git clone --depth 1 "https://x-access-token:${GITHUB_TOKEN}@${repo}" "$tmp"
  mkdir -p "$tmp/outputs/signals" "$tmp/outputs/executions" "$tmp/data/live"
  cp outputs/signals/signal_forex_*.json "$tmp/outputs/signals/"
  cp outputs/executions/execution_*.json "$tmp/outputs/executions/" 2>/dev/null || true
  cp data/live/mt5_history_metaapi.csv "$tmp/data/live/" 2>/dev/null || true
  cd "$tmp"
  git config user.email "railway-bot@users.noreply.github.com"
  git config user.name "Railway weekly rebalance"
  git add outputs/signals outputs/executions data/live/mt5_history_metaapi.csv
  if ! git diff --cached --quiet; then
    git commit -m "Weekly rebalance artifacts $(date -u +%F)"
    git push origin HEAD:main
  fi
else
  echo "WARNING: GITHUB_TOKEN not set — this week's signal and receipt will be"
  echo "lost when the container exits. Set it in Railway service variables."
fi

# Last: health checks. A non-zero exit here fails the cron run AFTER all the
# real work is done, which pushes a Railway app notification telling the
# human to act (demo account expiring, MetaApi balance low).
cd "$ROOT"
python scripts/preflight_check.py
