"""Write the live forex report as markdown.

All numbers come from ``src.live.report``, which the Streamlit dashboard
(``app.py``) also uses — so the report and the dashboard can never disagree.

Usage:
    .venv/bin/python scripts/live_report.py
"""
from __future__ import annotations

import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.live.report import (  # noqa: E402
    BACKTEST_RIC,
    COST_BPS,
    REPORT_DIR,
    ROOT,
    START_BALANCE,
    build_report,
)


def main() -> None:
    rep = build_report()
    s = rep.stats
    ic = rep.ic.drop(columns=["hit_rate", "reconstructed"])

    today = datetime.date.today().isoformat()
    lines = [
        f"# Live forex report — {today}",
        "",
        f"Window: {s['window_from']} → {s['window_to']} ({s['n_weeks']} "
        "realizable signal weeks; the newest signal has no realized 5d return "
        "yet).",
        "",
        "## 1. Signal quality — weekly cross-sectional rank IC",
        "",
        "Spearman(predicted, realized 5d fwd log return) across all 15 pairs, "
        "per week.",
        "",
        ic.to_markdown(index=False, floatfmt=".4f"),
        "",
        f"- Mean CS-RIC: **{s['mean_ric']:+.4f}** (SE ≈ {s['se_ric']:.4f}), "
        f"positive weeks: {s['pos_weeks']}/{s['n_weeks']}",
        f"- Backtest OOS CS-RIC for this model: **{BACKTEST_RIC:+.3f}** "
        "(no stored weekly backtest IC distribution exists, so this is a point "
        "comparison, not a percentile test).",
        "",
        "## 2. Execution fidelity — signal book vs MT5 book (end of signal day)",
        "",
        rep.fidelity.drop(columns=["ok"]).to_markdown(index=False),
        "",
        "## 3. PnL — live vs paper",
        "",
        f"- Live closed-trade PnL across both demo accounts: "
        f"**{s['closed_pnl']:+.2f} USD** on {START_BALANCE:.0f} "
        f"({s['closed_pnl'] / START_BALANCE:+.2%}).",
        f"  - Expired account 372709 (Jun 2 – Aug 14): {s['old_pnl']:+.2f} USD "
        "from the profit column; its MT5 footer read −77.15 including swaps, "
        "i.e. ≈ −7.5 USD of swap the backtest does not model.",
        f"  - Current account 438689 (from Aug 14): {s['new_pnl']:+.2f} USD "
        f"closed, {s['n_open']} positions still open.",
        *([f"  - Floating P&L on those open positions: "
           f"{s['floating_pnl']:+.2f} USD; balance {s['balance']:.2f}, equity "
           f"{s['equity']:.2f} (snapshot {s['snapshot_at']})."]
          if s.get("floating_pnl") is not None else
          ["  - No account snapshot on disk yet, so floating P&L on open "
           "positions is not included."]),
        f"- Of which manual-entry fumbles (opened and closed within minutes): "
        f"{s['fumble_pnl']:+.2f} USD across {s['n_fumbles']} trades.",
        f"- Paper strategy (signal followed exactly, h=5 windows, 1/6 equal "
        f"weight, {COST_BPS:.0f} bps/side): **{s['paper_total']:+.2%}** "
        "cumulative simple sum of weekly net returns.",
        "",
        "## Honesty notes",
        "",
        f"- {s['n_weeks']} weekly observations: portfolio Sharpe/return over "
        "this window is statistically uninformative (SE of annualized Sharpe "
        "≈ ±2.2). It is deliberately not reported. The IC row count (15 pairs "
        "× weeks) is the only metric here with any power.",
        "- MT5 history for the first (expired) demo account was transcribed "
        "from a screenshot; `profit` values are as-displayed, three close "
        "prices were unreadable. History for the current account comes from "
        "the MetaApi API.",
        "- The most recent week's IC is provisional: it is computed from the "
        "price snapshot taken during the signal run, before that day's close "
        "settles. Values shift slightly once the data finalizes (2026-08-07 "
        "read +0.67 one week, +0.48 the next).",
        *([f"- Reconstructed signal weeks (predictions regenerated after the "
           f"original file was lost, so their IC is approximate): "
           f"{', '.join(s['reconstructed_weeks'])}."]
          if s["reconstructed_weeks"] else []),
        "- Research tooling — not investment advice.",
    ]
    out = REPORT_DIR / f"live_report_{today}.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nReport saved → {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
