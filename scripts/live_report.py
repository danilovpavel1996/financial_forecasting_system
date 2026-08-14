"""Live-trading diagnostic report: signals vs realized returns vs MT5 execution.

Answers three questions the tiny live sample CAN answer honestly:
  1. Is the signal working?  Weekly cross-sectional rank IC (Spearman between
     predicted 5d return and realized 5d forward log return, across all 15
     pairs) — 15 data points per week, far higher-powered than portfolio PnL.
  2. Did execution match the signal?  Reconstructs the MT5 position book at
     the end of each signal day from the transcribed trade history and diffs
     it against the signal's LONG/SHORT book.
  3. Paper vs live PnL — what the signal alone would have earned (h=5,
     equal-weight 1/6, net of cost_bps per side) vs the account's realized USD.

It does NOT compute a live Sharpe: with ~10 weekly observations the standard
error is so wide the number would be noise, and reporting it would violate
the honest-reporting rule.

Usage:
    .venv/bin/python scripts/live_report.py
"""
from __future__ import annotations

import csv
import datetime
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT       = Path(__file__).resolve().parent.parent
SIGNAL_DIR = ROOT / "outputs" / "signals"
LIVE_DIR   = ROOT / "data" / "live"
REPORT_DIR = ROOT / "outputs" / "reports"
MT5_CSV    = LIVE_DIR / "mt5_history_2026-08-14.csv"

HORIZON       = 5
COST_BPS      = 3.0          # per side, same as backtest
BACKTEST_RIC  = 0.071        # phase-18 OOS cross-sectional rank IC (mean)
START_BALANCE = 2000.0


# ── Loaders ────────────────────────────────────────────────────────────────────

def load_signals() -> list[dict]:
    sigs = []
    for p in sorted(SIGNAL_DIR.glob("signal_forex_*.json")):
        sigs.append(json.loads(p.read_text()))
    return sigs


def load_prices() -> dict[str, pd.Series]:
    """Close series per mt5 symbol from the most recent live cache dir."""
    latest = sorted(d for d in LIVE_DIR.iterdir() if d.is_dir())[-1]
    out: dict[str, pd.Series] = {}
    for p in (latest / "prices").glob("*_eq_X_*.parquet"):
        sym = p.name.split("_eq_X_")[0]          # e.g. EURUSD
        df = pd.read_parquet(p)
        out[sym] = df["Close"].dropna()
    return out


def load_mt5_trades() -> list[dict]:
    rows = []
    with open(MT5_CSV) as f:
        for row in csv.DictReader(l for l in f if not l.startswith("#")):
            rows.append({
                "open_time":  pd.Timestamp(row["open_time"]),
                "symbol":     row["symbol"].upper(),
                "side":       row["side"],
                "close_time": pd.Timestamp(row["close_time"]) if row["close_time"] else None,
                "profit":     float(row["profit"]) if row["profit"] else None,
                "note":       row["note"],
            })
    return rows


# ── Realized returns ───────────────────────────────────────────────────────────

def fwd_log_return(prices: pd.Series, asof: datetime.date, h: int) -> float | None:
    """h-trading-day forward log return starting at the last close <= asof."""
    idx = prices.index[prices.index <= pd.Timestamp(asof)]
    if len(idx) == 0:
        return None
    i0 = prices.index.get_loc(idx[-1])
    i1 = i0 + h
    if i1 >= len(prices):
        return None
    return math.log(prices.iloc[i1] / prices.iloc[i0])


# ── 1. Weekly cross-sectional rank IC ─────────────────────────────────────────

def ic_table(sigs: list[dict], prices: dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    for s in sigs:
        asof = datetime.date.fromisoformat(s["feature_date"])
        pred, real, hits, n_act = [], [], 0, 0
        for r in s["rankings"]:
            fr = fwd_log_return(prices.get(r["mt5"], pd.Series(dtype=float)), asof, HORIZON)
            if fr is None:
                continue
            pred.append(r["predicted_return"])
            real.append(fr)
            if r["position"] in ("LONG", "SHORT"):
                n_act += 1
                sign = 1.0 if r["position"] == "LONG" else -1.0
                hits += int(sign * fr > 0)
        if len(pred) < 10:      # week not realizable yet
            continue
        ric = spearmanr(pred, real).statistic
        # paper portfolio: equal-weight 1/6 active book, h=5, cost both sides
        act = [(r["mt5"], 1.0 if r["position"] == "LONG" else -1.0)
               for r in s["rankings"] if r["position"] in ("LONG", "SHORT")]
        pr = np.mean([sgn * fwd_log_return(prices[t], asof, HORIZON) for t, sgn in act])
        pr_net = pr - 2 * COST_BPS / 1e4
        rows.append({"date": s["date"], "cs_ric": ric, "n_pairs": len(pred),
                     "active_hit": f"{hits}/{n_act}", "paper_ret_net": pr_net})
    return pd.DataFrame(rows)


# ── 2. Execution fidelity ─────────────────────────────────────────────────────

def book_at(trades: list[dict], t: pd.Timestamp) -> dict[str, str]:
    book = {}
    for tr in trades:
        if tr["open_time"] <= t and (tr["close_time"] is None or tr["close_time"] > t):
            book[tr["symbol"]] = tr["side"]
    return book


def fidelity_table(sigs: list[dict], trades: list[dict]) -> pd.DataFrame:
    rows = []
    for s in sigs:
        d = datetime.date.fromisoformat(s["date"])
        target = {r["mt5"]: ("buy" if r["position"] == "LONG" else "sell")
                  for r in s["rankings"] if r["position"] in ("LONG", "SHORT")}
        actual = book_at(trades, pd.Timestamp(d) + pd.Timedelta(hours=23, minutes=59))
        missing = sorted(t for t in target if t not in actual)
        extra   = sorted(t for t in actual if t not in target)
        wrong   = sorted(t for t in target if t in actual and target[t] != actual[t])
        ok = not (missing or extra or wrong)
        rows.append({"date": s["date"], "match": "✓" if ok else "✗",
                     "missing": ",".join(missing) or "—",
                     "extra": ",".join(extra) or "—",
                     "wrong_dir": ",".join(wrong) or "—"})
    return pd.DataFrame(rows)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    sigs   = load_signals()
    prices = load_prices()
    trades = load_mt5_trades()

    ic  = ic_table(sigs, prices)
    fid = fidelity_table(sigs, trades)

    closed  = [t for t in trades if t["profit"] is not None]
    fumbles = [t for t in closed if "fumble" in t["note"]]
    live_pnl    = sum(t["profit"] for t in closed)
    fumble_pnl  = sum(t["profit"] for t in fumbles)
    paper_total = float(ic["paper_ret_net"].sum())

    mean_ric = float(ic["cs_ric"].mean())
    se_ric   = float(ic["cs_ric"].std(ddof=1) / math.sqrt(len(ic)))
    pos_wk   = int((ic["cs_ric"] > 0).sum())

    today = datetime.date.today().isoformat()
    lines = [
        f"# Live forex report — {today}",
        "",
        f"Window: {ic['date'].iloc[0]} → {ic['date'].iloc[-1]} "
        f"({len(ic)} realizable signal weeks; the newest signal has no realized 5d return yet).",
        "",
        "## 1. Signal quality — weekly cross-sectional rank IC",
        "",
        "Spearman(predicted, realized 5d fwd log return) across all 15 pairs, per week.",
        "",
        ic.to_markdown(index=False, floatfmt=".4f"),
        "",
        f"- Mean CS-RIC: **{mean_ric:+.4f}** (SE ≈ {se_ric:.4f}), "
        f"positive weeks: {pos_wk}/{len(ic)}",
        f"- Backtest OOS CS-RIC for this model: **{BACKTEST_RIC:+.3f}** "
        "(no stored weekly backtest IC distribution exists, so this is a point "
        "comparison, not a percentile test).",
        "",
        "## 2. Execution fidelity — signal book vs MT5 book (end of signal day)",
        "",
        fid.to_markdown(index=False),
        "",
        "## 3. PnL — live vs paper",
        "",
        f"- Live closed-trade PnL: **{live_pnl:+.2f} USD** on {START_BALANCE:.0f} "
        f"({live_pnl / START_BALANCE:+.2%}); MT5 reports −77.15 incl. swaps "
        "(≈ −7.5 USD of swap not in the profit column).",
        f"- Of which manual-entry fumbles (opened and closed within minutes): "
        f"{fumble_pnl:+.2f} USD across {len(fumbles)} trades.",
        f"- Paper strategy (signal followed exactly, h=5 windows, 1/6 equal "
        f"weight, {COST_BPS:.0f} bps/side): **{paper_total:+.2%}** cumulative "
        "simple sum of weekly net returns.",
        "",
        "## Honesty notes",
        "",
        "- ~10 weekly observations: portfolio Sharpe/return over this window is "
        "statistically uninformative (SE of annualized Sharpe ≈ ±2.2). It is "
        "deliberately not reported. The IC row count (15 pairs × weeks) is the "
        "only metric here with any power.",
        "- MT5 history was transcribed from a screenshot; `profit` values are "
        "as-displayed, three close prices were unreadable.",
        "- Research tooling — not investment advice.",
    ]
    out = REPORT_DIR / f"live_report_{today}.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nReport saved → {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
