"""Shared data layer for the live forex report.

One source of truth for the live-trading numbers, consumed by both
``scripts/live_report.py`` (writes the markdown report) and ``app.py`` (the
Streamlit dashboard). No Streamlit or printing here — pure data.

What it answers, and why these metrics:
  1. Weekly cross-sectional rank IC — Spearman(predicted, realized h-day
     forward log return) across all 15 pairs each week. With ~10 weeks of
     live data this is the ONLY metric with real statistical power: each
     week contributes 15 cross-sectional points, where portfolio PnL
     contributes one noisy number.
  2. Execution fidelity — reconstructs the MT5 position book at the end of
     each signal day from trade history and diffs it against the signal.
  3. PnL — realized (closed trades) split per demo account, floating P&L
     from the account snapshot, and the paper equivalent of the signal.

Deliberately NOT computed: live Sharpe. At this sample size its standard
error is roughly ±2.2 annualized, so the number would be noise dressed as
a result.
"""
from __future__ import annotations

import csv
import datetime
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT       = Path(__file__).resolve().parents[2]
SIGNAL_DIR = ROOT / "outputs" / "signals"
LIVE_DIR   = ROOT / "data" / "live"
REPORT_DIR = ROOT / "outputs" / "reports"
EXEC_DIR   = ROOT / "outputs" / "executions"
SNAPSHOT   = LIVE_DIR / "account_snapshot.json"
# Trade history: hand-transcribed CSV for the first (expired) demo account,
# plus the MetaApi-fetched CSV for the current one.
MT5_CSVS   = [LIVE_DIR / "mt5_history_2026-08-14.csv",
              LIVE_DIR / "mt5_history_metaapi.csv"]

HORIZON       = 5
COST_BPS      = 3.0        # per side, same as the backtest charges
BACKTEST_RIC  = 0.071      # Phase 20 OOS cross-sectional rank IC (mean)
BACKTEST_SHARPE = 1.32     # Phase 20 OOS Sharpe, for context only
START_BALANCE = 2000.0
OLD_ACCOUNT   = "372709"   # expired demo, history transcribed from a screenshot
NEW_ACCOUNT   = "438689"   # current demo, history via MetaApi
# Fusion demos live 30 days. Override per environment with DEMO_EXPIRES;
# scripts/preflight_check.py reads the same value to decide when to page.
DEMO_EXPIRES_DEFAULT = "2026-09-13"


def demo_expiry() -> datetime.date:
    return datetime.date.fromisoformat(
        os.environ.get("DEMO_EXPIRES", DEMO_EXPIRES_DEFAULT))

PAIR_NAMES = {
    "EURUSD": "EUR/USD", "GBPUSD": "GBP/USD", "USDJPY": "USD/JPY",
    "AUDUSD": "AUD/USD", "USDCAD": "USD/CAD", "USDCHF": "USD/CHF",
    "NZDUSD": "NZD/USD", "EURJPY": "EUR/JPY", "GBPJPY": "GBP/JPY",
    "EURGBP": "EUR/GBP", "AUDJPY": "AUD/JPY", "EURAUD": "EUR/AUD",
    "GBPAUD": "GBP/AUD", "AUDNZD": "AUD/NZD", "CADJPY": "CAD/JPY",
}


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_signals() -> list[dict]:
    """Every weekly forex signal, oldest first."""
    return [json.loads(p.read_text())
            for p in sorted(SIGNAL_DIR.glob("signal_forex_*.json"))]


def load_prices() -> dict[str, pd.Series]:
    """Close series per MT5 symbol, from the most recent live price cache."""
    dirs = sorted(d for d in LIVE_DIR.iterdir()
                  if d.is_dir() and (d / "prices").is_dir())
    if not dirs:
        return {}
    out: dict[str, pd.Series] = {}
    for p in (dirs[-1] / "prices").glob("*_eq_X_*.parquet"):
        out[p.name.split("_eq_X_")[0]] = pd.read_parquet(p)["Close"].dropna()
    return out


def load_mt5_trades() -> list[dict]:
    """All trades across both demo accounts, oldest first."""
    rows = []
    for path in MT5_CSVS:
        if not path.exists():
            continue
        # Which account a trade belongs to cannot be decided from timestamps
        # (the transcribed file is broker-local, MetaApi returns UTC), but the
        # source file settles it.
        account = OLD_ACCOUNT if "2026-08-14" in path.name else NEW_ACCOUNT
        with open(path) as f:
            for row in csv.DictReader(l for l in f if not l.startswith("#")):
                rows.append({
                    "account":    account,
                    "open_time":  pd.Timestamp(row["open_time"]),
                    "symbol":     row["symbol"].upper(),
                    "side":       row["side"],
                    "close_time": (pd.Timestamp(row["close_time"])
                                   if row["close_time"] else None),
                    "profit":     float(row["profit"]) if row["profit"] else None,
                    "note":       row["note"],
                })
    return sorted(rows, key=lambda t: t["open_time"])


def load_account_snapshot() -> dict | None:
    """Balance / equity / floating P&L recorded by the last cron run."""
    if not SNAPSHOT.exists():
        return None
    return json.loads(SNAPSHOT.read_text())


def load_executions() -> list[dict]:
    return [json.loads(p.read_text())
            for p in sorted(EXEC_DIR.glob("execution_*.json"))]


# ── Realized returns ──────────────────────────────────────────────────────────

def fwd_log_return(prices: pd.Series, asof: datetime.date,
                   h: int = HORIZON) -> float | None:
    """h-trading-day forward log return from the last close at or before asof."""
    if prices is None or len(prices) == 0:
        return None
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
    """One row per signal week whose forward returns have realized."""
    rows = []
    for s in sigs:
        asof = datetime.date.fromisoformat(s["feature_date"])
        pred, real, hits, n_act = [], [], 0, 0
        for r in s["rankings"]:
            fr = fwd_log_return(prices.get(r["mt5"]), asof)
            if fr is None:
                continue
            pred.append(r["predicted_return"])
            real.append(fr)
            if r["position"] in ("LONG", "SHORT"):
                n_act += 1
                hits += int((1.0 if r["position"] == "LONG" else -1.0) * fr > 0)
        if len(pred) < 10:              # week not realizable yet
            continue
        # Paper portfolio: equal-weight 1/6 of the active book, charged both sides.
        act = [(r["mt5"], 1.0 if r["position"] == "LONG" else -1.0)
               for r in s["rankings"] if r["position"] in ("LONG", "SHORT")]
        paper = float(np.mean([sgn * fwd_log_return(prices[t], asof)
                               for t, sgn in act])) - 2 * COST_BPS / 1e4
        rows.append({
            "date": s["date"],
            "cs_ric": spearmanr(pred, real).statistic,
            "n_pairs": len(pred),
            "active_hit": f"{hits}/{n_act}",
            "hit_rate": hits / n_act if n_act else np.nan,
            "paper_ret_net": paper,
            "reconstructed": bool(s.get("reconstructed")),
        })
    return pd.DataFrame(rows)


# ── 2. Execution fidelity ─────────────────────────────────────────────────────

def book_at(trades: list[dict], t: pd.Timestamp) -> dict[str, str]:
    """Open positions at instant t, as {symbol: 'buy'|'sell'}."""
    return {tr["symbol"]: tr["side"] for tr in trades
            if tr["open_time"] <= t
            and (tr["close_time"] is None or tr["close_time"] > t)}


def fidelity_table(sigs: list[dict], trades: list[dict]) -> pd.DataFrame:
    """Did the account actually hold what each signal asked for?"""
    rows = []
    for s in sigs:
        d = datetime.date.fromisoformat(s["date"])
        target = {r["mt5"]: ("buy" if r["position"] == "LONG" else "sell")
                  for r in s["rankings"] if r["position"] in ("LONG", "SHORT")}
        actual = book_at(trades,
                         pd.Timestamp(d) + pd.Timedelta(hours=23, minutes=59))
        missing = sorted(t for t in target if t not in actual)
        extra   = sorted(t for t in actual if t not in target)
        wrong   = sorted(t for t in target
                         if t in actual and target[t] != actual[t])
        rows.append({
            "date": s["date"],
            "ok": not (missing or extra or wrong),
            "match": "✓" if not (missing or extra or wrong) else "✗",
            "missing": ",".join(missing) or "—",
            "extra": ",".join(extra) or "—",
            "wrong_dir": ",".join(wrong) or "—",
        })
    return pd.DataFrame(rows)


# ── Assembled report ──────────────────────────────────────────────────────────

@dataclass
class LiveReport:
    signals: list[dict]
    trades: list[dict]
    ic: pd.DataFrame
    fidelity: pd.DataFrame
    snapshot: dict | None
    stats: dict = field(default_factory=dict)

    @property
    def latest_signal(self) -> dict | None:
        return self.signals[-1] if self.signals else None

    def positions(self) -> pd.DataFrame:
        """The current target book with live per-position P&L when available."""
        sig = self.latest_signal
        if sig is None:
            return pd.DataFrame()
        live = {p["symbol"]: p for p in (self.snapshot or {}).get("positions", [])}
        rows = []
        for r in sig["rankings"]:
            if r["position"] not in ("LONG", "SHORT"):
                continue
            p = live.get(r["mt5"], {})
            rows.append({
                "Pair": PAIR_NAMES.get(r["mt5"], r["mt5"]),
                "Symbol": r["mt5"],
                "Side": "BUY" if r["position"] == "LONG" else "SELL",
                "Predicted": r["predicted_return"],
                "Open price": p.get("openPrice"),
                "Floating P&L": p.get("profit"),
            })
        return pd.DataFrame(rows)


def build_report() -> LiveReport:
    """Load everything from disk and compute the full picture."""
    sigs     = load_signals()
    prices   = load_prices()
    trades   = load_mt5_trades()
    snapshot = load_account_snapshot()

    ic  = ic_table(sigs, prices)
    fid = fidelity_table(sigs, trades)

    closed  = [t for t in trades if t["profit"] is not None]
    fumbles = [t for t in closed if "fumble" in t["note"]]
    n_weeks = len(ic)

    stats = {
        "n_weeks":     n_weeks,
        "window_from": ic["date"].iloc[0] if n_weeks else None,
        "window_to":   ic["date"].iloc[-1] if n_weeks else None,
        "mean_ric":    float(ic["cs_ric"].mean()) if n_weeks else float("nan"),
        "se_ric":      (float(ic["cs_ric"].std(ddof=1) / math.sqrt(n_weeks))
                        if n_weeks > 1 else float("nan")),
        "pos_weeks":   int((ic["cs_ric"] > 0).sum()) if n_weeks else 0,
        "closed_pnl":  sum(t["profit"] for t in closed),
        "old_pnl":     sum(t["profit"] for t in closed
                           if t["account"] == OLD_ACCOUNT),
        "new_pnl":     sum(t["profit"] for t in closed
                           if t["account"] == NEW_ACCOUNT),
        "fumble_pnl":  sum(t["profit"] for t in fumbles),
        "n_fumbles":   len(fumbles),
        "n_open":      sum(1 for t in trades if t["close_time"] is None
                           and t["account"] == NEW_ACCOUNT),
        "paper_total": float(ic["paper_ret_net"].sum()) if n_weeks else 0.0,
        "reconstructed_weeks": [s["date"] for s in sigs if s.get("reconstructed")],
        "n_clean_fidelity": int(fid["ok"].sum()) if len(fid) else 0,
        "n_fidelity": len(fid),
    }
    stats["floating_pnl"] = (snapshot or {}).get("floating_pnl")
    stats["balance"]      = (snapshot or {}).get("balance")
    stats["equity"]       = (snapshot or {}).get("equity")
    stats["snapshot_at"]  = (snapshot or {}).get("fetched_at")

    return LiveReport(signals=sigs, trades=trades, ic=ic, fidelity=fid,
                      snapshot=snapshot, stats=stats)
