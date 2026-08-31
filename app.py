"""Live Forex Monitor — the one page that shows how the weekly system is doing.

Everything here is computed by ``src.live.report``, the same module that writes
``outputs/reports/live_report_*.md``, so this dashboard and that report can
never disagree.

The weekly routine itself runs unattended on Railway (Fridays 15:00 UTC):
rebalance -> execute on MT5 via MetaApi -> fetch history -> write report ->
push artifacts to GitHub. This page just reads what it left behind, so run
`git pull` before opening it.

Usage:
    .venv/bin/streamlit run app.py
"""
from __future__ import annotations

import datetime
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from src.live.report import (  # noqa: E402
    BACKTEST_RIC,
    BACKTEST_SHARPE,
    DEMO_EXPIRES_DEFAULT,
    PAIR_NAMES,
    START_BALANCE,
    build_report,
    demo_expiry,
)

ROOT = Path(__file__).resolve().parent

# Diverging pair (blue <-> red), validated for the light surface: CVD ΔE 23.8,
# normal-vision ΔE 31.6, both >= 3:1 contrast. Sign is also encoded by whether
# the bar sits above or below zero, so colour is never the only cue.
POS_COLOR   = "#2a78d6"
NEG_COLOR   = "#d03b3b"
REF_COLOR   = "#52514e"
GRID_COLOR  = "rgba(128,128,128,0.25)"

st.set_page_config(page_title="Live Forex Monitor", page_icon="📈",
                   layout="wide")


@st.cache_data(ttl=300, show_spinner=False)
def _report():
    return build_report()


def _next_friday_utc() -> datetime.datetime:
    """Next scheduled cron fire (Fridays 15:00 UTC)."""
    now = datetime.datetime.utcnow()
    ahead = (4 - now.weekday()) % 7          # Monday=0 ... Friday=4
    nxt = (now + datetime.timedelta(days=ahead)).replace(
        hour=15, minute=0, second=0, microsecond=0)
    return nxt + datetime.timedelta(days=7) if nxt <= now else nxt


def fmt_usd(v: float | None) -> str:
    return "—" if v is None else f"{v:+,.2f}"


# ── Header ────────────────────────────────────────────────────────────────────

rep = _report()
s = rep.stats

st.title("Live Forex Monitor")
st.caption(
    "Weekly 15-pair forex ranking, traded automatically on a Fusion Markets "
    "**demo** account. Research tooling — not investment advice."
)

with st.sidebar:
    st.header("Data")
    st.write(f"Signals: **{len(rep.signals)}**")
    st.write(f"Latest: **{rep.latest_signal['date'] if rep.latest_signal else '—'}**")
    st.write(f"Account snapshot: **{s.get('snapshot_at') or 'none yet'}**")
    st.divider()
    if st.button("↻ Refresh live account", width="stretch"):
        with st.spinner("Deploying MetaApi terminal and fetching…"):
            r = subprocess.run(
                [sys.executable, "scripts/fetch_mt5_history.py", "--undeploy"],
                cwd=ROOT, capture_output=True, text=True, timeout=600)
        if r.returncode == 0:
            st.cache_data.clear()
            st.success("Updated.")
            st.rerun()
        else:
            st.error("Fetch failed — see details below.")
            st.code((r.stderr or r.stdout)[-1500:])
    st.caption(
        "Pulls balance, equity and floating P&L from MetaApi, then undeploys "
        "the terminal again. Takes ~1 minute and costs a few cents of MetaApi "
        "time. Otherwise these numbers come from the last Friday run."
    )
    st.divider()
    nxt = _next_friday_utc()
    st.write(f"**Next run:** {nxt:%a %d %b, %H:%M} UTC")
    exp = demo_expiry()
    days_left = (exp - datetime.date.today()).days
    if days_left <= 10:
        st.warning(f"Demo account expires in {days_left} days ({exp}). "
                   "Create a new one and update MetaApi + DEMO_EXPIRES.")
    else:
        st.write(f"**Demo expires:** {exp} ({days_left} days)")

# ── Headline numbers ──────────────────────────────────────────────────────────

c1, c2, c3, c4, c5 = st.columns(5)
equity = s.get("equity")
c1.metric("Equity", f"{equity:,.2f}" if equity else "—",
          delta=f"{equity - START_BALANCE:+,.2f} vs start" if equity else None)
c2.metric("Closed P&L", fmt_usd(s["closed_pnl"]),
          delta=f"{s['closed_pnl'] / START_BALANCE:+.2%}")
c3.metric("Floating P&L", fmt_usd(s.get("floating_pnl")),
          help="Unrealized P&L on the positions currently open.")
c4.metric("Mean weekly IC", f"{s['mean_ric']:+.3f}",
          delta=f"{s['mean_ric'] - BACKTEST_RIC:+.3f} vs backtest",
          delta_color="normal",
          help="Cross-sectional rank IC — the only metric with real "
               "statistical power at this sample size.")
c5.metric("Weeks of live data", f"{s['n_weeks']}",
          help="Decision point is roughly 20 weeks.")

# ── Verdict ───────────────────────────────────────────────────────────────────

mean_ric, se = s["mean_ric"], s["se_ric"]
if s["n_weeks"] < 20:
    if mean_ric - 2 * se > 0:
        st.success(
            f"**Signal is beating zero.** Mean weekly IC {mean_ric:+.3f} "
            f"(SE {se:.3f}) is more than two standard errors above zero over "
            f"{s['n_weeks']} weeks."
        )
    elif mean_ric + 2 * se < 0:
        st.error(
            f"**Signal is significantly negative.** Mean weekly IC "
            f"{mean_ric:+.3f} (SE {se:.3f}) over {s['n_weeks']} weeks — worth "
            "investigating for a sign error or a broken feature."
        )
    else:
        st.info(
            f"**No edge demonstrated yet, and none ruled out.** Mean weekly IC "
            f"is {mean_ric:+.3f} with a standard error of {se:.3f}, so it "
            f"cannot be told apart from zero after {s['n_weeks']} weeks. The "
            f"backtest predicted {BACKTEST_RIC:+.3f}. Keep collecting — about "
            f"20 weeks is where this starts to mean something."
        )

# ── Current positions ─────────────────────────────────────────────────────────

st.subheader("Current positions")
pos = rep.positions()
if pos.empty:
    st.write("No open book.")
else:
    st.dataframe(
        pos, hide_index=True, width="stretch",
        column_config={
            "Predicted": st.column_config.NumberColumn(
                "Predicted return", format="%.5f",
                help="Model's forecast 5-day return for this pair."),
            "Open price": st.column_config.NumberColumn(format="%.5f"),
            "Floating P&L": st.column_config.NumberColumn(
                format="%.2f", help="From the last account snapshot."),
        })
    sig = rep.latest_signal
    st.caption(
        f"Book from signal {sig['date']}"
        + (" · reconstructed signal (see notes)" if sig.get("reconstructed") else "")
        + ". Long the top 3 predicted returns, short the bottom 3, 0.01 lots each."
    )

# ── Signal quality ────────────────────────────────────────────────────────────

st.subheader("Signal quality — weekly cross-sectional rank IC")

ic = rep.ic
if not ic.empty:
    colors = [POS_COLOR if v > 0 else NEG_COLOR for v in ic["cs_ric"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=ic["date"], y=ic["cs_ric"], marker_color=colors,
        marker_line_width=0, width=0.62,
        customdata=ic[["active_hit", "paper_ret_net"]],
        hovertemplate=("<b>%{x}</b><br>Rank IC %{y:+.3f}"
                       "<br>Correct directions %{customdata[0]}"
                       "<br>Paper return %{customdata[1]:+.2%}<extra></extra>"),
        name="Weekly IC",
    ))
    fig.add_hline(y=0, line_width=1.5, line_color=REF_COLOR)
    fig.add_hline(y=s["mean_ric"], line_width=2, line_dash="dash",
                  line_color=REF_COLOR,
                  annotation_text=f"live mean {s['mean_ric']:+.3f}",
                  annotation_position="bottom left")
    fig.add_hline(y=BACKTEST_RIC, line_width=2, line_dash="dot",
                  line_color=REF_COLOR,
                  annotation_text=f"backtest {BACKTEST_RIC:+.3f}",
                  annotation_position="top left")
    fig.update_layout(
        height=380, margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False, bargap=0.3,
        yaxis=dict(title="Spearman rank IC", gridcolor=GRID_COLOR,
                   zeroline=False),
        xaxis=dict(title=None, gridcolor="rgba(0,0,0,0)", type="category"),
        hoverlabel=dict(font_size=13),
    )
    st.plotly_chart(fig, width="stretch")

    st.caption(
        f"Each bar is one week: the correlation between the model's predicted "
        f"ranking and what the 15 pairs actually did over the following 5 "
        f"trading days. Above zero means the ranking had information that "
        f"week. **{s['pos_weeks']} of {s['n_weeks']} weeks positive**, mean "
        f"{s['mean_ric']:+.3f} ± {se:.3f} (SE) versus {BACKTEST_RIC:+.3f} in "
        f"the backtest (which also showed Sharpe {BACKTEST_SHARPE:.2f})."
    )

    with st.expander("Week-by-week detail"):
        st.dataframe(
            ic.rename(columns={
                "date": "Signal date", "cs_ric": "Rank IC",
                "n_pairs": "Pairs", "active_hit": "Correct directions",
                "hit_rate": "Hit rate", "paper_ret_net": "Paper return (net)",
                "reconstructed": "Reconstructed",
            }), hide_index=True, width="stretch",
            column_config={
                "Rank IC": st.column_config.NumberColumn(format="%.4f"),
                "Hit rate": st.column_config.ProgressColumn(
                    format="%.0f%%", min_value=0, max_value=1),
                "Paper return (net)": st.column_config.NumberColumn(
                    format="%.2f%%"),
            })

# ── Execution fidelity ────────────────────────────────────────────────────────

st.subheader("Execution fidelity")
fid = rep.fidelity
st.write(
    f"**{s['n_clean_fidelity']} of {s['n_fidelity']}** weeks the account held "
    "exactly what the signal asked for."
)
st.dataframe(
    fid.drop(columns=["ok"]).rename(columns={
        "date": "Signal date", "match": "Match", "missing": "Missing",
        "extra": "Extra", "wrong_dir": "Wrong direction"}),
    hide_index=True, width="stretch")
st.caption(
    "Compares the signal's target book against the positions actually open at "
    "the end of each signal day. The early misses were manual-trading errors; "
    "everything from 2026-08-14 onward is automated."
)

# ── P&L ───────────────────────────────────────────────────────────────────────

st.subheader("Profit & loss")
left, right = st.columns(2)
with left:
    st.markdown(
        f"""
- **Closed trades, both demo accounts:** {s['closed_pnl']:+.2f} USD
  on {START_BALANCE:,.0f} ({s['closed_pnl'] / START_BALANCE:+.2%})
  - Expired account 372709 (Jun 2 – Aug 14): {s['old_pnl']:+.2f} USD
  - Current account 438689 (from Aug 14): {s['new_pnl']:+.2f} USD,
    {s['n_open']} positions open
- **Floating P&L on open positions:** {fmt_usd(s.get('floating_pnl'))} USD
- **Manual-entry fumbles** (opened and closed within minutes):
  {s['fumble_pnl']:+.2f} USD across {s['n_fumbles']} trades
"""
    )
with right:
    st.markdown(
        f"""
- **Paper equivalent** — the signal followed perfectly, 1/6 equal weight,
  3 bps per side: **{s['paper_total']:+.2%}** (sum of weekly net returns)
- The gap between paper and live is execution: the June weeks where only
  the long legs got placed, plus swap costs the backtest never charged
  (≈ −7.5 USD over the first account's life)
- No live Sharpe is shown on purpose: at {s['n_weeks']} weekly observations
  its standard error is roughly ±2.2 annualized, so the number would be
  noise presented as a result
"""
    )

# ── Trade history ─────────────────────────────────────────────────────────────

with st.expander(f"Trade history ({len(rep.trades)} trades)"):
    tdf = pd.DataFrame([{
        "Opened": t["open_time"], "Closed": t["close_time"],
        "Pair": PAIR_NAMES.get(t["symbol"], t["symbol"]),
        "Side": t["side"].upper(), "Profit": t["profit"],
        "Account": t["account"], "Note": t["note"],
    } for t in rep.trades])
    st.dataframe(tdf.sort_values("Opened", ascending=False), hide_index=True,
                 width="stretch",
                 column_config={"Profit": st.column_config.NumberColumn(
                     format="%.2f")})

with st.expander("How to read this / honesty notes"):
    st.markdown(
        f"""
- **Rank IC is the metric that matters here.** Each week contributes 15
  cross-sectional data points (one per pair), where portfolio return
  contributes a single noisy number. That is why IC, not P&L, decides
  whether this model works.
- **{s['n_weeks']} weeks is still a small sample.** The mean IC is
  {s['mean_ric']:+.3f} with SE {se:.3f}; the interval comfortably contains
  zero. Roughly 20 weeks is where the estimate starts to bite.
- **The newest week's IC is provisional.** It uses the price snapshot taken
  during the signal run, before that day's close settles, and shifts slightly
  once the data finalizes.
- **History provenance:** the first demo account's trades were transcribed
  from a screenshot (profits as displayed, three close prices unreadable);
  the current account's history comes from the MetaApi API.
{"- **Reconstructed weeks** (original signal file lost, predictions regenerated later, so IC is approximate): " + ", ".join(s["reconstructed_weeks"]) if s["reconstructed_weeks"] else ""}
- **Not investment advice.** This is a research harness running on a demo
  account, and the point of it is to find out whether the edge is real —
  not to act on it.
"""
    )
