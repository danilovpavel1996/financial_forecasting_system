"""Live Forex Monitor — what the weekly trading system has done to the money.

Built to be read once a week without remembering how any of it works: the
wallet first, the reasoning underneath. Every number comes from
``src.live.report``, the same module that writes
``outputs/reports/live_report_*.md``, so page and report cannot disagree.

The routine itself runs unattended on Railway (Fridays 15:00 UTC) and pushes
its results to GitHub, so run `git pull` before opening this.

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
from plotly.subplots import make_subplots

load_dotenv()

from src.live.report import (  # noqa: E402
    BACKTEST_RIC,
    BACKTEST_SHARPE,
    COST_BPS,
    PAIR_NAMES,
    START_BALANCE,
    build_report,
    demo_expiry,
    project_wallet,
)

ROOT = Path(__file__).resolve().parent

# Diverging pair, validated against BOTH the light and dark chart surfaces:
# CVD ΔE 25.7, normal-vision ΔE 31.9, contrast >= 3:1 in each mode. Gains are
# blue and losses red rather than the usual green/red precisely because
# green/red is the pair red-green colourblind readers cannot separate. Every
# figure also carries a sign or an arrow, so colour is never the only cue.
GAIN   = "#3987e5"
LOSS   = "#d03b3b"
MUTED  = "#8a8a86"          # neutral ink: readable on either surface
GRID   = "rgba(128,128,128,0.22)"
GAIN_F = "rgba(57,135,229,0.14)"
LOSS_F = "rgba(208,59,59,0.14)"
HANDOVER = "2026-08-14"     # first demo account retired

st.set_page_config(page_title="Live Forex Monitor", page_icon="📈",
                   layout="wide")

st.markdown("""
<style>
  .hero-label { font-size: 0.95rem; color: #8a8a86; margin-bottom: -0.4rem; }
  .hero-value { font-size: 3.6rem; font-weight: 700; line-height: 1.1; }
  .hero-sub   { font-size: 1.05rem; color: #8a8a86; }
  .plain      { font-size: 1.05rem; line-height: 1.6; }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=300, show_spinner=False)
def _report(_report_module_mtime: float):
    # The mtime is part of the cache key and unused in the body: editing
    # src/live/report.py invalidates the cache, so a long-running session can
    # never serve a report built by older code.
    return build_report()


def get_report():
    return _report((ROOT / "src" / "live" / "report.py").stat().st_mtime)


def _next_friday_utc() -> datetime.datetime:
    now = datetime.datetime.utcnow()
    ahead = (4 - now.weekday()) % 7                 # Monday=0 … Friday=4
    nxt = (now + datetime.timedelta(days=ahead)).replace(
        hour=15, minute=0, second=0, microsecond=0)
    return nxt + datetime.timedelta(days=7) if nxt <= now else nxt


def money(v: float | None, signed: bool = True) -> str:
    if v is None:
        return "—"
    return f"{v:+,.2f}" if signed else f"{v:,.2f}"


def arrow(v: float) -> str:
    return "▲" if v > 0 else ("▼" if v < 0 else "■")


def base_layout(fig: go.Figure, height: int, ytitle: str | None = None):
    """Recessive axes, transparent surface, no legend box for single series."""
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=14, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False, hoverlabel=dict(font_size=13),
        yaxis=dict(title=dict(text=ytitle, font=dict(color=MUTED)),
                   gridcolor=GRID, zeroline=False,
                   tickfont=dict(color=MUTED)),
        xaxis=dict(title=None, gridcolor="rgba(0,0,0,0)",
                   tickfont=dict(color=MUTED)),
    )
    return fig


rep = get_report()
s = rep.stats

# ── Sidebar: plumbing, kept out of the way ───────────────────────────────────

with st.sidebar:
    st.header("Status")
    st.write(f"**Next trade:** {_next_friday_utc():%a %d %b}, 15:00 UTC")
    st.write(f"Last signal: **{rep.latest_signal['date'] if rep.latest_signal else '—'}**")
    st.caption(f"Prices through {s.get('prices_through') or '—'} · "
               f"account checked {(s.get('snapshot_at') or '—')[:16].replace('T', ' ')}")

    exp = demo_expiry()
    days_left = (exp - datetime.date.today()).days
    if days_left <= 10:
        st.warning(f"Demo account expires in {days_left} days ({exp}). "
                   "Create a new one, then update MetaApi and DEMO_EXPIRES.")

    st.divider()
    if st.button("↻ Refresh market data", width="stretch"):
        with st.spinner("Downloading forex closes…"):
            from src.live.report import refresh_prices
            refresh_prices()
        st.cache_data.clear()
        st.rerun()
    if st.button("↻ Refresh live account", width="stretch"):
        with st.spinner("Asking the broker…"):
            r = subprocess.run(
                [sys.executable, "scripts/fetch_mt5_history.py", "--undeploy"],
                cwd=ROOT, capture_output=True, text=True, timeout=600)
        if r.returncode == 0:
            st.cache_data.clear()
            st.rerun()
        else:
            st.error("Fetch failed.")
            st.code((r.stderr or r.stdout)[-1200:])
    st.caption(
        "Both buttons are optional — the Friday run already refreshes "
        "everything. Use them only to see numbers from right now."
    )

if s.get("missing_history"):
    st.error(
        f"**Trade history missing: {', '.join(s['missing_history'])}.** "
        "Profit figures and the execution table below are wrong for the "
        "affected period. Restore the file before trusting this page."
    )

# ── 1. The money, in one glance ──────────────────────────────────────────────

st.title("Live Forex Monitor")
st.caption("An automated trading experiment on a **practice** account — real "
           "prices, pretend money. Research tooling, not investment advice.")

wallet, total = s["wallet_value"], s["total_pnl"]
weekly = rep.weekly_pnl()
this_week = float(weekly.iloc[-1]) if len(weekly) else 0.0

hero, side = st.columns([1.15, 2])
with hero:
    st.markdown(
        f"<div class='hero-label'>Your practice wallet is worth</div>"
        f"<div class='hero-value'>${wallet:,.0f}</div>"
        f"<div class='hero-sub'>{arrow(total)} {money(total)} since you "
        f"started with ${START_BALANCE:,.0f}</div>",
        unsafe_allow_html=True)
with side:
    a, b, c, d = st.columns(4)
    a.metric("This week", money(this_week),
             help="Profit or loss realized when this week's trades closed.")
    b.metric("Trades", f"{s['n_closed_trades']}",
             help="Completed round trips since 2 June 2026.")
    c.metric("Win rate", f"{s['win_rate']:.0%}",
             help=f"{s['n_winners']} of {s['n_closed_trades']} closed at a profit.")
    d.metric("Weeks", f"{s['n_weeks']}",
             help="Weeks of completed trading that can be scored.")

# Plain-language read of what the numbers mean, so the page explains itself.
avg_win, avg_loss = s["avg_win"], s["avg_loss"]
st.markdown(
    f"<div class='plain'><b>In plain terms.</b> Out of "
    f"<b>{s['n_closed_trades']}</b> finished trades, <b>{s['n_winners']}</b> "
    f"made money — a {s['win_rate']:.0%} hit rate, slightly better than a coin "
    f"flip. The wallet is still down because the wins are smaller than the "
    f"losses: an average winner makes <b>{money(avg_win)}</b> while an average "
    f"loser costs <b>{money(avg_loss)}</b>. Winning more often than not is not "
    f"enough when the losses are the bigger ones.</div>",
    unsafe_allow_html=True)
st.write("")

# ── 2. Wallet over time ──────────────────────────────────────────────────────

st.subheader("How the wallet got here")

pnl = rep.pnl
if pnl.empty:
    st.write("No closed trades yet.")
else:
    grain = st.radio("Granularity", ["Weekly", "Daily"], horizontal=True,
                     label_visibility="collapsed")
    series = (pnl if grain == "Daily"
              else pnl.resample("W-FRI").last().ffill())
    wallet_curve = START_BALANCE + series["TOTAL"]

    fig = go.Figure()
    # Shade between the wallet and the starting balance, not down to zero, so
    # the filled area reads as "how far ahead or behind you are".
    fig.add_trace(go.Scatter(
        x=wallet_curve.index, y=[START_BALANCE] * len(wallet_curve),
        mode="lines", line=dict(width=0), hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=wallet_curve.index, y=wallet_curve, mode="lines",
        line=dict(color=GAIN if total >= 0 else LOSS, width=2, shape="hv"),
        fill="tonexty", fillcolor=GAIN_F if total >= 0 else LOSS_F,
        hovertemplate="<b>%{x|%d %b %Y}</b><br>Wallet $%{y:,.2f}<extra></extra>",
    ))
    fig.add_hline(y=START_BALANCE, line_width=1.5, line_color=MUTED)
    fig.add_annotation(x=wallet_curve.index[0], y=START_BALANCE, xref="x",
                       yref="y", text=f"started at ${START_BALANCE:,.0f}",
                       showarrow=False, xanchor="left", yanchor="bottom",
                       font=dict(size=11, color=MUTED))
    base_layout(fig, 340, "Wallet value (USD)")
    # Keep the zero-profit line visible instead of letting autoscale hide it.
    lo = min(float(wallet_curve.min()), START_BALANCE)
    hi = max(float(wallet_curve.max()), START_BALANCE)
    pad = max(8.0, 0.08 * (hi - lo))
    fig.update_yaxes(range=[lo - pad, hi + pad])
    st.plotly_chart(fig, width="stretch")
    st.caption(
        f"Each step is money actually banked when a trade closed — the line "
        f"only moves on closing days. The flat line is your ${START_BALANCE:,.0f} "
        f"starting point: below it you are down, above it you are up. "
        f"{money(s.get('floating_pnl'))} is currently floating on the six open "
        "trades and is not counted until they close."
    )

# ── 3. Where it could go ─────────────────────────────────────────────────────

st.subheader("Where it could go from here")

proj = project_wallet(weekly, wallet, weeks=12)
if proj.empty:
    st.write("Not enough weeks yet to show a range.")
else:
    today = pd.Timestamp(datetime.date.today())
    proj["date"] = [today + pd.Timedelta(weeks=int(w)) for w in proj["week"]]
    fan = go.Figure()
    fan.add_trace(go.Scatter(x=proj["date"], y=proj["p90"], mode="lines",
                             line=dict(width=0), hoverinfo="skip"))
    fan.add_trace(go.Scatter(
        x=proj["date"], y=proj["p10"], mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor="rgba(128,128,128,0.28)", hoverinfo="skip"))
    fan.add_trace(go.Scatter(
        x=proj["date"], y=proj["p50"], mode="lines",
        line=dict(color=MUTED, width=2, dash="dash"),
        hovertemplate="<b>%{x|%d %b}</b><br>Middle outcome $%{y:,.0f}"
                      "<extra></extra>"))
    fan.add_hline(y=START_BALANCE, line_width=1.5, line_color=MUTED)
    fan.add_annotation(x=proj["date"].iloc[0], y=START_BALANCE, xref="x",
                       yref="y", text=f"${START_BALANCE:,.0f} break-even",
                       showarrow=False, xanchor="left", yanchor="bottom",
                       font=dict(size=11, color=MUTED))
    base_layout(fan, 300, "Wallet value (USD)")
    # Autoscale clips the top of the band against the break-even line; pad so
    # the whole fan and that line are always visible.
    lo = min(float(proj["p10"].min()), START_BALANCE)
    hi = max(float(proj["p90"].max()), START_BALANCE)
    pad = max(10.0, 0.07 * (hi - lo))
    fan.update_yaxes(range=[lo - pad, hi + pad])
    st.plotly_chart(fan, width="stretch")

    p10, p50, p90 = (proj["p10"].iloc[-1], proj["p50"].iloc[-1],
                     proj["p90"].iloc[-1])
    st.markdown(
        f"<div class='plain'><b>Read this as a spread, not a prediction.</b> "
        f"It shuffles the {len(weekly)} weeks already traded and deals them out "
        f"12 more times. In 8 of 10 of those shuffles the wallet lands between "
        f"<b>${p10:,.0f}</b> and <b>${p90:,.0f}</b>, with the middle around "
        f"<b>${p50:,.0f}</b>. The spread is far bigger than the drift, which is "
        f"the real message: at this sample size the week-to-week noise swamps "
        f"any trend, and nothing here says the system will make money. If the "
        f"model has no edge, the middle line drifts down by roughly the trading "
        f"costs.</div>",
        unsafe_allow_html=True)
st.write("")

# ── 4. Which pairs made and lost the money ───────────────────────────────────

st.subheader("Which currency pairs made or lost the money")

pair_stats = rep.per_pair_stats()
pairs = rep.per_pair_pnl()
if pairs.empty:
    st.write("No closed trades yet.")
else:
    labels = [PAIR_NAMES.get(k, k) for k in pairs.index]
    counts = pair_stats.loc[pairs.index, "n_trades"]
    wins = pair_stats.loc[pairs.index, "n_wins"]
    bar = go.Figure(go.Bar(
        x=pairs.values, y=labels, orientation="h",
        marker_color=[GAIN if v > 0 else LOSS for v in pairs.values],
        marker_line_width=0,
        text=[f"{v:+,.0f}" for v in pairs.values], textposition="outside",
        textfont=dict(color=MUTED),
        customdata=list(zip(counts, wins)),
        hovertemplate="<b>%{y}</b><br>%{x:+,.2f} USD over %{customdata[0]} "
                      "trades<br>%{customdata[1]} of them profitable"
                      "<extra></extra>",
    ))
    bar.add_vline(x=0, line_width=1.5, line_color=MUTED)
    base_layout(bar, 30 * len(pairs) + 110, None)
    # A pair's total reads very differently over 2 trades than over 8, so the
    # count sits beside every bar as its own right-hand column.
    for lbl, n, w in zip(labels, counts, wins):
        bar.add_annotation(xref="paper", x=1.0, xanchor="left", y=lbl,
                           yref="y",
                           text=f"{n} trade{'s' if n != 1 else ''} · {w} won",
                           showarrow=False, font=dict(size=11, color=MUTED))
    bar.add_annotation(xref="paper", x=1.0, xanchor="left", yref="paper",
                       y=1.0, yanchor="bottom", text="<b>from</b>",
                       showarrow=False, font=dict(size=11, color=MUTED))
    # Leave room inside the plot for the outside value labels, otherwise the
    # longest bar's label runs into the trade-count column.
    span = max(abs(float(pairs.min())), abs(float(pairs.max())))
    bar.update_xaxes(range=[float(pairs.min()) - 0.18 * span,
                            float(pairs.max()) + 0.18 * span])
    bar.update_layout(
        bargap=0.35, margin=dict(l=10, r=140, t=28, b=10),
        xaxis=dict(title=dict(text="Realized P&L (USD)",
                              font=dict(color=MUTED)),
                   gridcolor=GRID, zeroline=False,
                   tickfont=dict(color=MUTED)),
        yaxis=dict(gridcolor="rgba(0,0,0,0)", tickfont=dict(color=MUTED)))
    st.plotly_chart(bar, width="stretch")

    worst_sym, worst_val = s["worst_trade"]
    worst_pair = PAIR_NAMES.get(worst_sym, worst_sym)
    worst_n = int(pair_stats.loc[worst_sym, "n_trades"]) if worst_sym in pair_stats.index else 0
    st.caption(
        f"Money banked per pair across every account, with the number of "
        f"trades behind each figure. That count matters: "
        f"{worst_pair} is the worst line on the chart, and it came from "
        f"{'a single trade' if worst_n == 1 else f'just {worst_n} trades'} "
        f"losing {money(worst_val)} — one bad week, not a pattern. "
        f"At one to seven trades per pair, none of these differences is "
        "evidence that the model is good or bad at particular currencies."
    )

    with st.expander("Each pair's running total over time"):
        ncols = 4
        nrows = -(-len(pairs) // ncols)
        facets = make_subplots(
            rows=nrows, cols=ncols, shared_yaxes=True,
            subplot_titles=[f"{PAIR_NAMES.get(k, k)}  {v:+.1f}"
                            for k, v in pairs.items()],
            vertical_spacing=0.12, horizontal_spacing=0.04)
        for i, (sym, final) in enumerate(pairs.items()):
            facets.add_trace(
                go.Scatter(x=series.index, y=series[sym], mode="lines",
                           line=dict(color=GAIN if final > 0 else LOSS,
                                     width=2, shape="hv"),
                           hovertemplate=(f"<b>{sym}</b><br>%{{x|%d %b}}"
                                          "<br>%{y:+.2f} USD<extra></extra>"),
                           showlegend=False),
                row=i // ncols + 1, col=i % ncols + 1)
        # shared_yaxes only links within a row; pin one range across all panels
        # so the facets are genuinely comparable.
        lo = float(series[pairs.index].min().min())
        hi = float(series[pairs.index].max().max())
        pad = 0.08 * (hi - lo)
        facets.update_layout(height=190 * nrows,
                             margin=dict(l=10, r=10, t=30, b=10),
                             paper_bgcolor="rgba(0,0,0,0)",
                             plot_bgcolor="rgba(0,0,0,0)")
        facets.update_xaxes(showticklabels=False, gridcolor="rgba(0,0,0,0)")
        facets.update_yaxes(range=[lo - pad, hi + pad], gridcolor=GRID,
                            zeroline=True, zerolinecolor=MUTED,
                            zerolinewidth=1, tickfont=dict(color=MUTED))
        for ann in facets.layout.annotations:
            ann.font.size = 12
            ann.font.color = MUTED
        st.plotly_chart(facets, width="stretch")

st.divider()

# ── 5. Under the hood ────────────────────────────────────────────────────────

st.subheader("Under the hood")
st.caption("The reasoning behind the numbers above. You only need this when a "
           "result looks surprising and you want to know why.")

t_signal, t_trades, t_exec, t_data = st.tabs(
    ["Is the model any good?", "This week's trades", "Did it trade correctly?",
     "Data & caveats"])

with t_signal:
    mean_ric, se = s["mean_ric"], s["se_ric"]
    if mean_ric - 2 * se > 0:
        st.success(f"**The model is beating chance.** Its weekly skill score "
                   f"averages {mean_ric:+.3f} (± {se:.3f}), clear of zero over "
                   f"{s['n_weeks']} weeks.")
    elif mean_ric + 2 * se < 0:
        st.error(f"**The model is reliably wrong**, averaging {mean_ric:+.3f} "
                 f"(± {se:.3f}) over {s['n_weeks']} weeks — worth checking for "
                 "a sign error or a broken input.")
    else:
        st.info(
            f"**Still too early to say.** The weekly skill score averages "
            f"{mean_ric:+.3f}, give or take {se:.3f} — a range that comfortably "
            f"includes zero, so after {s['n_weeks']} weeks we cannot tell "
            f"whether the model has any skill at all. Testing on 20 years of "
            f"history suggested {BACKTEST_RIC:+.3f}. About 20 live weeks is "
            "where this starts to carry weight."
        )

    st.markdown(
        "**What the score means.** Each Friday the model ranks all 15 currency "
        "pairs from most to least promising. A week scores **+1** if the "
        "ranking turned out perfect, **0** if it was no better than guessing, "
        "and **−1** if it was exactly backwards. One week tells you nothing; "
        "the average over many weeks is the thing to watch."
    )

    ic = rep.ic
    if not ic.empty:
        fig = go.Figure(go.Bar(
            x=ic["date"], y=ic["cs_ric"],
            marker_color=[GAIN if v > 0 else LOSS for v in ic["cs_ric"]],
            marker_line_width=0, width=0.62,
            customdata=ic[["active_hit", "paper_ret_net"]],
            hovertemplate=("<b>%{x}</b><br>Skill score %{y:+.3f}"
                           "<br>Right direction on %{customdata[0]} trades"
                           "<extra></extra>")))
        fig.add_hline(y=0, line_width=1.5, line_color=MUTED)
        fig.add_hline(y=mean_ric, line_width=2, line_dash="dash",
                      line_color=MUTED,
                      annotation_text=f"live average {mean_ric:+.3f}",
                      annotation_position="bottom left")
        fig.add_hline(y=BACKTEST_RIC, line_width=2, line_dash="dot",
                      line_color=MUTED,
                      annotation_text=f"what history suggested {BACKTEST_RIC:+.3f}",
                      annotation_position="top left")
        base_layout(fig, 360, "Weekly skill score")
        fig.update_layout(bargap=0.3, xaxis=dict(type="category",
                                                 tickfont=dict(color=MUTED)))
        st.plotly_chart(fig, width="stretch")
        st.caption(
            f"**{s['pos_weeks']} of {s['n_weeks']} weeks scored above zero.** "
            f"Backtesting on 2005–2024 also suggested a Sharpe ratio of "
            f"{BACKTEST_SHARPE:.2f}; the live sample is nowhere near long "
            "enough to confirm or deny that."
        )
        if s.get("unscored_signals"):
            st.caption(
                f"Not yet scored: {', '.join(s['unscored_signals'])} — a week "
                "needs five trading days of prices after it, and local prices "
                f"reach {s.get('prices_through') or 'nowhere'}."
            )
        with st.expander("Week-by-week numbers"):
            st.dataframe(
                ic.rename(columns={
                    "date": "Week", "cs_ric": "Skill score", "n_pairs": "Pairs",
                    "active_hit": "Right direction", "hit_rate": "Hit rate",
                    "paper_ret_net": "Return if followed exactly",
                    "reconstructed": "Reconstructed"}),
                hide_index=True, width="stretch",
                column_config={
                    "Skill score": st.column_config.NumberColumn(format="%.3f"),
                    "Hit rate": st.column_config.ProgressColumn(
                        format="%.0f%%", min_value=0, max_value=1),
                    "Return if followed exactly":
                        st.column_config.NumberColumn(format="%.2f%%")})

with t_trades:
    pos = rep.positions()
    if pos.empty:
        st.write("No open trades.")
    else:
        st.markdown(
            f"These six trades were opened on **{rep.latest_signal['date']}** "
            "and will be closed and replaced on the next run. The model buys "
            "the three pairs it rates highest and sells the three it rates "
            "lowest, 0.01 lots each — a deliberately tiny, equal stake on every "
            "one."
        )
        st.dataframe(
            pos, hide_index=True, width="stretch",
            column_config={
                "Predicted": st.column_config.NumberColumn(
                    "Model's forecast", format="%.5f",
                    help="Expected 5-day move. Positive → buy, negative → sell."),
                "Open price": st.column_config.NumberColumn(format="%.5f"),
                "Floating P&L": st.column_config.NumberColumn(
                    "Unrealized P&L", format="%.2f",
                    help="Profit if the trade were closed right now.")})
        st.caption(f"Unrealized total: {money(s.get('floating_pnl'))} USD. "
                   "This is not banked until the trades close.")

    with st.expander(f"Every trade ever made ({len(rep.trades)})"):
        tdf = pd.DataFrame([{
            "Opened": t["open_time"], "Closed": t["close_time"],
            "Pair": PAIR_NAMES.get(t["symbol"], t["symbol"]),
            "Direction": "Buy" if t["side"] == "buy" else "Sell",
            "Profit": t["profit"], "Account": t["account"], "Note": t["note"],
        } for t in rep.trades])
        st.dataframe(tdf.sort_values("Opened", ascending=False),
                     hide_index=True, width="stretch",
                     column_config={"Profit": st.column_config.NumberColumn(
                         format="%.2f")})

with t_exec:
    st.markdown(
        f"**{s['n_clean_fidelity']} of {s['n_fidelity']} weeks** the account "
        "held exactly the trades the model asked for. This checks the robot "
        "against its own instructions — a mismatch means the trading failed, "
        "not the model."
    )
    st.dataframe(
        rep.fidelity.drop(columns=["ok"]).rename(columns={
            "date": "Week", "match": "Matched", "missing": "Never opened",
            "extra": "Shouldn't be there", "wrong_dir": "Traded backwards"}),
        hide_index=True, width="stretch")
    st.caption(
        "The early misses were from placing trades by hand; everything from "
        "14 August onward has been automated and clean."
    )

    st.markdown("**Costs you are paying that the model never charged itself**")
    st.markdown(
        f"- Trading on real spreads, the strategy run perfectly on paper would "
        f"have returned **{s['paper_total']:+.2%}**, against **"
        f"{s['closed_pnl'] / START_BALANCE:+.2%}** actually realized.\n"
        f"- The gap is execution: the June weeks when only half the trades got "
        f"placed, plus overnight financing (≈ −7.5 USD on the first account) "
        f"that the backtest charges nothing for. The backtest does charge "
        f"{COST_BPS:.0f} bps of spread per side."
    )

with t_data:
    st.markdown(
        f"""
**Where the numbers come from.** Every Friday at 15:00 UTC a server retrains
the model, picks the six trades, places them on the practice account, records
what happened, and pushes the files to GitHub. This page reads those files —
so `git pull` first, or you are reading last week's.

**The wallet figure.** Practice accounts get replaced every so often and each
replacement starts with a fresh balance, so the headline wallet is your
original ${START_BALANCE:,.0f} plus every trade banked since, plus what is
currently floating. It deliberately ignores the balance resets.
"""
    )
    rows = [{"Account": a["login"],
             "Status": "Live" if a["active"] else "Retired",
             "Opened with": a["start_balance"],
             "Closed P&L": a["closed_pnl"],
             "Trades": a["n_trades"],
             "Still open": a["n_open"],
             "History from": a["source"]} for a in s["per_account"]]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                 column_config={
                     "Opened with": st.column_config.NumberColumn(format="%.2f"),
                     "Closed P&L": st.column_config.NumberColumn(format="%.2f")})

    st.markdown(
        f"""
**Things that would make these numbers wrong, stated plainly**

- **{s['n_weeks']} weeks is a small sample.** No profit or loss figure here is
  statistically meaningful yet. A live Sharpe ratio is deliberately not shown:
  at this many weeks its margin of error is about ±2.2, so the number would be
  noise dressed up as a result.
- **The newest week's score is provisional** — it uses prices captured during
  the Friday run, before that day's close settles, and shifts slightly later.
- **The first account's history was transcribed from a screenshot** (three
  closing prices were unreadable); everything since comes from the broker's API.
- **Positions orphaned by account changes never realize their profit or loss**,
  so they are excluded rather than guessed at.
{"- **Reconstructed weeks** (the original signal file was lost and predictions were regenerated later, so the score is approximate): " + ", ".join(s["reconstructed_weeks"]) if s["reconstructed_weeks"] else ""}
- **This is a practice account.** Nothing here is advice to buy or sell
  anything, and the point of the exercise is to find out whether the edge is
  real — not to act as though it already is.
"""
    )
