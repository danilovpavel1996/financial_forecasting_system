"""Live Signal page — today's commodity ranking with real-time data."""
from __future__ import annotations

import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import load_config
from src.data import universe
from dashboard.charts import _COLOURS, _LAYOUT

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Live Signal",
    page_icon="🚦",
    layout="wide",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📊 Forecast Research")
    st.caption("Daily commodity ranking system")
    st.divider()
    horizon = st.select_slider("Horizon (trading days)", [1, 5, 10, 21], value=5)
    model_choice = st.radio("Model", ["MeanReversion", "LightGBM"], index=0)
    force_retrain = st.checkbox("Force retrain", value=False,
                                help="Retrain LightGBM even if model is fresh")
    force_refresh = st.checkbox("Force data refresh", value=False,
                                help="Re-fetch all data, bypass same-day cache")
    run_btn  = st.button("🚦 Refresh Signal",  type="primary", use_container_width=True)
    train_btn = st.button("🔁 Retrain Model", use_container_width=True,
                          disabled=(model_choice != "LightGBM"))
    st.divider()
    st.caption("⚠️ Research tooling — not investment advice.")

# ── Session state ─────────────────────────────────────────────────────────────

if "live_signal" not in st.session_state:
    st.session_state.live_signal = None

@st.cache_data
def _load_cfg():
    return load_config()

cfg = _load_cfg()

# ── Run signal generation ─────────────────────────────────────────────────────

def _generate(horizon, model_name, retrain, refresh):
    from src.live.data import fetch_live_data
    from src.live.signal import generate_signal
    from src.live.trainer import load_latest_model, model_is_stale, train_live_model

    live_data = fetch_live_data(cfg, force_refresh=refresh, include_cot=False)

    trained_model = None
    if model_name == "LightGBM":
        existing = load_latest_model(cfg)
        needs_train = retrain or model_is_stale(existing, staleness_days=7)
        if needs_train:
            with st.spinner("Training LightGBM on full history…"):
                trained_model = train_live_model(cfg, live_data, horizon=horizon, use_cot=False)
        else:
            trained_model = existing

    return generate_signal(
        cfg=cfg,
        live_data=live_data,
        trained_model=trained_model,
        horizon=horizon,
        model_name=model_name,
        use_cot=False,
        backtest_sharpe=0.63,
        backtest_cs_ric=0.020,
        vol_target=0.10,
    ), live_data

if run_btn or train_btn:
    with st.spinner("Fetching live data and generating signal…"):
        try:
            sig, live_data = _generate(
                horizon, model_choice, force_retrain or train_btn, force_refresh
            )
            st.session_state.live_signal = (sig, live_data)
            st.success("Signal updated!")
        except Exception as exc:
            st.error(f"Signal generation failed: {exc}")
            st.session_state.live_signal = None

# ── Display ───────────────────────────────────────────────────────────────────

st.title("🚦 Live Signal")

if st.session_state.live_signal is None:
    st.info("Click **🚦 Refresh Signal** in the sidebar to generate today's ranking.")
    st.markdown("""
**What this page shows:**
- Today's cross-sectional ranking of the 9 commodities based on the selected model
- LONG = top-2 predicted forward returns, SHORT = bottom-2, FLAT = middle 5
- The MeanReversion model ranks by **negative 5-day trailing return** (mean-reversion heuristic)
- Features use yesterday's closes — same exact code path as the backtest

*Phase 8 out-of-sample Sharpe: 0.63 (h=5, MeanReversion, 2010–2024)*
""")
    st.stop()

sig, live_data = st.session_state.live_signal

# Header metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Signal date", str(sig.date))
col2.metric("Features as of", str(sig.feature_date))
col3.metric("Vol regime", sig.confidence.current_vol_regime.upper())
col4.metric("Vol scale", f"{sig.confidence.vol_scale:.2f}×")

st.divider()

# ── Ranking table ─────────────────────────────────────────────────────────────

st.subheader(f"Ranking — {sig.model_name}  (horizon={sig.horizon}d)")

_POS_COLOUR = {"LONG": "#2A9D8F", "SHORT": "#E76F51", "FLAT": "#888888"}

rows = []
for r in sig.rankings:
    rows.append({
        "Rank": r.rank,
        "Ticker": r.ticker,
        "Name": r.name,
        "Position": r.position,
        "Pred. return": f"{r.predicted_return:+.4f}",
        "Mom 5d": f"{r.recent_mom_5d:+.2%}" if np.isfinite(r.recent_mom_5d) else "N/A",
        "EWMA vol": f"{r.current_vol:.1%}" if np.isfinite(r.current_vol) else "N/A",
    })

tbl_df = pd.DataFrame(rows).set_index("Rank")

def _colour_position(val):
    c = _POS_COLOUR.get(str(val), "")
    return f"color: {c}; font-weight: bold" if c else ""

styled = tbl_df.style.applymap(_colour_position, subset=["Position"])
st.dataframe(styled, use_container_width=True)

# ── Price charts for LONG and SHORT picks ─────────────────────────────────────

picks = [r for r in sig.rankings if r.position in ("LONG", "SHORT")]
if picks and live_data is not None:
    st.divider()
    st.subheader("Price charts — LONG and SHORT picks (last 90 days)")

    ncols = min(len(picks), 4)
    cols = st.columns(ncols)
    for i, r in enumerate(picks):
        with cols[i % ncols]:
            ticker = r.ticker
            if ticker not in live_data.prices:
                st.caption(f"{ticker}: no data")
                continue
            close = live_data.prices[ticker]["Close"].dropna().tail(90)
            colour = _POS_COLOUR[r.position]
            fig = go.Figure(go.Scatter(
                x=close.index, y=close.values,
                mode="lines", line=dict(color=colour, width=2),
                hovertemplate="%{x|%Y-%m-%d}<br>$%{y:,.2f}<extra></extra>",
            ))
            fig.update_layout(
                **_LAYOUT,
                title=f"{ticker} ({r.position})",
                showlegend=False,
                height=220,
            )
            st.plotly_chart(fig, use_container_width=True)

# ── Context panel ─────────────────────────────────────────────────────────────

with st.expander("Signal context"):
    c = sig.confidence
    st.markdown(f"""
| Metric | Value |
|--------|-------|
| Backtest Sharpe (out-of-sample) | {c.backtest_sharpe:.2f} |
| Backtest CS-RIC | {c.backtest_cs_ric:+.4f} |
| Vol regime | {c.current_vol_regime} |
| Vol-targeting scale | {c.vol_scale:.2f}× |
| Model staleness | {c.days_since_retrain} day(s) |
| Model | {sig.model_name} |
""")
    st.caption(
        "Vol scale < 1 means current vol is above target; positions would be reduced "
        "if vol-targeting were applied. The signal here shows UNSCALED ranks — the "
        "vol scale is informational only."
    )

# ── Signal history ────────────────────────────────────────────────────────────

st.divider()
st.subheader("Signal history")

signal_dir = cfg.paths.outputs_reports.parent / "signals"
signal_files = sorted(signal_dir.glob("signal_*.json"), reverse=True) if signal_dir.exists() else []

if not signal_files:
    st.info("No saved signals yet. Run the signal and it will be saved automatically.")
else:
    hist_rows = []
    for p in signal_files[:20]:
        try:
            data = json.loads(p.read_text())
            longs  = [r["ticker"] for r in data["rankings"] if r["position"] == "LONG"]
            shorts = [r["ticker"] for r in data["rankings"] if r["position"] == "SHORT"]
            hist_rows.append({
                "Date": data["date"],
                "Model": data["model_name"],
                "Horizon": data["horizon"],
                "LONG": ", ".join(longs),
                "SHORT": ", ".join(shorts),
                "Vol regime": data["confidence"]["current_vol_regime"],
                "Vol scale": f"{data['confidence']['vol_scale']:.2f}×",
            })
        except Exception:
            continue

    if hist_rows:
        st.dataframe(pd.DataFrame(hist_rows).set_index("Date"), use_container_width=True)
