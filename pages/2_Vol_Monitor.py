"""Vol Monitor page — EWMA volatility forecasts and regime signals."""
from __future__ import annotations

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
from src.data.prices import fetch_all_tickers
from src.models.vol_forecast import EWMAVolForecast, realized_vol, vol_forecast_corr

from dashboard.charts import ewma_vol_chart, _COLOURS

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Vol Monitor",
    page_icon="📉",
    layout="wide",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📊 Forecast Research")
    st.caption("Daily commodity ranking system")
    st.divider()
    ewma_lam = st.slider("EWMA lambda", 0.85, 0.99, 0.94, 0.01,
                         help="RiskMetrics standard: 0.94")
    rv_window = st.slider("Realized vol window (days)", 5, 63, 21,
                          help="Rolling std window for realized vol benchmark")
    st.divider()
    st.caption("⚠️ Research tooling — not investment advice.")

# ── Cached loaders ────────────────────────────────────────────────────────────

@st.cache_data
def _load_cfg():
    return load_config()

@st.cache_data
def _load_prices(_cfg):
    return fetch_all_tickers(
        universe.ranked_tickers(_cfg),
        _cfg.dates["start"], _cfg.dates["end"],
        _cfg.paths.data_raw,
    )

# ── Compute vol forecasts ─────────────────────────────────────────────────────

@st.cache_data
def _compute_vols(_cfg, lam: float, window: int):
    """Compute EWMA and realized vol for all ranked tickers."""
    prices = _load_prices(_cfg)
    ranked = universe.ranked_tickers(_cfg)
    model = EWMAVolForecast(lam=lam)
    rows = []
    series = {}
    for ticker in ranked:
        if ticker not in prices:
            continue
        close = prices[ticker]["Close"].dropna()
        log_ret = np.log(close / close.shift(1)).dropna()
        ewma = model.fit_predict(log_ret)
        rv = realized_vol(log_ret, window=window)
        corr = vol_forecast_corr(ewma, log_ret, proxy_window=window)

        # Current regime: where is current vol vs historical?
        current_vol = ewma.dropna().iloc[-1] if ewma.dropna().shape[0] else np.nan
        hist = ewma.dropna()
        regime_pct = float((hist < current_vol).mean()) if len(hist) else np.nan

        series[ticker] = (log_ret, ewma, rv)
        rows.append({
            "Ticker": ticker,
            "EWMA vol (current)": f"{current_vol:.1%}" if np.isfinite(current_vol) else "N/A",
            "Realized vol (current)": f"{rv.dropna().iloc[-1]:.1%}" if rv.dropna().shape[0] else "N/A",
            "Vol regime %ile": f"{regime_pct:.0%}" if np.isfinite(regime_pct) else "N/A",
            "Forecast corr": f"{corr:.3f}" if np.isfinite(corr) else "N/A",
        })
    return rows, series

# ── Main ──────────────────────────────────────────────────────────────────────

st.title("📉 Vol Monitor")

with st.spinner("Computing volatility forecasts…"):
    cfg = _load_cfg()
    ranked = universe.ranked_tickers(cfg)
    try:
        summary_rows, vol_series = _compute_vols(cfg, ewma_lam, rv_window)
    except Exception as e:
        st.error(f"Failed to compute vol: {e}")
        st.stop()

# Summary table
st.markdown("### Current volatility snapshot")
summary_df = pd.DataFrame(summary_rows).set_index("Ticker")
st.dataframe(summary_df, use_container_width=True)
st.caption(
    f"EWMA lambda={ewma_lam}, realized vol window={rv_window}d. "
    "Vol regime = percentile of current EWMA vol vs its full history."
)

st.divider()

# Individual ticker charts
st.markdown("### EWMA vol forecast — by commodity")

ticker_choice = st.selectbox("Select commodity", [t for t in ranked if t in vol_series])

if ticker_choice and ticker_choice in vol_series:
    log_ret, ewma, rv = vol_series[ticker_choice]
    fig = ewma_vol_chart(ticker_choice, log_ret, ewma, rv)
    st.plotly_chart(fig, use_container_width=True)

    # Distribution of current vol relative to history
    hist_vol = ewma.dropna()
    current = hist_vol.iloc[-1]
    c1, c2, c3 = st.columns(3)
    c1.metric("Current EWMA vol", f"{current:.1%}")
    c2.metric("Historical mean", f"{hist_vol.mean():.1%}")
    hist_pct = float((hist_vol < current).mean())
    c3.metric("Regime percentile", f"{hist_pct:.0%}",
              delta=("high" if hist_pct > 0.75 else ("low" if hist_pct < 0.25 else "normal")))

st.divider()

# All-commodities chart in one figure for overview
st.markdown("### EWMA vol — all commodities (overview)")

fig_all = go.Figure()
for i, ticker in enumerate(ranked):
    if ticker not in vol_series:
        continue
    _, ewma, _ = vol_series[ticker]
    fig_all.add_trace(go.Scatter(
        x=ewma.index, y=(ewma * 100).values,
        mode="lines", name=ticker,
        line=dict(color=_COLOURS[i % len(_COLOURS)], width=1.5),
        hovertemplate="%{x|%Y-%m-%d}<br>%{y:.1f}%<extra>" + ticker + "</extra>",
    ))
fig_all.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    title="EWMA 1-day-ahead Annualised Vol — all commodities",
    yaxis_title="Ann. vol (%)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=40, r=20, t=50, b=40),
    xaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.2)"),
    yaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.2)"),
)
st.plotly_chart(fig_all, use_container_width=True)
