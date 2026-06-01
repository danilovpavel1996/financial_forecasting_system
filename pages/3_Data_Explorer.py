"""Data Explorer page — price charts, COT charts, correlations, feature preview."""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import pandas as pd
import streamlit as st

from src.config import load_config
from src.data import universe
from src.data.prices import fetch_all_tickers
from src.data.cot import fetch_all_cot

from dashboard.charts import correlation_heatmap, cot_positioning_chart, price_chart

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Data Explorer",
    page_icon="🔍",
    layout="wide",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📊 Forecast Research")
    st.caption("Daily commodity ranking system")
    st.divider()
    st.caption("⚠️ Research tooling — not investment advice.")

# ── Cached data loaders ───────────────────────────────────────────────────────

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

@st.cache_data
def _load_cot(_cfg):
    cftc = _cfg.cftc
    if not cftc or not cftc.get("codes"):
        return {}
    try:
        return fetch_all_cot(
            codes=cftc["codes"],
            start=_cfg.dates["start"],
            end=_cfg.dates["end"],
            cache_dir=_cfg.paths.data_raw,
            report_type=cftc.get("report_type", "disaggregated_fut"),
        )
    except Exception as e:
        return {}

# ── Main ──────────────────────────────────────────────────────────────────────

st.title("🔍 Data Explorer")

with st.spinner("Loading data…"):
    cfg = _load_cfg()
    ranked = universe.ranked_tickers(cfg)
    try:
        prices = _load_prices(cfg)
    except Exception as e:
        st.error(f"Failed to load prices: {e}")
        st.stop()
    cot_raw = _load_cot(cfg)

# ── Tab layout ────────────────────────────────────────────────────────────────

tab_cov, tab_price, tab_cot, tab_corr = st.tabs([
    "📋 Coverage", "📈 Prices", "📊 COT Positioning", "🔗 Correlations"
])

# ── Tab: Coverage ─────────────────────────────────────────────────────────────

with tab_cov:
    st.markdown("### Data coverage summary")
    rows = []
    for ticker in ranked:
        if ticker not in prices:
            rows.append({"Ticker": ticker, "Rows": 0, "Start": "N/A", "End": "N/A",
                         "COT weeks": 0})
            continue
        df = prices[ticker]
        cot_weeks = len(cot_raw.get(ticker, pd.DataFrame()))
        rows.append({
            "Ticker": ticker,
            "Rows": len(df),
            "Start": str(df.index.min().date()),
            "End": str(df.index.max().date()),
            "COT weeks": cot_weeks,
        })
    st.dataframe(pd.DataFrame(rows).set_index("Ticker"), use_container_width=True)
    st.caption(
        f"Price data: {cfg.dates['start']} → {cfg.dates['end']}. "
        f"COT data: CFTC Disaggregated Futures-Only (release lag = 1 trading day after Friday)."
    )

# ── Tab: Prices ───────────────────────────────────────────────────────────────

with tab_price:
    st.markdown("### Closing prices (log scale)")
    col_ticker, col_log = st.columns([2, 1])
    selected_ticker = col_ticker.selectbox("Commodity", ranked, key="price_ticker")
    log_scale = col_log.checkbox("Log scale", value=True)

    fig = price_chart(prices, selected_ticker, log_scale=log_scale)
    st.plotly_chart(fig, use_container_width=True)

    if selected_ticker in prices:
        df = prices[selected_ticker]
        close = df["Close"].dropna()
        log_ret = np.log(close / close.shift(1)).dropna()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("First date", str(close.index.min().date()))
        c2.metric("Last date", str(close.index.max().date()))
        c3.metric("Ann. vol", f"{log_ret.std() * np.sqrt(252):.1%}")
        c4.metric("Total return", f"{(close.iloc[-1] / close.iloc[0] - 1):.1%}")

# ── Tab: COT ─────────────────────────────────────────────────────────────────

with tab_cot:
    st.markdown("### CFTC Commitment of Traders — Managed Money positioning")
    if not cot_raw:
        st.warning("No COT data loaded. Check that the CFTC cache exists.")
    else:
        cot_ticker = st.selectbox("Commodity", [t for t in ranked if t in cot_raw], key="cot_ticker")
        if cot_ticker:
            fig_cot = cot_positioning_chart(cot_raw[cot_ticker], cot_ticker)
            st.plotly_chart(fig_cot, use_container_width=True)

            w = cot_raw[cot_ticker]
            mm_net = w["mm_long"] - w["mm_short"]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Weeks", len(w))
            c2.metric("Current MM net", f"{mm_net.iloc[-1]:,.0f}")
            c3.metric("52w max", f"{mm_net.tail(52).max():,.0f}")
            c4.metric("52w min", f"{mm_net.tail(52).min():,.0f}")

            st.caption(
                "Red dashed line at 0.80 (crowded long). "
                "Green dashed at 0.20 (crowded short). "
                "The 52-week percentile is the primary contrarian signal."
            )

# ── Tab: Correlations ─────────────────────────────────────────────────────────

with tab_corr:
    st.markdown("### Daily log-return correlations")
    corr_tickers = st.multiselect(
        "Select tickers", ranked, default=ranked,
    )
    if len(corr_tickers) < 2:
        st.info("Select at least 2 tickers.")
    else:
        fig_corr = correlation_heatmap(prices, corr_tickers)
        st.plotly_chart(fig_corr, use_container_width=True)

        # Show descriptive stats for selected tickers
        rets = {}
        for t in corr_tickers:
            if t in prices:
                c = prices[t]["Close"].dropna()
                rets[t] = np.log(c / c.shift(1)).dropna()
        if rets:
            stats_df = pd.DataFrame({
                t: {
                    "Ann. vol": f"{s.std() * np.sqrt(252):.1%}",
                    "Skewness": f"{s.skew():.2f}",
                    "Kurtosis": f"{s.kurtosis():.1f}",
                    "Min daily ret": f"{s.min():.2%}",
                    "Max daily ret": f"{s.max():.2%}",
                }
                for t, s in rets.items()
            }).T
            st.dataframe(stats_df, use_container_width=True)
