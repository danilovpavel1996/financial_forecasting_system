"""Run Experiment — main Streamlit page.

Usage:
    streamlit run app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure src/ is importable when running via `streamlit run`
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import pandas as pd
import streamlit as st

from src.config import load_config
from src.data import universe
from src.eval.rank_backtester import ranking_comparison_table
from src.pipeline_ranking import run_ranking_pipeline

from dashboard.charts import (
    build_verdict,
    equity_curve,
    rolling_ric_chart,
    style_comparison_table,
    vol_scale_chart,
)
from dashboard.config_override import ALL_MODELS, DEFAULT_MODELS, UNIVERSE_META, RunConfig

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Research Dashboard — Run Experiment",
    page_icon="📊",
    layout="wide",
)

# ── Load project config once ─────────────────────────────────────────────────

@st.cache_data
def _load_cfg():
    return load_config()

cfg = _load_cfg()
ranked = universe.ranked_tickers(cfg)

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📊 Forecast Research")
    st.caption("Daily return forecasting research system")
    st.divider()

    st.subheader("Experiment settings")

    universe_choice = st.radio(
        "Universe",
        list(UNIVERSE_META.keys()),
        index=0,
        help="Select the asset universe to run the backtest on.",
    )
    _uni_meta = UNIVERSE_META[universe_choice]
    st.caption(_uni_meta["label"])

    horizon = st.select_slider(
        "Horizon (trading days)",
        options=[1, 5, 10, 21, 63],
        value=5,
        help="Forward return horizon for targets and evaluation.",
    )

    st.subheader("Prediction averaging")
    use_pred_avg = st.checkbox(
        "Prediction averaging (B3 variant)",
        value=False,
        help=(
            "Average predictions over the last N trading days before ranking. "
            "Phase 17 showed +0.07 Sharpe on equity sectors (h=63). "
            "Phase 18 showed +0.22 Sharpe on forex (h=5)."
        ),
    )
    pred_avg_window = 21
    if use_pred_avg:
        pred_avg_window = st.slider(
            "Prediction averaging window (days)",
            min_value=3, max_value=63, value=21, step=1,
        )
        st.caption(f"Will average predictions over the last {pred_avg_window} days before ranking.")

    st.subheader("Vol targeting")
    use_vt = st.checkbox("Enable vol targeting", value=False)
    vol_target = None
    if use_vt:
        vol_target = st.slider(
            "Target vol (annualised)",
            min_value=0.05, max_value=0.20, value=0.10, step=0.01,
            format="%.0f%%",
        )
        max_leverage = st.number_input(
            "Max leverage", min_value=1.0, max_value=5.0, value=2.0, step=0.5,
        )
        vol_lookback = st.number_input(
            "Lookback days", min_value=5, max_value=63, value=21, step=1,
        )
    else:
        max_leverage = 2.0
        vol_lookback = 21

    st.subheader("Features")
    use_cot = st.checkbox(
        "COT features",
        value=False,
        help="Add CFTC Commitment of Traders features. Phase 9 showed neutral/negative impact at h=5.",
    )
    force_refresh = st.checkbox(
        "Refresh features",
        value=False,
        help="Delete and rebuild cached feature matrices from raw data. Use after config or feature changes.",
    )

    st.subheader("Models")
    selected_models = st.multiselect(
        "Models to run",
        options=ALL_MODELS,
        default=DEFAULT_MODELS,
        help="LambdaMART requires lightgbm ≥ 4.0 and 'lambdarank' objective.",
    )

    run_btn = st.button("▶  Run Backtest", type="primary", use_container_width=True)

    st.divider()
    st.caption("⚠️ Research tooling — not investment advice.")

# ── Session state ─────────────────────────────────────────────────────────────

if "results" not in st.session_state:
    st.session_state.results = None
if "run_cfg" not in st.session_state:
    st.session_state.run_cfg = None

# ── Run on button click ───────────────────────────────────────────────────────

if run_btn:
    if not selected_models:
        st.error("Select at least one model.")
    elif universe_choice != "Commodities":
        st.warning(
            f"**{universe_choice}** universe is supported on the Live Signal page. "
            "The Run Experiment backtester currently runs on Commodities. "
            "Forex and Equity Sector backtests can be run from the command line via "
            "`python scripts/run_backtest.py`. Showing commodity results."
        )
    else:
        run_cfg = RunConfig(
            horizon=horizon,
            use_cot=use_cot,
            vol_target=vol_target,
            max_leverage=max_leverage,
            vol_lookback=int(vol_lookback),
            models=selected_models,
            force_refresh=force_refresh,
            universe=universe_choice,
            pred_avg_window=pred_avg_window if use_pred_avg else 1,
        )
        with st.spinner("Running backtest… this takes ~30–60 s"):
            try:
                results = run_ranking_pipeline(cfg, **run_cfg.pipeline_kwargs())
                # Filter to selected models only
                results = {k: v for k, v in results.items() if k in selected_models}
                st.session_state.results = results
                st.session_state.run_cfg = run_cfg
                st.success("Done!")
            except Exception as exc:
                st.error(f"Pipeline failed: {exc}")
                st.session_state.results = None

# ── Display results ───────────────────────────────────────────────────────────

st.title("Run Experiment")

if st.session_state.results is None:
    st.info(
        "Configure parameters in the sidebar and click **▶ Run Backtest** to start."
    )
    _meta = UNIVERSE_META.get(universe_choice, {})
    st.markdown(f"""
**Quick guide:**
- *Universe*: select the asset basket to backtest. Currently only Commodities runs via this page.
- *Horizon*: how many trading days forward the model predicts returns for.
- *Prediction averaging*: average predictions over N days before ranking (B3 variant).
- *Vol targeting*: scales position sizes so portfolio targets a constant annual volatility.
- *COT features*: adds CFTC positioning data (commodities only, best at horizons > 5d).
- *LambdaMART*: LightGBM with lambdarank objective (list-wise ranking loss).

**Reference baselines:**
{_meta.get('baseline_text', '')}
""")
    st.stop()

results = st.session_state.results
run_cfg: RunConfig = st.session_state.run_cfg

# Header
col_a, col_b = st.columns([3, 1])
with col_a:
    st.subheader(f"Results — horizon={run_cfg.horizon}, config: `{run_cfg.label()}`")
with col_b:
    save_btn = st.button("💾 Save run report")

# Comparison table
tbl = ranking_comparison_table(results)
st.markdown("#### Comparison table")
try:
    styled = style_comparison_table(tbl)
    st.dataframe(styled, use_container_width=True)
except Exception:
    st.dataframe(tbl, use_container_width=True)

# Verdict
verdict = build_verdict(results, run_cfg.horizon, run_cfg.vol_target)
st.info(verdict)

# Charts
col1, col2 = st.columns(2)
with col1:
    st.markdown("#### Equity curve")
    fig_eq = equity_curve(results, horizon=horizon)
    st.plotly_chart(fig_eq, use_container_width=True)

with col2:
    st.markdown("#### Rolling 63-day CS-RIC")
    fig_ric = rolling_ric_chart(results, window=63)
    st.plotly_chart(fig_ric, use_container_width=True)

# Vol scale chart (only when vol targeting is on)
if run_cfg.vol_target is not None:
    fig_scale = vol_scale_chart(results)
    if fig_scale is not None:
        st.markdown("#### Position scale factor (vol targeting)")
        st.plotly_chart(fig_scale, use_container_width=True)

# Detailed per-model stats
with st.expander("Detailed stats per model"):
    for name, r in results.items():
        if not np.isfinite(r.ls_sharpe):
            continue
        st.markdown(f"**{name}**")
        cols = st.columns(5)
        cols[0].metric("Sharpe", f"{r.ls_sharpe:.2f}")
        cols[1].metric("CS-RIC", f"{r.mean_cs_ric:+.4f}")
        cols[2].metric("Ann. return", f"{r.ls_ann_return:.1%}")
        cols[3].metric("Max DD", f"{r.ls_max_dd:.1%}")
        cols[4].metric("Turnover", f"{r.turnover:.3f}")

# Save report
if save_btn:
    try:
        _save_run_report(results, run_cfg, cfg)
        st.success(f"Report saved to outputs/reports/")
    except Exception as exc:
        st.error(f"Save failed: {exc}")


# ── Save helper ───────────────────────────────────────────────────────────────

def _save_run_report(results: dict, run_cfg: RunConfig, cfg) -> None:
    """Write a markdown report for this run to outputs/reports/."""
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"run_{run_cfg.label()}_{ts}.md"
    path = cfg.paths.outputs_reports / fname

    tbl = ranking_comparison_table(results)
    verdict = build_verdict(results, run_cfg.horizon, run_cfg.vol_target)

    lines = [
        f"# Backtest Run — {run_cfg.label()}",
        f"",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}  ",
        f"**Horizon:** {run_cfg.horizon}  ",
        f"**Vol target:** {run_cfg.vol_target}  ",
        f"**COT features:** {run_cfg.use_cot}  ",
        f"**Models:** {', '.join(run_cfg.models)}  ",
        f"",
        f"## Results",
        f"",
        f"```",
        tbl.to_string(),
        f"```",
        f"",
        f"## Verdict",
        f"",
        verdict,
        f"",
        f"---",
        f"",
        f"*Research tooling only — not investment advice.*",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
