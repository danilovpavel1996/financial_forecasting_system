"""Live Signal page — today's commodity or equity sector ranking."""
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

# ── Ensemble conviction colour ────────────────────────────────────────────────

_POS_TO_SCORE = {"LONG": 1.0, "SHORT": -1.0, "FLAT": 0.0}

# Colour constants used in both ensemble and single-signal views
_AGREE_LONG  = "#2A9D8F"   # green
_AGREE_SHORT = "#E76F51"   # red-orange
_DISAGREE    = "#E9C46A"   # yellow
_NEUTRAL     = "#888888"   # grey


def _ensemble_conviction(h5_pos: str, h63_pos: str, w_short: float, w_long: float
                         ) -> tuple[float, str, str]:
    """Return (blended_score, display_position, hex_colour) for one asset."""
    s5  = _POS_TO_SCORE.get(h5_pos,  0.0)
    s63 = _POS_TO_SCORE.get(h63_pos, 0.0)
    blend = w_short * s5 + w_long * s63

    if h5_pos == "LONG"  and h63_pos == "LONG":
        disp, colour = "LONG (both)",  _AGREE_LONG
    elif h5_pos == "SHORT" and h63_pos == "SHORT":
        disp, colour = "SHORT (both)", _AGREE_SHORT
    elif blend > 0.05:
        disp, colour = "LONG (lean)",  _DISAGREE
    elif blend < -0.05:
        disp, colour = "SHORT (lean)", _DISAGREE
    else:
        disp, colour = "FLAT",         _NEUTRAL

    return blend, disp, colour


def _render_ensemble(mr_sig, lgbm_sig, w_short: float, w_long: float, live_data) -> None:
    """Render the ensemble dual-ranking view."""
    # Header
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Signal date", str(mr_sig.date))
    col2.metric("Features as of", str(mr_sig.feature_date))
    col3.metric("Vol regime", mr_sig.confidence.current_vol_regime.upper())
    col4.metric("Ensemble weights", f"h5={w_short:.0%} / h63={w_long:.0%}")

    st.divider()

    # Build lookup dicts: ticker → position string
    mr_pos   = {r.ticker: r.position for r in mr_sig.rankings}
    lgbm_pos = {r.ticker: r.position for r in lgbm_sig.rankings}
    mr_pred   = {r.ticker: r.predicted_return for r in mr_sig.rankings}
    lgbm_pred = {r.ticker: r.predicted_return for r in lgbm_sig.rankings}

    # Side-by-side rankings + blended view
    st.subheader("Ensemble View — Rankings Side by Side")

    # Compute blended scores for all tickers (use MR ranking order)
    all_tickers = [r.ticker for r in mr_sig.rankings]
    blended_scores = {}
    for tkr in all_tickers:
        score, _, _ = _ensemble_conviction(
            mr_pos.get(tkr, "FLAT"), lgbm_pos.get(tkr, "FLAT"), w_short, w_long
        )
        blended_scores[tkr] = score

    # Sort by blended score descending
    sorted_tickers = sorted(all_tickers, key=lambda t: blended_scores[t], reverse=True)

    rows = []
    for rank, tkr in enumerate(sorted_tickers, 1):
        h5_p  = mr_pos.get(tkr, "FLAT")
        h63_p = lgbm_pos.get(tkr, "FLAT")
        blend, disp, _ = _ensemble_conviction(h5_p, h63_p, w_short, w_long)
        rows.append({
            "Rank": rank,
            "Ticker": tkr,
            "h=5 MeanRev": h5_p,
            "h=63 LightGBM": h63_p,
            "Blended Position": disp,
            "Blend score": f"{blend:+.3f}",
        })

    tbl_df = pd.DataFrame(rows).set_index("Rank")

    # Colour the blended position column
    _BLEND_COLOURS = {
        "LONG (both)":  f"color: {_AGREE_LONG};  font-weight: bold",
        "SHORT (both)": f"color: {_AGREE_SHORT}; font-weight: bold",
        "LONG (lean)":  f"color: {_DISAGREE};    font-weight: bold",
        "SHORT (lean)": f"color: {_DISAGREE};    font-weight: bold",
        "FLAT":         f"color: {_NEUTRAL}",
    }
    _H5_COLOURS  = {"LONG": f"color: {_AGREE_LONG}", "SHORT": f"color: {_AGREE_SHORT}", "FLAT": f"color: {_NEUTRAL}"}
    _H63_COLOURS = {"LONG": f"color: {_AGREE_LONG}", "SHORT": f"color: {_AGREE_SHORT}", "FLAT": f"color: {_NEUTRAL}"}

    def _colour_blend(val):
        return _BLEND_COLOURS.get(str(val), "")

    def _colour_sub(val):
        return _H5_COLOURS.get(str(val), "")

    styled = (
        tbl_df.style
        .applymap(_colour_blend, subset=["Blended Position"])
        .applymap(_colour_sub,   subset=["h=5 MeanRev", "h=63 LightGBM"])
    )
    st.dataframe(styled, use_container_width=True)

    # Legend
    st.caption(
        "🟢 LONG (both) = high conviction long  |  "
        "🔴 SHORT (both) = high conviction short  |  "
        "🟡 lean = signals disagree (reduced conviction)  |  "
        "⚫ FLAT = no net signal"
    )

    # Price charts for high-conviction picks
    high_conv = [r for r in rows if "both" in r["Blended Position"]]
    if high_conv and live_data is not None:
        st.divider()
        st.subheader("Price charts — High-conviction picks (last 90 days)")
        ncols = min(len(high_conv), 4)
        cols = st.columns(ncols)
        for i, r in enumerate(high_conv):
            with cols[i % ncols]:
                ticker = r["Ticker"]
                if ticker not in live_data.prices:
                    st.caption(f"{ticker}: no data")
                    continue
                close = live_data.prices[ticker]["Close"].dropna().tail(90)
                colour = _AGREE_LONG if "LONG" in r["Blended Position"] else _AGREE_SHORT
                fig = go.Figure(go.Scatter(
                    x=close.index, y=close.values,
                    mode="lines", line=dict(color=colour, width=2),
                    hovertemplate="%{x|%Y-%m-%d}<br>$%{y:,.2f}<extra></extra>",
                ))
                fig.update_layout(
                    **_LAYOUT,
                    title=f"{ticker} ({r['Blended Position']})",
                    showlegend=False,
                    height=220,
                )
                st.plotly_chart(fig, use_container_width=True)

    with st.expander("Ensemble context"):
        st.markdown(f"""
| Metric | Value |
|--------|-------|
| h=5 MeanReversion backtest Sharpe | {mr_sig.confidence.backtest_sharpe:.2f} |
| h=63 LightGBM backtest Sharpe | {lgbm_sig.confidence.backtest_sharpe:.2f} |
| Ensemble weights | h5={w_short:.0%} / h63={w_long:.0%} |
| Vol regime | {mr_sig.confidence.current_vol_regime} |
| h=63 model staleness | {lgbm_sig.confidence.days_since_retrain} day(s) |

*Theoretical diversified Sharpe ≈ √(0.40² + 0.79²) ≈ 0.89 if signals are uncorrelated.*
""")
        st.caption(
            "Blended score = w_short × h5_position + w_long × h63_position. "
            "Positions are: LONG=+1, SHORT=-1, FLAT=0. "
            "High-conviction = both sub-signals agree."
        )


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Live Signal",
    page_icon="🚦",
    layout="wide",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📊 Forecast Research")
    st.divider()
    universe_choice = st.radio(
        "Universe",
        ["Commodities", "Equity Sectors"],
        index=0,
        help=(
            "Commodities: 9 futures (GC=F, SI=F, PL=F, PA=F, CL=F, NG=F, HG=F, ZC=F, ZS=F)\n\n"
            "Equity Sectors: 9 SPDR ETFs (XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY)"
        ),
    )
    st.divider()
    horizon = st.select_slider("Horizon (trading days)", [1, 5, 10, 21, 63], value=63)
    _model_options = ["MeanReversion", "LightGBM", "Ensemble"]
    if universe_choice == "Equity Sectors":
        _model_options = ["MeanReversion", "LightGBM"]  # Ensemble mode is commodities-only for now
    model_choice = st.radio("Model", _model_options, index=0)
    if model_choice == "Ensemble":
        w_short = st.slider("Ensemble: weight h=5 (MeanRev)", 0.0, 1.0, 0.35, step=0.05,
                            help="Weight on h=5 MeanReversion; h=63 LightGBM gets the remainder")
        w_long = round(1.0 - w_short, 2)
        st.caption(f"h=5 weight: {w_short:.2f}  |  h=63 weight: {w_long:.2f}")
    else:
        w_short, w_long = 0.35, 0.65
    force_retrain = st.checkbox("Force retrain", value=False,
                                help="Retrain LightGBM even if model is fresh")
    force_refresh = st.checkbox("Force data refresh", value=False,
                                help="Re-fetch all data, bypass same-day cache")
    run_btn  = st.button("🚦 Refresh Signal",  type="primary", use_container_width=True)
    train_btn = st.button("🔁 Retrain Model", use_container_width=True,
                          disabled=(model_choice not in ("LightGBM", "Ensemble")))
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

def _generate(horizon, model_name, retrain, refresh, w_short=0.35, w_long=0.65,
              universe_name="Commodities"):
    from src.live.data import fetch_live_data
    from src.live.signal import generate_signal
    from src.live.trainer import load_latest_model, model_is_stale, train_live_model

    is_sectors = (universe_name == "Equity Sectors")

    if is_sectors:
        # Fetch sector ETF prices with sector start date
        import datetime, os
        from src.data.prices import fetch_all_tickers
        from src.data.macro import fetch_all_series

        today = datetime.date.today()
        end_prices = (today + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
        end_date   = today.strftime("%Y-%m-%d")
        start_date = universe.sector_start_date(cfg)

        live_cache = cfg.paths.data_raw.parent / "live" / end_date
        live_cache.mkdir(parents=True, exist_ok=True)

        sector_tkrs  = universe.sector_tickers(cfg)
        context_tkrs = universe.sector_context_tickers(cfg)
        all_tkrs     = universe.sector_price_tickers(cfg)

        prices = fetch_all_tickers(all_tkrs, start_date, end_prices, live_cache,
                                   force_refresh=refresh)
        macro_raw: dict = {}
        api_key = os.environ.get("FRED_API_KEY", "").strip()
        if api_key:
            try:
                macro_raw = fetch_all_series(
                    universe.fred_series(cfg), start_date, end_date, live_cache,
                    api_key=api_key, force_refresh=refresh,
                )
            except Exception as exc:
                st.warning(f"Macro fetch failed: {exc}")

        from src.live.data import LiveData
        live_data = LiveData(
            prices=prices, macro_raw=macro_raw, cot_raw={},
            as_of=today, start=start_date, end=end_date,
        )

        # Sector-specific overrides
        _ranked_override      = sector_tkrs
        _context_override     = context_tkrs
        _late_close_override  = set()          # all ETFs close at 4pm ET
        _ref_ticker_override  = sector_tkrs[0] # e.g., XLB

        # LightGBM for sectors: train on sector pooled dataset
        trained_model = None
        if model_name == "LightGBM":
            from src.features.pooled_dataset import build_pooled_dataset, feature_cols
            from src.models.gbm import LightGBMModel
            import numpy as _np

            existing = load_latest_model(cfg)
            sector_model_path = cfg.paths.outputs_models / "live_lgbm_sectors_latest.pkl"
            needs_train = retrain or not sector_model_path.exists()

            if needs_train:
                with st.spinner("Training LightGBM on sector history…"):
                    import joblib, datetime as _dt
                    pooled = build_pooled_dataset(
                        cfg, live_data.prices, live_data.macro_raw, horizon=horizon,
                        cot_raw=None,
                        ranked_override=_ranked_override,
                        context_override=_context_override,
                        ref_ticker_override=_ref_ticker_override,
                        late_close_override=_late_close_override,
                        carry_pairs_override={},
                    )
                    fcols = feature_cols(pooled)
                    X = pooled[fcols].values.astype(float)
                    y = pooled["target"].values.astype(float)
                    ok = _np.isfinite(y) & _np.all(_np.isfinite(X), axis=1)
                    model = LightGBMModel(random_state=cfg.random_seed)
                    model.fit(X[ok], y[ok])
                    ts = _dt.datetime.now()
                    payload = {"model": model, "feature_cols": fcols,
                               "trained_at": ts, "horizon": horizon, "n_rows": int(ok.sum())}
                    sector_model_path.parent.mkdir(parents=True, exist_ok=True)
                    joblib.dump(payload, sector_model_path)
                    from src.live.trainer import TrainedModel
                    trained_model = TrainedModel(
                        model=model, feature_cols=fcols, trained_at=ts,
                        horizon=horizon, n_rows=int(ok.sum()), path=sector_model_path,
                    )
            else:
                import joblib
                payload = joblib.load(sector_model_path)
                from src.live.trainer import TrainedModel
                trained_model = TrainedModel(
                    model=payload["model"], feature_cols=payload["feature_cols"],
                    trained_at=payload["trained_at"], horizon=payload["horizon"],
                    n_rows=payload["n_rows"], path=sector_model_path,
                )

        return generate_signal(
            cfg=cfg, live_data=live_data, trained_model=trained_model,
            horizon=horizon, model_name=model_name, use_cot=False,
            backtest_sharpe=float("nan"), backtest_cs_ric=float("nan"),
            vol_target=0.10,
            ranked_override=_ranked_override,
            context_override=_context_override,
            late_close_override=_late_close_override,
            ref_ticker_override=_ref_ticker_override,
        ), live_data

    # ── Commodity universe (existing logic, unchanged) ────────────────────────
    live_data = fetch_live_data(cfg, force_refresh=refresh, include_cot=False)

    if model_name == "Ensemble":
        mr_sig = generate_signal(
            cfg=cfg, live_data=live_data, trained_model=None,
            horizon=5, model_name="MeanReversion", use_cot=False,
            backtest_sharpe=0.40, backtest_cs_ric=0.020, vol_target=0.10,
        )
        existing = load_latest_model(cfg)
        needs_train = retrain or model_is_stale(existing, staleness_days=7)
        if needs_train:
            with st.spinner("Training LightGBM on full history…"):
                lgbm_model = train_live_model(cfg, live_data, horizon=63, use_cot=False)
        else:
            lgbm_model = existing
        lgbm_sig = generate_signal(
            cfg=cfg, live_data=live_data, trained_model=lgbm_model,
            horizon=63, model_name="LightGBM", use_cot=False,
            backtest_sharpe=0.79, backtest_cs_ric=0.055, vol_target=0.10,
        )
        return (mr_sig, lgbm_sig, w_short, w_long), live_data

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
        backtest_sharpe=0.79,
        backtest_cs_ric=0.055,
        vol_target=0.10,
    ), live_data

if run_btn or train_btn:
    with st.spinner("Fetching live data and generating signal…"):
        try:
            result = _generate(
                horizon, model_choice, force_retrain or train_btn, force_refresh,
                w_short=w_short, w_long=w_long,
                universe_name=universe_choice,
            )
            st.session_state.live_signal = result
            st.session_state.live_universe = universe_choice
            st.success("Signal updated!")
        except Exception as exc:
            st.error(f"Signal generation failed: {exc}")
            st.session_state.live_signal = None

# ── Display ───────────────────────────────────────────────────────────────────

_live_universe = st.session_state.get("live_universe", "Commodities")
st.title(f"🚦 Live Signal — {_live_universe}")

if st.session_state.live_signal is None:
    st.info("Click **🚦 Refresh Signal** in the sidebar to generate today's ranking.")
    st.markdown("""
**What this page shows:**
- Today's cross-sectional ranking of 9 assets based on the selected model and universe
- LONG = top-2 predicted forward returns, SHORT = bottom-2, FLAT = middle 5
- The MeanReversion model ranks by **negative 5-day trailing return** (mean-reversion heuristic)
- Features use yesterday's closes — same exact code path as the backtest
- **Ensemble mode** (Commodities only) combines h=5 MeanReversion + h=63 LightGBM

**Commodities universe:** GC=F, SI=F, PL=F, PA=F, CL=F, NG=F, HG=F, ZC=F, ZS=F
*Phase 14 out-of-sample Sharpe: 0.79 (h=63, LightGBM no-COT, 2005–2024)*

**Equity Sectors universe:** XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY
*Phase 17 best: 0.48 Sharpe (h=63, LightGBM + 21-day prediction averaging; turnover 0.031)*

**Cross-asset ensemble (Phase 17):** 50/50 commodity + equity → Sharpe 0.97 (2016–2023 OOS, ρ=0.13)
*See outputs/reports/phase17_summary.md for full results.*
""")
    st.stop()

_POS_COLOUR = {"LONG": "#2A9D8F", "SHORT": "#E76F51", "FLAT": "#888888"}

# ── Detect ensemble vs single-signal mode ────────────────────────────────────

_is_ensemble = (
    isinstance(st.session_state.live_signal, tuple)
    and len(st.session_state.live_signal) == 2
    and isinstance(st.session_state.live_signal[0], tuple)
    and len(st.session_state.live_signal[0]) == 4
)

if _is_ensemble:
    (mr_sig, lgbm_sig, _w_short, _w_long), live_data = st.session_state.live_signal
    _render_ensemble(mr_sig, lgbm_sig, _w_short, _w_long, live_data)
    st.stop()

sig, live_data = st.session_state.live_signal

# ── Single-signal display ─────────────────────────────────────────────────────

# Header metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Signal date", str(sig.date))
col2.metric("Features as of", str(sig.feature_date))
col3.metric("Vol regime", sig.confidence.current_vol_regime.upper())
col4.metric("Vol scale", f"{sig.confidence.vol_scale:.2f}×")

st.divider()

# ── Ranking table ─────────────────────────────────────────────────────────────

st.subheader(f"Ranking — {sig.model_name}  (horizon={sig.horizon}d)")

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
