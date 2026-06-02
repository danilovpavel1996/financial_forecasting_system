"""Live Signal page — config-driven, all 4 signals from 18 phases."""
from __future__ import annotations

import datetime
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
from dashboard.signal_configs import SIGNALS, FOREX_NAMES

# ── Constants ─────────────────────────────────────────────────────────────────

_POS_COLOUR = {"LONG": "#2A9D8F", "SHORT": "#E76F51", "FLAT": "#888888"}

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Live Signal",
    page_icon="🚦",
    layout="wide",
)

# ── Config ────────────────────────────────────────────────────────────────────

@st.cache_data
def _load_cfg():
    return load_config()

cfg = _load_cfg()

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📊 Forecast Research")
    st.divider()

    signal_name = st.selectbox(
        "Signal",
        list(SIGNALS.keys()),
        index=0,
        help=(
            "Choose one of the 4 signals built across 18 phases.\n\n"
            "• Commodity h=63 (Sharpe 0.79)\n"
            "• Forex h=5 PredAvg21 (Sharpe 0.94)\n"
            "• Equity B3 h=63 (Sharpe 0.41)\n"
            "• Cross-Asset Blend 50/50 (Sharpe 1.05)"
        ),
    )
    sig_cfg = SIGNALS[signal_name]

    st.divider()
    force_retrain = st.checkbox("Force retrain", value=False,
                                help="Retrain the model even if a cached version exists")
    force_refresh = st.checkbox("Force data refresh", value=False,
                                help="Re-fetch all data, bypass same-day cache")

    run_btn   = st.button("🚦 Refresh Signal", type="primary", use_container_width=True)
    train_btn = st.button("🔁 Retrain Model", use_container_width=True,
                          disabled=(sig_cfg.get("model", "") not in ("LightGBM",)))
    st.divider()

    st.subheader("Portfolio Simulator")
    sim_capital = st.number_input(
        "Starting capital ($)",
        min_value=100.0, max_value=10_000_000.0, value=1_000.0, step=100.0,
        format="%.0f",
    )
    sim_platform = st.radio(
        "Platform type",
        ["Long-only", "Long-Short (futures/CFD)"],
        index=0,
        help=(
            "Long-only: allocate capital only to BUY positions.\n\n"
            "Long-Short: allocate capital to both BUY and SELL positions."
        ),
    )
    st.divider()
    st.caption("⚠️ Research tooling — not investment advice.")

# ── Session state ─────────────────────────────────────────────────────────────

if "live_signal" not in st.session_state:
    st.session_state.live_signal = None
if "live_signal_name" not in st.session_state:
    st.session_state.live_signal_name = None

# ── Forex live data fetcher ───────────────────────────────────────────────────

def _fetch_forex_live_data(refresh: bool):
    """Fetch live price data for the forex universe."""
    from src.data.prices import fetch_all_tickers
    from src.live.data import LiveData

    today = datetime.date.today()
    end_prices = (today + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    end_date   = today.strftime("%Y-%m-%d")
    start_date = universe.forex_start_date(cfg)

    live_cache = cfg.paths.data_raw.parent / "live" / end_date
    live_cache.mkdir(parents=True, exist_ok=True)

    all_tkrs = universe.forex_price_tickers(cfg)
    prices = fetch_all_tickers(all_tkrs, start_date, end_prices, live_cache,
                               force_refresh=refresh)
    return LiveData(
        prices=prices, macro_raw={}, cot_raw={},
        as_of=today, start=start_date, end=end_date,
    )


# ── Signal generation ─────────────────────────────────────────────────────────

def _generate_single(scfg: dict, retrain: bool, refresh: bool):
    """Generate a single-universe signal and return (LiveSignal, LiveData)."""
    from src.live.data import fetch_live_data
    from src.live.signal import generate_signal
    from src.live.trainer import (
        TrainedModel, load_latest_model, model_is_stale, train_live_model,
    )

    uni = scfg["universe"]
    model_name  = scfg["model"]
    horizon     = scfg["horizon"]
    backtest_sh = scfg["backtest_sharpe"]
    backtest_ric = scfg["backtest_cs_ric"]

    # ── Fetch data ────────────────────────────────────────────────────────────
    if uni == "commodities":
        live_data = fetch_live_data(cfg, force_refresh=refresh, include_cot=False)
        ranked_tkrs   = None   # default from cfg
        context_tkrs  = None
        late_close    = None
        ref_ticker    = None

    elif uni == "forex":
        live_data    = _fetch_forex_live_data(refresh)
        ranked_tkrs  = universe.forex_tickers(cfg)
        context_tkrs = universe.forex_context_tickers(cfg)
        late_close   = set()   # forex: all pairs on same timestamp
        ref_ticker   = ranked_tkrs[0]

    elif uni == "equity_sectors":
        import os
        from src.data.prices import fetch_all_tickers
        from src.data.macro import fetch_all_series
        from src.live.data import LiveData

        today       = datetime.date.today()
        end_prices  = (today + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
        end_date    = today.strftime("%Y-%m-%d")
        start_date  = universe.sector_start_date(cfg)
        live_cache  = cfg.paths.data_raw.parent / "live" / end_date
        live_cache.mkdir(parents=True, exist_ok=True)

        all_tkrs = universe.sector_price_tickers(cfg)
        prices   = fetch_all_tickers(all_tkrs, start_date, end_prices, live_cache,
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

        live_data    = LiveData(prices=prices, macro_raw=macro_raw, cot_raw={},
                                as_of=today, start=start_date, end=end_date)
        ranked_tkrs  = universe.sector_tickers(cfg)
        context_tkrs = universe.sector_context_tickers(cfg)
        late_close   = set()
        ref_ticker   = ranked_tkrs[0]
    else:
        raise ValueError(f"Unknown universe: {uni}")

    # ── Train or load model ───────────────────────────────────────────────────
    trained_model = None
    if model_name == "LightGBM":
        model_key = scfg.get("model_path_key", "live_lgbm_latest")
        model_path = cfg.paths.outputs_models / f"{model_key}.pkl"

        needs_train = retrain or not model_path.exists()
        if not needs_train:
            import joblib
            try:
                payload = joblib.load(model_path)
                from src.live.trainer import TrainedModel
                trained_model = TrainedModel(
                    model=payload["model"], feature_cols=payload["feature_cols"],
                    trained_at=payload["trained_at"], horizon=payload["horizon"],
                    n_rows=payload["n_rows"], path=model_path,
                )
                # stale if trained > 7 days ago
                if (datetime.datetime.now() - trained_model.trained_at).days > 7:
                    needs_train = True
            except Exception:
                needs_train = True

        if needs_train:
            if uni == "commodities":
                with st.spinner("Training LightGBM on commodity history…"):
                    trained_model = train_live_model(cfg, live_data, horizon=horizon,
                                                     use_cot=False)
            else:
                with st.spinner(f"Training LightGBM on {uni} history…"):
                    from src.features.pooled_dataset import build_pooled_dataset, feature_cols
                    from src.models.gbm import LightGBMModel
                    import joblib, datetime as _dt

                    kw = {}
                    if uni == "forex":
                        kw = dict(
                            ranked_override=ranked_tkrs,
                            context_override=context_tkrs,
                            ref_ticker_override=ref_ticker,
                            late_close_override=late_close,
                            carry_pairs_override={},
                        )
                    elif uni == "equity_sectors":
                        kw = dict(
                            ranked_override=ranked_tkrs,
                            context_override=context_tkrs,
                            ref_ticker_override=ref_ticker,
                            late_close_override=late_close,
                            carry_pairs_override={},
                        )

                    pooled = build_pooled_dataset(
                        cfg, live_data.prices, live_data.macro_raw,
                        horizon=horizon, cot_raw=None, **kw,
                    )
                    fcols = feature_cols(pooled)
                    X = pooled[fcols].values.astype(float)
                    y = pooled["target"].values.astype(float)
                    ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1)

                    model = LightGBMModel(random_state=cfg.random_seed)
                    model.fit(X[ok], y[ok])
                    ts = _dt.datetime.now()
                    payload = {
                        "model": model, "feature_cols": fcols,
                        "trained_at": ts, "horizon": horizon, "n_rows": int(ok.sum()),
                    }
                    model_path.parent.mkdir(parents=True, exist_ok=True)
                    joblib.dump(payload, model_path)
                    trained_model = TrainedModel(
                        model=model, feature_cols=fcols, trained_at=ts,
                        horizon=horizon, n_rows=int(ok.sum()), path=model_path,
                    )

    # ── Generate signal ───────────────────────────────────────────────────────
    from src.live.signal import generate_signal
    sig = generate_signal(
        cfg=cfg, live_data=live_data, trained_model=trained_model,
        horizon=horizon, model_name=model_name, use_cot=False,
        backtest_sharpe=backtest_sh, backtest_cs_ric=backtest_ric,
        vol_target=0.10,
        ranked_override=ranked_tkrs,
        context_override=context_tkrs,
        late_close_override=late_close,
        ref_ticker_override=ref_ticker,
    )
    return sig, live_data


def _generate_ensemble(scfg: dict, retrain: bool, refresh: bool):
    """Generate a 50/50 ensemble signal and return ((sig_a, sig_b, weights), live_data)."""
    comp_names  = scfg["components"]
    weights     = scfg["weights"]

    results = []
    for name in comp_names:
        csig = SIGNALS[name]
        sig, live_data = _generate_single(csig, retrain, refresh)
        results.append((sig, live_data))

    return (results[0][0], results[1][0], weights), results[0][1]


# ── Trigger generation ────────────────────────────────────────────────────────

if run_btn or train_btn:
    with st.spinner("Fetching live data and generating signal…"):
        try:
            if sig_cfg["type"] == "ensemble":
                result = _generate_ensemble(sig_cfg, force_retrain or train_btn, force_refresh)
            else:
                result = _generate_single(sig_cfg, force_retrain or train_btn, force_refresh)
            st.session_state.live_signal      = result
            st.session_state.live_signal_name = signal_name
            st.success("Signal updated!")
        except Exception as exc:
            st.error(f"Signal generation failed: {exc}")
            st.session_state.live_signal = None

# ── Title ─────────────────────────────────────────────────────────────────────

active_name = st.session_state.get("live_signal_name") or signal_name
st.title(f"🚦 Live Signal — {active_name}")

# ── Welcome screen ────────────────────────────────────────────────────────────

if st.session_state.live_signal is None:
    st.info("Choose a signal in the sidebar and click **🚦 Refresh Signal**.")
    st.markdown("""
| Signal | Universe | Horizon | OOS Sharpe |
|--------|----------|---------|-----------|
| Commodity LightGBM h=63 | 9 commodity futures | 63 d | **0.79** |
| Forex LightGBM h=5 | 7 major USD pairs | 5 d | **0.94** |
| Equity B3 LightGBM h=63 | 9 SPDR sector ETFs | 63 d | **0.41** |
| Cross-Asset Blend | Commodity + Forex 50/50 | — | **1.05** |

*All Sharpe ratios are out-of-sample walk-forward, net of transaction costs.*
""")
    st.stop()

# ── Helpers ───────────────────────────────────────────────────────────────────

def _colour_position(val):
    c = _POS_COLOUR.get(str(val), "")
    return f"color: {c}; font-weight: bold" if c else ""


def _action_from_position(pos: str) -> str:
    return {"LONG": "BUY", "SHORT": "SELL", "FLAT": "HOLD"}.get(pos, "HOLD")


def _ranking_table(rankings, live_data=None, sig_date=None, horizon: int = 63):
    """Build a styled ranking DataFrame with trading columns."""
    rows = []
    for r in rankings:
        name = FOREX_NAMES.get(r.ticker, r.name)
        action = _action_from_position(r.position)
        pred_pct = (np.exp(r.predicted_return) - 1) * 100

        # Entry price: today's close if available
        entry_price = np.nan
        if live_data is not None and r.ticker in live_data.prices:
            close = live_data.prices[r.ticker]["Close"].dropna()
            if len(close):
                entry_price = float(close.iloc[-1])

        target_price = entry_price * np.exp(r.predicted_return) if np.isfinite(entry_price) else np.nan

        # Review date: entry_date + horizon business days
        if sig_date is not None:
            review_dates = pd.bdate_range(pd.Timestamp(sig_date), periods=horizon + 1)
            review_date = str(review_dates[-1].date())
        else:
            review_date = "—"

        entry_str  = f"{entry_price:,.4g}"  if np.isfinite(entry_price)  else "N/A"
        target_str = f"{target_price:,.4g}" if np.isfinite(target_price) else "N/A"

        rows.append({
            "Rank":            r.rank,
            "Ticker":          r.ticker,
            "Name":            name,
            "Action":          action,
            "Entry price":     entry_str,
            "Target price":    target_str,
            "Exp. return":     f"{pred_pct:+.2f}%",
            "Entry date":      str(sig_date) if sig_date else "—",
            "Review date":     review_date,
            "Horizon (d)":     horizon,
            "Mom 5d":          f"{r.recent_mom_5d:+.2%}" if np.isfinite(r.recent_mom_5d) else "N/A",
            "EWMA vol":        f"{r.current_vol:.1%}"    if np.isfinite(r.current_vol)  else "N/A",
        })

    df = pd.DataFrame(rows).set_index("Rank")

    _ACTION_COLOUR = {"BUY": f"color: {_POS_COLOUR['LONG']}; font-weight: bold",
                      "SELL": f"color: {_POS_COLOUR['SHORT']}; font-weight: bold",
                      "HOLD": f"color: {_POS_COLOUR['FLAT']}"}

    def _colour_action(val):
        return _ACTION_COLOUR.get(str(val), "")

    return df.style.map(_colour_action, subset=["Action"])


def _action_summary(rankings, live_data, sig_date, horizon: int) -> str:
    """Return a markdown string listing BUY/SELL lines for LONG/SHORT picks."""
    lines = []
    review_dates = pd.bdate_range(pd.Timestamp(sig_date), periods=horizon + 1)
    review_date = str(review_dates[-1].date())
    for r in rankings:
        if r.position not in ("LONG", "SHORT"):
            continue
        action = _action_from_position(r.position)
        name = FOREX_NAMES.get(r.ticker, r.name)
        pred_pct = (np.exp(r.predicted_return) - 1) * 100
        sign = "+" if pred_pct >= 0 else ""

        entry_str = target_str = "N/A"
        if live_data is not None and r.ticker in live_data.prices:
            close = live_data.prices[r.ticker]["Close"].dropna()
            if len(close):
                entry = float(close.iloc[-1])
                target = entry * np.exp(r.predicted_return)
                entry_str  = f"{entry:,.4g}"
                target_str = f"{target:,.4g}"

        lines.append(
            f"**{action} {r.ticker}** ({name}) "
            f"at {entry_str} → target {target_str} "
            f"by {review_date} ({sign}{pred_pct:.2f}%)"
        )
    return "\n\n".join(lines) if lines else "_No LONG/SHORT picks._"


def _hex_to_rgba(h: str, alpha: float = 0.12) -> str:
    """Convert #RRGGBB to rgba(r,g,b,alpha)."""
    h = h.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _forecast_subplots(picks_with_horizon: list[tuple], live_data) -> list[go.Figure]:
    """Return one Plotly figure per (AssetRanking, horizon) pair.

    Each figure shows:
      - Solid line: last 90 days of actual prices
      - Dashed line: forecast from today to today+horizon
      - Shaded ±1σ band (σ = std of predicted returns across all picks)
      - Annotation showing predicted return as a percentage
    Y-axis is zoomed to [min_90d, max(max_90d, proj_price)] ± 5% padding.
    """
    if not picks_with_horizon:
        return []

    # Pre-compute σ across all picks for the uncertainty band
    all_preds = [r.predicted_return for r, _ in picks_with_horizon if np.isfinite(r.predicted_return)]
    sigma_r = float(np.std(all_preds)) if len(all_preds) > 1 else 0.01

    figs = []
    for r, horizon in picks_with_horizon:
        ticker = r.ticker
        if ticker not in live_data.prices:
            continue
        close = live_data.prices[ticker]["Close"].dropna().tail(90)
        if len(close) == 0:
            continue

        colour = _POS_COLOUR[r.position]
        name   = FOREX_NAMES.get(ticker, r.name)

        current_price = float(close.iloc[-1])
        pred_return   = r.predicted_return
        proj_price    = current_price * np.exp(pred_return)
        pred_pct      = (np.exp(pred_return) - 1) * 100  # % change

        # Forecast dates and prices
        try:
            forecast_dates  = pd.bdate_range(pd.Timestamp(close.index[-1]), periods=horizon + 1)
        except Exception:
            continue
        forecast_prices = np.linspace(current_price, proj_price, len(forecast_dates))

        upper_prices = current_price * np.exp(
            np.linspace(0, pred_return + sigma_r, len(forecast_dates))
        )
        lower_prices = current_price * np.exp(
            np.linspace(0, pred_return - sigma_r, len(forecast_dates))
        )

        # Y-axis range: zoom to actual + forecast range with 5% padding
        y_min = min(float(close.min()), float(lower_prices.min()))
        y_max = max(float(close.max()), float(upper_prices.max()))
        pad   = (y_max - y_min) * 0.05 if y_max > y_min else abs(y_max) * 0.05 + 1e-8
        y_range = [y_min - pad, y_max + pad]

        fig = go.Figure()

        # Actual history (solid)
        fig.add_trace(go.Scatter(
            x=close.index, y=close.values,
            mode="lines",
            line=dict(color="#555555", width=1.8),
            name="Actual",
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:,.4g}<extra>Actual</extra>",
        ))

        # Shaded uncertainty band
        fill_colour = _hex_to_rgba(colour) if colour.startswith("#") else colour
        fig.add_trace(go.Scatter(
            x=list(forecast_dates) + list(forecast_dates[::-1]),
            y=list(upper_prices)   + list(lower_prices[::-1]),
            fill="toself",
            fillcolor=fill_colour,
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False,
            hoverinfo="skip",
        ))

        # Dashed forecast line
        sign = "+" if pred_pct >= 0 else ""
        fig.add_trace(go.Scatter(
            x=forecast_dates, y=forecast_prices,
            mode="lines",
            line=dict(color=colour, width=2.5, dash="dash"),
            name=f"Forecast ({sign}{pred_pct:.2f}%)",
            hovertemplate=(
                f"%{{x|%Y-%m-%d}}<br>%{{y:,.4g}}"
                f"<br>Pred: {sign}{pred_pct:.2f}% over {horizon}d<extra>Forecast</extra>"
            ),
        ))

        # Annotation: relative-to-basket message
        rel_word = "OUTPERFORM" if pred_return >= 0 else "UNDERPERFORM"
        annotation_text = (
            f"<b>Model says:</b> {rel_word} basket<br>"
            f"by <b>{sign}{pred_pct:.2f}%</b> over {horizon} trading days"
        )
        fig.add_annotation(
            x=forecast_dates[-1],
            y=proj_price,
            text=annotation_text,
            showarrow=True,
            arrowhead=2,
            arrowcolor=colour,
            font=dict(color=colour, size=11),
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor=colour,
            borderwidth=1,
            ax=-80, ay=-40,
            align="left",
        )

        subplot_title = f"{name} — {r.position}  (h={horizon}d)"
        layout = {**_LAYOUT}
        layout["title"] = subplot_title
        layout["height"] = 300
        layout["yaxis"] = dict(
            showgrid=True,
            gridcolor="rgba(128,128,128,0.2)",
            range=y_range,
            tickformat=".4g",
        )
        fig.update_layout(**layout)
        figs.append(fig)

    return figs


def _portfolio_positions(
    rankings_with_horizon: list[tuple],   # [(AssetRanking, horizon, leg_capital), ...]
    live_data,
    long_only: bool,
    sig_date,
) -> list[dict]:
    """Compute position rows for the portfolio simulator.

    rankings_with_horizon: list of (ranking, horizon, leg_capital) tuples.
    Each leg_capital is the dollar amount allocated to that sub-universe leg.
    Within a leg, capital is split equally across LONG picks (and SHORT picks
    in long-short mode).
    """
    # Group by leg_capital so we can split within each leg independently
    from collections import defaultdict
    legs: dict[float, list[tuple]] = defaultdict(list)
    for r, h, cap in rankings_with_horizon:
        legs[cap].append((r, h))

    rows = []
    for cap, items in legs.items():
        long_picks  = [(r, h) for r, h in items if r.position == "LONG"]
        short_picks = [(r, h) for r, h in items if r.position == "SHORT"]

        active = long_picks if long_only else long_picks + short_picks
        if not active:
            continue
        per_pos = cap / len(active)

        for r, h in active:
            action = _action_from_position(r.position)
            name   = FOREX_NAMES.get(r.ticker, r.name)
            pred_pct = (np.exp(r.predicted_return) - 1) * 100
            sign     = "+" if pred_pct >= 0 else ""

            entry_price = target_price = np.nan
            if live_data is not None and r.ticker in live_data.prices:
                close = live_data.prices[r.ticker]["Close"].dropna()
                if len(close):
                    entry_price  = float(close.iloc[-1])
                    target_price = entry_price * np.exp(r.predicted_return)

            expected_pnl = per_pos * (np.exp(r.predicted_return) - 1)

            review_dates = pd.bdate_range(pd.Timestamp(sig_date), periods=h + 1)
            review_date  = str(review_dates[-1].date())

            rows.append({
                "Ticker":          r.ticker,
                "Name":            name,
                "Action":          action,
                "Allocation ($)":  round(per_pos, 2),
                "Entry price":     f"{entry_price:,.4g}"  if np.isfinite(entry_price)  else "N/A",
                "Target price":    f"{target_price:,.4g}" if np.isfinite(target_price) else "N/A",
                "Exp. return":     f"{sign}{pred_pct:.2f}%",
                "Exp. P&L ($)":    round(expected_pnl, 2),
                "Entry date":      str(sig_date),
                "Review date":     review_date,
                "_pred_return":    r.predicted_return,   # kept for totals, dropped in display
            })

    return rows


def _render_portfolio_sim(
    rankings_with_horizon: list[tuple],
    live_data,
    capital: float,
    long_only: bool,
    sig_date,
    scfg: dict,
    is_ensemble: bool = False,
) -> None:
    """Render the Portfolio Simulator section."""
    st.divider()
    st.subheader("Portfolio Simulator")

    rows = _portfolio_positions(rankings_with_horizon, live_data, long_only, sig_date)

    if not rows:
        st.info("No active positions to display (no LONG picks, or Long-only mode with no LONGs).")
        return

    display_cols = [
        "Ticker", "Name", "Action", "Allocation ($)",
        "Entry price", "Target price", "Exp. return", "Exp. P&L ($)",
        "Entry date", "Review date",
    ]
    df = pd.DataFrame(rows)[display_cols]

    _ACTION_COLOUR = {
        "BUY":  f"color: {_POS_COLOUR['LONG']}; font-weight: bold",
        "SELL": f"color: {_POS_COLOUR['SHORT']}; font-weight: bold",
    }

    def _col_action(val):
        return _ACTION_COLOUR.get(str(val), "")

    st.dataframe(df.style.map(_col_action, subset=["Action"]), use_container_width=True)

    # Totals
    total_pnl      = sum(r["Exp. P&L ($)"] for r in rows)
    total_alloc    = sum(r["Allocation ($)"] for r in rows)
    total_ret_pct  = (total_pnl / total_alloc * 100) if total_alloc else 0.0
    sign_tot       = "+" if total_pnl >= 0 else ""

    col1, col2 = st.columns(2)
    col1.metric("Total expected P&L", f"${total_pnl:+,.2f}",
                delta=f"{sign_tot}{total_ret_pct:.2f}% on deployed capital")

    # Annualised return estimate from Sharpe and target vol (10%)
    backtest_sharpe = scfg.get("backtest_sharpe", float("nan"))
    target_vol      = 0.10   # 10% annualised vol target used in backtests
    if np.isfinite(backtest_sharpe):
        ann_return_est = backtest_sharpe * target_vol * 100
        col2.metric("Ann. return estimate (Sharpe × vol)",
                    f"{ann_return_est:+.1f}%/yr",
                    help=f"backtest_sharpe ({backtest_sharpe:.2f}) × target_vol (10%). "
                         "Not a forward-looking guarantee.")
    else:
        col2.metric("Ann. return estimate", "N/A")

    # Rebalance dates
    st.markdown("**Next rebalance dates:**")
    if is_ensemble:
        comm_dates  = pd.bdate_range(pd.Timestamp(sig_date), periods=64)
        forex_dates = pd.bdate_range(pd.Timestamp(sig_date), periods=6)
        st.markdown(
            f"- Commodity leg (h=63): **{comm_dates[-1].date()}**\n"
            f"- Forex leg (h=5): **{forex_dates[-1].date()}**"
        )
    else:
        h = rankings_with_horizon[0][1] if rankings_with_horizon else 63
        reb_dates = pd.bdate_range(pd.Timestamp(sig_date), periods=h + 1)
        st.markdown(f"- All positions: **{reb_dates[-1].date()}** ({h} business days)")

    # Disclaimer
    st.error(
        "**DISCLAIMER:** This simulator shows hypothetical position sizing based on model "
        "estimates. It is **NOT a trading recommendation**. Past backtest performance "
        f"(Sharpe {backtest_sharpe:.2f}) does not guarantee future results. "
        "Consult a financial advisor before investing real money."
    )


def _render_single(sig, live_data, scfg: dict, name: str) -> None:
    """Render a single-signal view."""
    # Header metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Signal date", str(sig.date))
    col2.metric("Features as of", str(sig.feature_date))
    col3.metric("Vol regime", sig.confidence.current_vol_regime.upper())
    col4.metric("OOS Sharpe", f"{scfg['backtest_sharpe']:.2f}")

    st.divider()

    # Ranking table
    st.subheader(f"Today's Ranking — {sig.model_name}  (h={sig.horizon}d)")
    st.dataframe(
        _ranking_table(sig.rankings, live_data=live_data, sig_date=sig.date, horizon=sig.horizon),
        use_container_width=True,
    )
    st.caption("Action: BUY=LONG / SELL=SHORT / HOLD=FLAT  |  "
               "Target = entry × exp(pred_return)  |  "
               "Review date = entry + horizon business days")

    # ── Action Summary ────────────────────────────────────────────────────────
    picks = [r for r in sig.rankings if r.position in ("LONG", "SHORT")]
    if picks and live_data is not None:
        st.divider()
        st.subheader("Action Summary")
        summary_md = _action_summary(sig.rankings, live_data, sig.date, sig.horizon)
        st.markdown(summary_md)
        st.caption(
            "Action summary is for research reference only. "
            "These are model estimates, not trading recommendations."
        )

    # ── HERO: forward forecast subplots ──────────────────────────────────────
    if picks and live_data is not None:
        st.divider()
        st.subheader("Forward Forecast — LONG & SHORT picks")
        st.warning(
            "**Projected prices are model estimates, not predictions. "
            f"Historical accuracy (CS-RIC): {scfg['backtest_cs_ric']:+.3f}. "
            "Research tooling only — not investment advice.**"
        )
        picks_with_horizon = [(r, sig.horizon) for r in picks]
        figs = _forecast_subplots(picks_with_horizon, live_data)
        # Render two per row
        for i in range(0, len(figs), 2):
            row_cols = st.columns(min(2, len(figs) - i))
            for j, col in enumerate(row_cols):
                with col:
                    st.plotly_chart(figs[i + j], use_container_width=True)

        st.caption(
            "Gray solid = last 90 trading days of actual prices. "
            f"Colored dashed = forecast over {sig.horizon} business days "
            "(current price × exp(pred_return)). "
            "Shaded band = ±1σ of predicted returns across all picks. "
            "Y-axis zoomed to actual + forecast range."
        )

    # ── Historical equity curve ───────────────────────────────────────────────
    with st.expander("Historical OOS performance", expanded=False):
        _render_equity_curve(scfg)

    # ── Signal context ────────────────────────────────────────────────────────
    with st.expander("Signal context"):
        c = sig.confidence
        st.markdown(f"""
| Metric | Value |
|--------|-------|
| Backtest Sharpe (OOS) | **{c.backtest_sharpe:.2f}** |
| Backtest CS-RIC | {c.backtest_cs_ric:+.4f} |
| Vol regime | {c.current_vol_regime} |
| Vol-targeting scale | {c.vol_scale:.2f}× |
| Model staleness | {c.days_since_retrain} day(s) |
| Model | {sig.model_name} |
| Description | {scfg.get('description', '')} |
""")
        st.caption(
            "Vol scale < 1 means current vol is above target — positions would be "
            "reduced if vol-targeting were applied. Signal shows unscaled ranks."
        )

    # ── Portfolio Simulator ───────────────────────────────────────────────────
    _render_portfolio_sim(
        rankings_with_horizon=[(r, sig.horizon, sim_capital) for r in sig.rankings],
        live_data=live_data,
        capital=sim_capital,
        long_only=(sim_platform == "Long-only"),
        sig_date=sig.date,
        scfg=scfg,
        is_ensemble=False,
    )

    # ── Signal history ────────────────────────────────────────────────────────
    _render_signal_history(sig)


def _render_ensemble(sig_a, sig_b, weights, live_data, scfg: dict) -> None:
    """Render the 50/50 ensemble (commodity + forex) view."""
    w_a, w_b = weights[0], weights[1]

    col1, col2, col3 = st.columns(3)
    col1.metric("OOS Sharpe (ensemble)", f"{scfg['backtest_sharpe']:.2f}")
    col2.metric("Weights", f"{w_a:.0%} / {w_b:.0%}")
    col3.metric("Signal date", str(sig_a.date))

    st.divider()
    st.subheader("Ensemble Ranking — Blended Score")

    _POS_TO_SCORE = {"LONG": 1.0, "SHORT": -1.0, "FLAT": 0.0}

    # Build score lookup per-signal
    scores_a = {r.ticker: _POS_TO_SCORE.get(r.position, 0.0) * w_a for r in sig_a.rankings}
    scores_b = {r.ticker: _POS_TO_SCORE.get(r.position, 0.0) * w_b for r in sig_b.rankings}
    pos_a = {r.ticker: r.position for r in sig_a.rankings}
    pos_b = {r.ticker: r.position for r in sig_b.rankings}

    all_tickers = [r.ticker for r in sig_a.rankings] + [
        r.ticker for r in sig_b.rankings if r.ticker not in {r.ticker for r in sig_a.rankings}
    ]
    blended = {t: scores_a.get(t, 0.0) + scores_b.get(t, 0.0) for t in all_tickers}
    sorted_tkrs = sorted(all_tickers, key=lambda t: blended[t], reverse=True)

    n = len(sorted_tkrs)
    n_long, n_short = 2, 2

    rows = []
    for rank, tkr in enumerate(sorted_tkrs, 1):
        if rank <= n_long:
            blend_pos = "LONG"
        elif rank > n - n_short:
            blend_pos = "SHORT"
        else:
            blend_pos = "FLAT"
        a_p = pos_a.get(tkr, "—")
        b_p = pos_b.get(tkr, "—")
        rows.append({
            "Rank": rank,
            "Ticker": tkr,
            "Commodity": a_p,
            "Forex": b_p,
            "Blended Position": blend_pos,
            "Blend score": f"{blended[tkr]:+.3f}",
        })

    tbl_df = pd.DataFrame(rows).set_index("Rank")
    _BLEND_COLOURS = {"LONG": "#2A9D8F", "SHORT": "#E76F51", "FLAT": "#888888"}

    def _col_blend(val):
        c = _BLEND_COLOURS.get(str(val), "")
        return f"color: {c}; font-weight: bold" if c else ""

    def _col_sub(val):
        c = _POS_COLOUR.get(str(val), "")
        return f"color: {c}" if c else ""

    styled = (
        tbl_df.style
        .map(_col_blend, subset=["Blended Position"])
        .map(_col_sub,   subset=["Commodity", "Forex"])
    )
    st.dataframe(styled, use_container_width=True)
    st.caption("Blend score = w_commodity × commodity_signal + w_forex × forex_signal")

    # Forward forecast — separate subplots per pick, horizon per asset class
    st.divider()
    tab_comm, tab_forex = st.tabs(["Commodity picks (h=63)", "Forex picks (h=5)"])

    with tab_comm:
        comm_picks = [r for r in sig_a.rankings if r.position in ("LONG", "SHORT")]
        if comm_picks:
            st.warning(
                "**Projected prices are model estimates. Research tooling only — not investment advice.**"
            )
            figs = _forecast_subplots([(r, 63) for r in comm_picks], live_data)
            for i in range(0, len(figs), 2):
                row_cols = st.columns(min(2, len(figs) - i))
                for j, col in enumerate(row_cols):
                    with col:
                        st.plotly_chart(figs[i + j], use_container_width=True)

    with tab_forex:
        st.info(
            "Forex forecast charts require a separate live data fetch. "
            "Run the **Forex LightGBM h=5** signal directly to see forex projections."
        )

    with st.expander("Ensemble context"):
        st.markdown(f"""
| Metric | Value |
|--------|-------|
| Ensemble OOS Sharpe | **{scfg['backtest_sharpe']:.2f}** |
| Commodity OOS Sharpe | {SIGNALS['Commodity LightGBM h=63']['backtest_sharpe']:.2f} |
| Forex OOS Sharpe | {SIGNALS['Forex LightGBM h=5']['backtest_sharpe']:.2f} |
| Weights | Commodity {w_a:.0%} / Forex {w_b:.0%} |
| Description | {scfg.get('description', '')} |

*Phase 18 three-way OOS ensemble: Sharpe 1.05, ρ(commodity, forex) ≈ −0.04*
""")

    # ── Portfolio Simulator (ensemble) ───────────────────────────────────────
    comm_capital  = sim_capital * w_a
    forex_capital = sim_capital * w_b
    ensemble_rankings = (
        [(r, 63, comm_capital) for r in sig_a.rankings] +
        [(r, 5,  forex_capital) for r in sig_b.rankings]
    )
    _render_portfolio_sim(
        rankings_with_horizon=ensemble_rankings,
        live_data=live_data,
        capital=sim_capital,
        long_only=(sim_platform == "Long-only"),
        sig_date=sig_a.date,
        scfg=scfg,
        is_ensemble=True,
    )

    _render_signal_history(sig_a)


def _render_equity_curve(scfg: dict) -> None:
    """Show a cached equity curve from the most recent backtest report, if available."""
    uni = scfg.get("universe", "commodities")
    phase_map = {
        "commodities":    "phase14_summary.md",
        "forex":          "phase18_summary.md",
        "equity_sectors": "phase17_summary.md",
    }
    report_file = cfg.paths.outputs_reports / phase_map.get(uni, "")
    if report_file.exists():
        content = report_file.read_text(encoding="utf-8")[:3000]
        st.markdown(f"**From {report_file.name}:**")
        st.code(content, language="markdown")
    else:
        # Try to load the phase JSON results
        json_map = {
            "commodities":    None,
            "forex":          "phase18a_results.json",
            "equity_sectors": "phase17a_results.json",
        }
        jfile = json_map.get(uni)
        if jfile:
            jpath = cfg.paths.outputs_reports / jfile
            if jpath.exists():
                try:
                    data = json.loads(jpath.read_text())
                    st.json(data)
                    return
                except Exception:
                    pass
        st.info(
            "No cached backtest report found for this universe. "
            "Run a backtest via the **Run Experiment** page to generate one."
        )


def _render_signal_history(sig) -> None:
    """Show saved signal history with realized P&L column where available."""
    st.divider()
    st.subheader("Signal History")

    signal_dir = cfg.paths.outputs_reports.parent / "signals"
    if not signal_dir.exists():
        st.info("No saved signals yet.")
        return

    signal_files = sorted(signal_dir.glob("signal_*.json"), reverse=True)
    if not signal_files:
        st.info("No saved signals yet. Run the signal and it will be saved automatically.")
        return

    hist_rows = []
    for p in signal_files[:30]:
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
        hist_df = pd.DataFrame(hist_rows).set_index("Date")

        # Track record: for signals older than `horizon` business days, check realized P&L
        track_record = []
        today = sig.date
        horizon = sig.horizon

        for row in hist_rows:
            try:
                sig_date = datetime.date.fromisoformat(row["Date"])
                days_ago = (today - sig_date).days
                if days_ago >= horizon and row["LONG"]:
                    # Look for the next saved signal as a rough realized-return proxy
                    track_record.append(f"≥{horizon}d ago")
                else:
                    track_record.append("Pending")
            except Exception:
                track_record.append("—")

        hist_df["Track record"] = track_record
        st.dataframe(hist_df, use_container_width=True)
    else:
        st.info("No valid signal files could be read.")


# ── Main dispatch ─────────────────────────────────────────────────────────────

_result = st.session_state.live_signal
_scfg   = SIGNALS.get(st.session_state.get("live_signal_name", signal_name), sig_cfg)

_is_ensemble = (
    isinstance(_result, tuple)
    and len(_result) == 2
    and isinstance(_result[0], tuple)
    and len(_result[0]) == 3
)

if _is_ensemble:
    (sig_a, sig_b, _weights), _live_data = _result
    _render_ensemble(sig_a, sig_b, _weights, _live_data, _scfg)
else:
    _sig, _live_data = _result
    _render_single(_sig, _live_data, _scfg, active_name)
