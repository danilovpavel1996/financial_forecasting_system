"""Plotly chart builders for the research dashboard.

All functions return plotly Figure objects — the Streamlit layer calls
st.plotly_chart(fig). No Streamlit imports here; this is pure chart logic.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Colour palette — works on both light and dark Streamlit themes
_COLOURS = [
    "#4C9BE8",  # blue
    "#F4A261",  # orange
    "#2A9D8F",  # teal
    "#E76F51",  # red-orange
    "#A8DADC",  # light blue
    "#E9C46A",  # yellow
    "#264653",  # dark slate
]

_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(size=12),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=40, r=20, t=40, b=40),
    xaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.2)"),
    yaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.2)"),
)


# ── Backtest charts ───────────────────────────────────────────────────────────

def equity_curve(results: dict, skip_flat: bool = True, horizon: int = 1) -> go.Figure:
    """Cumulative equity curve for all models in results dict."""
    fig = go.Figure()
    for i, (name, r) in enumerate(results.items()):
        pnl = r.ls_pnl_ts.dropna()
        if skip_flat and pnl.abs().sum() < 1e-10:
            continue
        # Subsample to non-overlapping periods when horizon > 1 to avoid
        # compounding overlapping returns, which inflates the curve by ~horizon.
        if horizon > 1:
            pnl = pnl.iloc[::horizon]
        cum = (1 + pnl).cumprod()
        colour = _COLOURS[i % len(_COLOURS)]
        fig.add_trace(go.Scatter(
            x=cum.index, y=cum.values,
            mode="lines", name=name,
            line=dict(color=colour, width=2),
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:.4f}<extra>" + name + "</extra>",
        ))
    fig.add_hline(y=1.0, line_dash="dot", line_color="rgba(128,128,128,0.5)")
    fig.update_layout(
        **_LAYOUT,
        title="Cumulative Equity (net of costs)",
        yaxis_title="Cumulative return (1 = start)",
        xaxis_title="",
    )
    return fig


def rolling_ric_chart(results: dict, window: int = 63, skip_flat: bool = True) -> go.Figure:
    """Rolling mean CS-RIC (Spearman rank IC) per model."""
    fig = go.Figure()
    for i, (name, r) in enumerate(results.items()):
        ric = r.cs_ric_ts.dropna()
        if skip_flat and ric.abs().sum() < 1e-10:
            continue
        rolled = ric.rolling(window, min_periods=window // 2).mean()
        colour = _COLOURS[i % len(_COLOURS)]
        fig.add_trace(go.Scatter(
            x=rolled.index, y=rolled.values,
            mode="lines", name=name,
            line=dict(color=colour, width=2),
            hovertemplate="%{x|%Y-%m-%d}<br>RIC=%{y:.4f}<extra>" + name + "</extra>",
        ))
    fig.add_hline(y=0.0, line_dash="dot", line_color="rgba(128,128,128,0.5)")
    fig.update_layout(
        **_LAYOUT,
        title=f"Rolling {window}-day Mean CS-RIC",
        yaxis_title="CS-RIC (rolling mean)",
        xaxis_title="",
    )
    return fig


def vol_scale_chart(results: dict, skip_flat: bool = True) -> go.Figure:
    """Vol scale factor over time for vol-targeted runs."""
    fig = go.Figure()
    has_data = False
    for i, (name, r) in enumerate(results.items()):
        if r.vol_scale_ts is None:
            continue
        colour = _COLOURS[i % len(_COLOURS)]
        fig.add_trace(go.Scatter(
            x=r.vol_scale_ts.index, y=r.vol_scale_ts.values,
            mode="lines", name=name,
            line=dict(color=colour, width=1.5),
            hovertemplate="%{x|%Y-%m-%d}<br>scale=%{y:.3f}<extra>" + name + "</extra>",
        ))
        has_data = True
    fig.add_hline(y=1.0, line_dash="dot", line_color="rgba(128,128,128,0.5)")
    fig.add_hline(y=2.0, line_dash="dash", line_color="rgba(200,80,80,0.4)")
    fig.update_layout(
        **_LAYOUT,
        title="Position Scale Factor (vol targeting)",
        yaxis_title="Scale (1 = unscaled, 2 = max leverage)",
        xaxis_title="",
    )
    return fig if has_data else None


# ── Price / returns charts ────────────────────────────────────────────────────

def price_chart(prices: dict, ticker: str, log_scale: bool = True) -> go.Figure:
    """Closing price history for one ticker."""
    if ticker not in prices:
        return go.Figure()
    close = prices[ticker]["Close"].dropna()
    fig = go.Figure(go.Scatter(
        x=close.index, y=close.values,
        mode="lines", name=ticker,
        line=dict(color=_COLOURS[0], width=1.5),
        hovertemplate="%{x|%Y-%m-%d}<br>$%{y:,.2f}<extra></extra>",
    ))
    fig.update_layout(
        **_LAYOUT,
        title=f"{ticker} — Daily Close",
        yaxis_type="log" if log_scale else "linear",
        yaxis_title="Price (log scale)" if log_scale else "Price",
        xaxis_title="",
    )
    return fig


def correlation_heatmap(prices: dict, tickers: list[str]) -> go.Figure:
    """Pearson correlation heatmap of daily log-returns."""
    rets = {}
    for t in tickers:
        if t not in prices:
            continue
        close = prices[t]["Close"].dropna()
        rets[t] = np.log(close / close.shift(1)).dropna()

    if len(rets) < 2:
        return go.Figure()

    df = pd.DataFrame(rets).dropna()
    corr = df.corr()

    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        colorscale="RdBu_r",
        zmid=0,
        zmin=-1, zmax=1,
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        colorbar=dict(title="ρ"),
        hovertemplate="%{y} / %{x}<br>ρ = %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(**_LAYOUT, title="Daily Log-Return Correlations")
    return fig


# ── COT charts ────────────────────────────────────────────────────────────────

def cot_positioning_chart(cot_weekly: pd.DataFrame, ticker: str) -> go.Figure:
    """COT mm_net and mm_net_pct for one commodity."""
    if cot_weekly is None or cot_weekly.empty:
        return go.Figure()

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.6, 0.4],
        subplot_titles=["Net Managed Money Positions", "52-Week Percentile"],
        vertical_spacing=0.08,
    )

    # mm_net (contracts)
    mm_net = cot_weekly["mm_long"] - cot_weekly["mm_short"]
    fig.add_trace(go.Bar(
        x=mm_net.index, y=mm_net.values,
        name="MM Net", marker_color=_COLOURS[0], opacity=0.8,
        hovertemplate="%{x|%Y-%m-%d}<br>MM Net=%{y:,.0f}<extra></extra>",
    ), row=1, col=1)

    # mm_net_pct (rolling percentile — compute it here for display)
    mm_pct = _rolling_pct(mm_net)
    fig.add_trace(go.Scatter(
        x=mm_pct.index, y=mm_pct.values,
        mode="lines", name="52w %ile",
        line=dict(color=_COLOURS[1], width=2),
        hovertemplate="%{x|%Y-%m-%d}<br>Pctile=%{y:.2f}<extra></extra>",
    ), row=2, col=1)
    fig.add_hline(y=0.8, line_dash="dash", line_color="rgba(200,80,80,0.4)", row=2, col=1)
    fig.add_hline(y=0.2, line_dash="dash", line_color="rgba(80,200,80,0.4)", row=2, col=1)

    fig.update_layout(
        **_LAYOUT,
        title=f"{ticker} — COT Managed Money Positioning",
        showlegend=True,
    )
    fig.update_yaxes(title_text="Net contracts", row=1, col=1,
                     gridcolor="rgba(128,128,128,0.2)")
    fig.update_yaxes(title_text="Percentile", row=2, col=1, range=[0, 1],
                     gridcolor="rgba(128,128,128,0.2)")
    return fig


# ── Vol monitor charts ────────────────────────────────────────────────────────

def ewma_vol_chart(ticker: str, returns: pd.Series, ewma_vol: pd.Series,
                   realized: pd.Series) -> go.Figure:
    """EWMA and trailing realized vol time series for one commodity."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=realized.index, y=(realized * 100).values,
        mode="lines", name="Realized vol (21d)",
        line=dict(color=_COLOURS[2], width=1.5, dash="dot"),
        hovertemplate="%{x|%Y-%m-%d}<br>%{y:.1f}%<extra>Realized</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=ewma_vol.index, y=(ewma_vol * 100).values,
        mode="lines", name="EWMA vol",
        line=dict(color=_COLOURS[0], width=2),
        hovertemplate="%{x|%Y-%m-%d}<br>%{y:.1f}%<extra>EWMA</extra>",
    ))
    fig.update_layout(
        **_LAYOUT,
        title=f"{ticker} — Annualised Volatility",
        yaxis_title="Ann. vol (%)",
        xaxis_title="",
    )
    return fig


# ── Helpers ───────────────────────────────────────────────────────────────────

def _rolling_pct(s: pd.Series, window: int = 52) -> pd.Series:
    """52-observation rolling min-max percentile (same as cot_features.py logic)."""
    roll_min = s.rolling(window, min_periods=window // 2).min()
    roll_max = s.rolling(window, min_periods=window // 2).max()
    denom = (roll_max - roll_min).replace(0, np.nan)
    return ((s - roll_min) / denom).clip(0, 1)


def build_verdict(results: dict, horizon: int, vol_target: Optional[float]) -> str:
    """One-paragraph auto-generated text verdict on backtest results."""
    finite_results = {
        name: r for name, r in results.items()
        if np.isfinite(r.ls_sharpe) and abs(r.turnover) > 1e-6
    }
    if not finite_results:
        return "No models produced tradeable signals (all positions flat)."

    best_name = max(finite_results, key=lambda n: finite_results[n].ls_sharpe)
    best = finite_results[best_name]

    vt_note = (
        f" with {vol_target:.0%} vol target"
        if vol_target is not None else ""
    )
    ric_note = (
        f"mean CS-RIC {best.mean_cs_ric:+.4f} "
        f"(stability {best.cs_ric_stability:.1%})"
    )
    dd_note = f"max drawdown {best.ls_max_dd:.1%}"
    sharpe_note = f"net Sharpe {best.ls_sharpe:.2f}"

    if best.ls_sharpe > 0.5:
        signal = "a moderate positive signal"
    elif best.ls_sharpe > 0:
        signal = "a weak positive signal"
    else:
        signal = "no positive signal"

    return (
        f"Best model at horizon={horizon}{vt_note}: **{best_name}** with {sharpe_note}, "
        f"{ric_note}, {dd_note}, turnover {best.turnover:.2f}. "
        f"The cross-sectional ranking shows {signal}. "
        f"Results are research outputs only — not investment advice."
    )


def style_comparison_table(df: pd.DataFrame, leakage_ric: float = 0.15,
                            leakage_sharpe: float = 2.5) -> pd.DataFrame.style:
    """Apply highlighting: best Sharpe in green, leakage cells in red."""
    def _highlight(col):
        if col.name == "Sharpe_net":
            best_val = col[col.notna()].max() if col.notna().any() else None
            return [
                "background-color: rgba(42,157,143,0.35)"
                if (v == best_val and best_val is not None and best_val > 0) else ""
                for v in col
            ]
        return [""] * len(col)

    def _flag_leakage(val):
        if not isinstance(val, (int, float)) or not np.isfinite(float(val)):
            return ""
        f = float(val)
        if abs(f) > leakage_ric:
            return "color: #E76F51; font-weight: bold"
        if f > leakage_sharpe:
            return "color: #E76F51; font-weight: bold"
        return ""

    styled = df.style.apply(_highlight, axis=0)
    for col in df.columns:
        if col in ("mean_CS_RIC", "Sharpe_net"):
            styled = styled.applymap(_flag_leakage, subset=[col])
    return styled
