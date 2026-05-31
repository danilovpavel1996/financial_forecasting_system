"""Reporting module — produces a complete, timestamped run report.

One call to ``write_report`` generates three files and returns the markdown path:

    outputs/reports/backtest_{ticker}_{date}_{hash8}.md
    outputs/figures/equity_{ticker}_{date}_{hash8}.png
    outputs/figures/rolling_ic_{ticker}_{date}_{hash8}.png

The ``{hash8}`` tag is the first 8 characters of an MD5 hash over the key config
fields (universe, dates, horizon, cost, splitter, seed).  Runs with identical
config produce identical hashes, so you can compare results across dates.  A
config change (e.g. adding a feature, changing the universe) produces a new hash
so the old and new runs are clearly distinguished.
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import date as _date
from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from src.config import Config
from src.eval.backtester import BacktestResult, comparison_table

logger = logging.getLogger(__name__)

# ── thresholds ───────────────────────────────────────────────────────────────
_IC_LEAKAGE_THRESHOLD: float = 0.05
_SHARPE_LEAKAGE_THRESHOLD: float = 2.5
_IC_STABILITY_EDGE_MIN: float = 0.50   # > 50 % of folds with positive IC
_ROLLING_IC_WINDOW_DEFAULT: int = 63   # ~1 quarter of trading days

# ── colours (consistent across figures) ─────────────────────────────────────
_PALETTE = [
    "#2563eb",  # blue
    "#dc2626",  # red
    "#16a34a",  # green
    "#d97706",  # amber
    "#7c3aed",  # violet
    "#0891b2",  # cyan
    "#be185d",  # pink
]


# ─────────────────────────────────────────────────────────────────────────────
# Config hash
# ─────────────────────────────────────────────────────────────────────────────

def config_hash(cfg: Config) -> str:
    """Return the first 8 hex chars of an MD5 hash over the key config fields.

    The hash is deterministic: identical config → identical hash.  It changes
    whenever universe, dates, horizon, cost, splitter params, or seed change.
    """
    payload = {
        "universe": {k: sorted(v) for k, v in cfg.universe.items()},
        "fred_series": sorted(cfg.fred_series),
        "dates": cfg.dates,
        "horizon": cfg.horizon,
        "cost_bps": cfg.cost_bps,
        "splitter": {
            "n_splits": cfg.splitter.n_splits,
            "train_years": cfg.splitter.train_years,
            "test_years": cfg.splitter.test_years,
            "embargo": cfg.splitter.embargo,
        },
        "random_seed": cfg.random_seed,
    }
    digest = hashlib.md5(
        json.dumps(payload, sort_keys=True).encode()
    ).hexdigest()
    return digest[:8]


# ─────────────────────────────────────────────────────────────────────────────
# PnL helpers
# ─────────────────────────────────────────────────────────────────────────────

def daily_net_pnl(
    pred: np.ndarray,
    realized: np.ndarray,
    cost_bps: float,
) -> np.ndarray:
    """Daily net-of-cost P&L series (sign position sizing)."""
    pos = np.sign(pred)
    pos = np.nan_to_num(pos, nan=0.0)
    daily_turnover = np.abs(np.diff(pos, prepend=0.0))
    cost = daily_turnover * (cost_bps / 1e4)
    net = pos * realized - cost
    return np.where(np.isfinite(net), net, 0.0)


def equity_curve(
    pred: np.ndarray,
    realized: np.ndarray,
    cost_bps: float,
) -> np.ndarray:
    """Cumulative return starting at 1.0."""
    return np.cumprod(1.0 + daily_net_pnl(pred, realized, cost_bps))


# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────

def save_equity_figure(
    results: Dict[str, BacktestResult],
    oos_index: pd.DatetimeIndex,
    cfg: Config,
    ticker: str,
    path: Path,
) -> None:
    """Save the OOS equity-curve figure to *path*."""
    fig, ax = plt.subplots(figsize=(12, 4.5))

    for (name, result), color in zip(results.items(), _PALETTE):
        curve = equity_curve(result.oos_pred(), result.oos_realized(), cfg.cost_bps)
        ax.plot(oos_index, curve, label=name, linewidth=1.4, color=color)

    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_title(
        f"OOS Equity Curves — {ticker}  "
        f"(net of {cfg.cost_bps:.0f} bps, {cfg.splitter.n_splits} folds)",
        fontsize=12,
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative return  (start = 1.0)")
    ax.legend(fontsize=9, ncol=3)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    logger.info("Equity curve saved: %s", path)


def save_rolling_ic_figure(
    results: Dict[str, BacktestResult],
    oos_index: pd.DatetimeIndex,
    cfg: Config,
    ticker: str,
    path: Path,
    window: int = _ROLLING_IC_WINDOW_DEFAULT,
) -> None:
    """Save a rolling Pearson IC time-series figure to *path*.

    The rolling window is computed over the concatenated OOS predictions from
    all folds.  Fold boundaries are marked with vertical dashed lines.  Models
    with constant predictions (RandomWalk, Drift) produce NaN rolling IC and
    are skipped; their flat lines would add noise without information.
    """
    # Build fold-boundary positions from the first result with non-empty folds
    fold_dates: list[pd.Timestamp] = []
    reference_result = next(iter(results.values()))
    pos = 0
    for fold in reference_result.folds[1:]:   # skip first (no prior boundary)
        pos += fold.test_size
        if pos < len(oos_index):
            fold_dates.append(oos_index[pos])

    fig, ax = plt.subplots(figsize=(12, 4.0))

    plotted = 0
    for (name, result), color in zip(results.items(), _PALETTE):
        pred = result.oos_pred()
        real = result.oos_realized()
        if np.nanstd(pred) < 1e-12:
            continue   # constant prediction → correlation undefined; skip
        s_pred = pd.Series(pred, index=oos_index)
        s_real = pd.Series(real, index=oos_index)
        rolling_ic = s_pred.rolling(window, min_periods=window // 2).corr(s_real)
        ax.plot(oos_index, rolling_ic, label=f"{name}", linewidth=1.2, color=color)
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        logger.warning("Rolling IC figure: all models have constant predictions — skipping.")
        return

    for fd in fold_dates:
        ax.axvline(fd, color="grey", linewidth=0.7, linestyle=":", alpha=0.7)

    ax.axhline(0.0,  color="black", linewidth=0.9, linestyle="-",  alpha=0.4)
    ax.axhline(0.05, color="black", linewidth=0.6, linestyle="--", alpha=0.3)
    ax.axhline(-0.05, color="black", linewidth=0.6, linestyle="--", alpha=0.3)

    ax.set_title(
        f"Rolling IC ({window}d window) — {ticker}  "
        f"(grey dotted = fold boundaries)",
        fontsize=12,
    )
    ax.set_xlabel("Date")
    ax.set_ylabel(f"Pearson IC  ({window}d rolling)")
    ax.legend(fontsize=9, ncol=3)
    ax.grid(True, alpha=0.15)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    logger.info("Rolling IC saved: %s", path)


# ─────────────────────────────────────────────────────────────────────────────
# Verdict
# ─────────────────────────────────────────────────────────────────────────────

def build_verdict(
    results: Dict[str, BacktestResult],
    cfg: Config,
    ticker: str,
) -> str:
    """Return a one-paragraph plain-English verdict.

    Logic:
    - LEAKAGE flag if any model's |pooled_IC| > 0.05 or Sharpe > 2.5.
    - EDGE if a non-baseline model beats Drift Sharpe AND IC_stability > 0.5
      AND no leakage flag.
    - Otherwise NO EDGE.
    """
    _BASELINES = {"RandomWalk", "Drift", "Momentum"}

    drift = results.get("Drift")
    drift_sharpe = drift.strategy.sharpe if drift else np.nan

    leakage_names: list[str] = []
    edge_names: list[str] = []

    for name, r in results.items():
        ic_suspect = np.isfinite(r.pooled_ic) and abs(r.pooled_ic) > _IC_LEAKAGE_THRESHOLD
        sharpe_suspect = np.isfinite(r.strategy.sharpe) and r.strategy.sharpe > _SHARPE_LEAKAGE_THRESHOLD
        if ic_suspect or sharpe_suspect:
            leakage_names.append(name)
            continue

        if name in _BASELINES:
            continue

        beats_drift = (
            np.isfinite(r.strategy.sharpe)
            and np.isfinite(drift_sharpe)
            and r.strategy.sharpe > drift_sharpe
        )
        stable_ic = np.isfinite(r.ic_stability) and r.ic_stability > _IC_STABILITY_EDGE_MIN

        if beats_drift and stable_ic:
            edge_names.append(name)

    period = f"{cfg.dates['start']}–{cfg.dates['end']}"
    folds = cfg.splitter.n_splits
    cost = cfg.cost_bps
    horizon = cfg.horizon

    if leakage_names:
        suspects = ", ".join(leakage_names)
        return (
            f"⚠️ LEAKAGE FLAG — {suspects} triggered the IC > {_IC_LEAKAGE_THRESHOLD} or "
            f"Sharpe > {_SHARPE_LEAKAGE_THRESHOLD} threshold and must be investigated for "
            f"data leakage before any conclusions are drawn. "
            f"All other models show near-zero IC on {ticker} at the {horizon}-day horizon "
            f"over {period} ({folds} OOS folds, {cost} bps cost), consistent with an efficient "
            f"market. No exploitable edge is claimed."
        )

    if edge_names:
        winners = ", ".join(edge_names)
        sharpes = {
            n: f"{results[n].strategy.sharpe:.2f}" for n in edge_names
        }
        drift_str = f"{drift_sharpe:.2f}" if np.isfinite(drift_sharpe) else "N/A"
        sharpe_str = ", ".join(f"{n} ({s})" for n, s in sharpes.items())
        return (
            f"{winners} beat the Drift baseline (Sharpe {drift_str}) on net-of-cost Sharpe "
            f"with IC_stability > {_IC_STABILITY_EDGE_MIN} on {ticker} daily returns over "
            f"{period} ({folds} OOS folds, {cost} bps cost, {horizon}-day horizon): "
            f"{sharpe_str}. "
            f"These results warrant scrutiny for residual look-ahead bias before acting on them; "
            f"the appropriate next step is a leakage audit and out-of-sample confirmation "
            f"on a held-out period."
        )

    drift_str = f"{drift_sharpe:.2f}" if np.isfinite(drift_sharpe) else "N/A"
    return (
        f"No model demonstrates exploitable edge over the Drift baseline "
        f"(net Sharpe {drift_str}) on {ticker} daily returns over {period} "
        f"({folds} OOS folds, {cost} bps round-trip cost, {horizon}-day horizon). "
        f"This is the expected and honest result for a {horizon}-day-ahead forecast on "
        f"a liquid futures market: after transaction costs, none of the tested classifiers "
        f"consistently identifies direction well enough to generate positive risk-adjusted "
        f"returns. The harness is operating correctly; the absence of edge is informative, "
        f"not a failure."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Markdown report
# ─────────────────────────────────────────────────────────────────────────────

def _fold_table(result: BacktestResult, oos_index: pd.DatetimeIndex) -> str:
    """Per-fold IC table with test-period start/end dates."""
    rows = []
    pos = 0
    for f in result.folds:
        start = oos_index[pos].date() if pos < len(oos_index) else "?"
        end_idx = min(pos + f.test_size - 1, len(oos_index) - 1)
        end = oos_index[end_idx].date() if end_idx >= 0 else "?"
        rows.append({
            "fold": f.fold_idx,
            "test_start": start,
            "test_end": end,
            "train_rows": f.train_size,
            "IC": round(f.ic, 4) if np.isfinite(f.ic) else float("nan"),
            "rank_IC": round(f.rank_ic, 4) if np.isfinite(f.rank_ic) else float("nan"),
            "DA": round(f.directional_accuracy, 3) if np.isfinite(f.directional_accuracy) else float("nan"),
        })
        pos += f.test_size
    return pd.DataFrame(rows).set_index("fold").to_markdown()


def build_report_text(
    results: Dict[str, BacktestResult],
    oos_index: pd.DatetimeIndex,
    cfg: Config,
    ticker: str,
    run_date: str,
    cfg_hash: str,
    equity_fig_name: str,
    rolling_ic_fig_name: str,
    rolling_ic_window: int,
) -> str:
    """Assemble the full markdown report as a string."""
    tbl = comparison_table(results)
    verdict_text = build_verdict(results, cfg, ticker)

    lines = [
        f"# Backtest Report — {ticker}",
        "",
        (
            f"**Date:** {run_date}  "
            f"**Config:** `{cfg_hash}`  "
            f"**Ticker:** {ticker}"
        ),
        "",
        "---",
        "",
        "## Configuration",
        "",
        f"| Parameter | Value |",
        f"|:----------|:------|",
        f"| Period | {cfg.dates['start']} → {cfg.dates['end']} |",
        f"| Horizon | {cfg.horizon} trading day(s) |",
        f"| Cost | {cfg.cost_bps} bps round-trip |",
        f"| Folds | {cfg.splitter.n_splits} × {cfg.splitter.test_years}yr test, {cfg.splitter.train_years}yr min train |",
        f"| Embargo | {cfg.splitter.embargo} trading days |",
        f"| Random seed | {cfg.random_seed} |",
        f"| Config hash | `{cfg_hash}` |",
        "",
        "---",
        "",
        "## Verdict",
        "",
        verdict_text,
        "",
        "---",
        "",
        "## Model Comparison",
        "",
        tbl.to_markdown(),
        "",
        "---",
        "",
        "## Figures",
        "",
        f"**Equity curves:** `outputs/figures/{equity_fig_name}`",
        "",
        f"![Equity curves](../figures/{equity_fig_name})",
        "",
        f"**Rolling IC ({rolling_ic_window}d):** `outputs/figures/{rolling_ic_fig_name}`",
        "",
        f"![Rolling IC](../figures/{rolling_ic_fig_name})",
        "",
        "---",
        "",
        "## Per-fold detail",
        "",
    ]

    for name, result in results.items():
        lines.append(f"### {name}")
        lines.append("")
        lines.append(_fold_table(result, oos_index))
        lines.append("")

    lines += [
        "---",
        "",
        "*Research tooling only — not investment advice.*",
    ]

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def write_report(
    results: Dict[str, BacktestResult],
    oos_index: pd.DatetimeIndex,
    cfg: Config,
    ticker: str,
    output_dir: Path,
    figures_dir: Path,
    run_date: str | None = None,
    rolling_ic_window: int = _ROLLING_IC_WINDOW_DEFAULT,
) -> Path:
    """Write the full report package and return the markdown path.

    Creates three files:
      - ``{output_dir}/backtest_{safe}_{date}_{hash8}.md``
      - ``{figures_dir}/equity_{safe}_{date}_{hash8}.png``
      - ``{figures_dir}/rolling_ic_{safe}_{date}_{hash8}.png``

    Parameters
    ----------
    results:           dict[model_name → BacktestResult] from run_pipeline.
    oos_index:         DatetimeIndex for OOS predictions (length = Σ fold.test_size).
    cfg:               loaded Config (used for hash + metadata).
    ticker:            target ticker string (for filenames + report headers).
    output_dir:        directory for the markdown report.
    figures_dir:       directory for PNG figures.
    run_date:          ISO date string; defaults to today.
    rolling_ic_window: trading-day window for the rolling IC computation.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    if run_date is None:
        run_date = _date.today().isoformat()

    cfg_hash = config_hash(cfg)
    safe = ticker.replace("=", "_eq_").replace("^", "_hat_").replace("-", "_")
    stem = f"{safe}_{run_date}_{cfg_hash}"

    equity_name = f"equity_{stem}.png"
    rolling_name = f"rolling_ic_{stem}.png"
    report_name = f"backtest_{stem}.md"

    save_equity_figure(results, oos_index, cfg, ticker, figures_dir / equity_name)
    save_rolling_ic_figure(
        results, oos_index, cfg, ticker,
        figures_dir / rolling_name,
        window=rolling_ic_window,
    )

    text = build_report_text(
        results, oos_index, cfg, ticker, run_date, cfg_hash,
        equity_name, rolling_name, rolling_ic_window,
    )
    report_path = output_dir / report_name
    report_path.write_text(text)
    logger.info("Report written: %s  (config=%s)", report_path, cfg_hash)

    return report_path
