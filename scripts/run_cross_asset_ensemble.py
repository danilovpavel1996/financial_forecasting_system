"""Phase 17 Experiment A — Cross-asset ensemble: commodity h=63 LightGBM + equity h=63 MeanReversion.

Approach (mirrors run_ensemble.py from Phase 15):
  1. Run commodity h=63 LightGBM backtest (no COT, embargo=63)
  2. Run equity sector h=63 MeanReversion backtest (embargo=63)
  3. Align daily P&L to common date range
  4. Report correlation between the two P&L series
  5. Sweep weight combos: 100/0, 75/25, 50/50, 25/75, 0/100
  6. For each blend: Sharpe (subsampled every 63 days), ann_ret, max_dd

Usage
-----
    python scripts/run_cross_asset_ensemble.py [--refresh]

Output
------
    Prints weight-sweep table to stdout.
    Saves phase17_ensemble.md to outputs/reports/.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import pandas as pd

from src.config import load_config
from src.pipeline_ranking import run_ranking_pipeline
from src.pipeline_ranking_sectors import run_sectors_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_cross_asset_ensemble")

TRADING_DAYS_PER_YEAR = 252
SUBSAMPLE = 63  # non-overlapping period for h=63

WEIGHT_COMBOS = [
    (1.00, 0.00),
    (0.75, 0.25),
    (0.50, 0.50),
    (0.25, 0.75),
    (0.00, 1.00),
]

COMMODITY_BASELINE_SHARPE = 0.79  # Phase 14 best


def _sharpe_stats(pnl: np.ndarray, subsample: int = 1) -> tuple[float, float, float]:
    """Return (Sharpe, ann_ret, max_dd); subsample > 1 for non-overlapping."""
    s = pnl[::subsample]
    s = s[np.isfinite(s)]
    if len(s) < 5:
        return np.nan, np.nan, np.nan

    ppy = TRADING_DAYS_PER_YEAR / subsample
    ann_ret = float(np.mean(s) * ppy)
    ann_vol = float(np.std(s) * np.sqrt(ppy))
    sharpe = ann_ret / ann_vol if ann_vol > 1e-12 else np.nan

    curve = np.cumprod(1.0 + s)
    peak = np.maximum.accumulate(curve)
    max_dd = float(np.min(curve / peak - 1.0))

    return sharpe, ann_ret, max_dd


def _fmt(v: float, decimals: int = 2) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "N/A"
    return f"{v:.{decimals}f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 17A cross-asset ensemble")
    parser.add_argument("--refresh", action="store_true",
                        help="Force re-fetch data from network")
    args = parser.parse_args()

    cfg = load_config()

    # ── 1. Commodity h=63 LightGBM (no COT, embargo=63) ─────────────────────
    logger.info("Running commodity h=63 LightGBM (no COT, embargo=63) …")
    t0 = time.time()
    results_comm = run_ranking_pipeline(
        cfg,
        horizon=63,
        force_refresh=args.refresh,
        use_cot=False,
        embargo=63,
        model_names=["LightGBM"],
    )
    logger.info("Commodity done in %.1fs", time.time() - t0)

    res_comm = results_comm["LightGBM"]
    pnl_comm: pd.Series = res_comm.ls_pnl_ts.copy()
    logger.info(
        "Commodity P&L: %s → %s  (%d days)  Sharpe=%.2f",
        pnl_comm.index.min().date(), pnl_comm.index.max().date(),
        len(pnl_comm), res_comm.ls_sharpe,
    )

    # ── 2. Equity sector h=63 MeanReversion (embargo=63) ────────────────────
    logger.info("Running equity sector h=63 MeanReversion (embargo=63) …")
    t0 = time.time()
    results_equity = run_sectors_pipeline(
        cfg,
        horizon=63,
        force_refresh=False,
        embargo=63,
        model_names=["MeanReversion"],
    )
    logger.info("Equity done in %.1fs", time.time() - t0)

    res_equity = results_equity["MeanReversion"]
    pnl_equity: pd.Series = res_equity.ls_pnl_ts.copy()
    logger.info(
        "Equity P&L: %s → %s  (%d days)  Sharpe=%.2f",
        pnl_equity.index.min().date(), pnl_equity.index.max().date(),
        len(pnl_equity), res_equity.ls_sharpe,
    )

    # ── 3. Align to common date range ────────────────────────────────────────
    common_idx = pnl_comm.index.intersection(pnl_equity.index)
    n_dropped_comm   = len(pnl_comm)   - len(pnl_comm.loc[common_idx])
    n_dropped_equity = len(pnl_equity) - len(pnl_equity.loc[common_idx])

    pnl_c = pnl_comm.loc[common_idx].values.astype(float)
    pnl_e = pnl_equity.loc[common_idx].values.astype(float)

    logger.info(
        "Common period: %s → %s  (%d days) [dropped %d comm, %d equity]",
        common_idx.min().date(), common_idx.max().date(), len(common_idx),
        n_dropped_comm, n_dropped_equity,
    )

    # ── 4. Correlation ───────────────────────────────────────────────────────
    mask = np.isfinite(pnl_c) & np.isfinite(pnl_e)
    corr = float(np.corrcoef(pnl_c[mask], pnl_e[mask])[0, 1])
    logger.info("P&L correlation (commodity vs equity): %.4f", corr)

    if abs(corr) < 0.2:
        div_comment = "Very low — strong diversification benefit expected."
    elif abs(corr) < 0.4:
        div_comment = "Low — meaningful diversification benefit."
    elif abs(corr) < 0.6:
        div_comment = "Moderate — some diversification benefit."
    else:
        div_comment = "High — limited diversification benefit."

    # ── 5–6. Weight sweep ────────────────────────────────────────────────────
    rows = []
    for w_c, w_e in WEIGHT_COMBOS:
        pnl_blend = w_c * pnl_c + w_e * pnl_e

        sharpe_raw, _,       _      = _sharpe_stats(pnl_blend, subsample=1)
        sharpe_63, ann_ret, max_dd  = _sharpe_stats(pnl_blend, subsample=SUBSAMPLE)

        label = f"{int(w_c*100)}/{int(w_e*100)}"
        rows.append({
            "weights (comm/eq)": label,
            "Sharpe_raw":  sharpe_raw,
            "Sharpe_sub63": sharpe_63,
            "ann_ret":      ann_ret,
            "max_dd":       max_dd,
        })

    sweep_df = pd.DataFrame(rows).set_index("weights (comm/eq)")

    # ── Print results ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  PHASE 17A — Cross-Asset Ensemble  (commodity vs equity sectors)")
    print("=" * 70)
    print(f"\n  Commodity h=63 LightGBM Sharpe (sub63):  {_fmt(res_comm.ls_sharpe)}")
    print(f"  Equity h=63 MeanReversion Sharpe (sub63): {_fmt(res_equity.ls_sharpe)}")
    print(f"\n  P&L correlation: {corr:+.4f}  — {div_comment}")

    if np.isfinite(res_comm.ls_sharpe) and np.isfinite(res_equity.ls_sharpe):
        theo = np.sqrt(res_comm.ls_sharpe**2 + res_equity.ls_sharpe**2)
        print(f"  Theoretical Sharpe if ρ=0:  "
              f"√({_fmt(res_comm.ls_sharpe)}² + {_fmt(res_equity.ls_sharpe)}²) = {_fmt(theo)}")

    print(f"\n  Common OOS period: {common_idx.min().date()} → {common_idx.max().date()}"
          f"  ({len(common_idx)} days)\n")

    print("  Weight Sweep Table:")
    print("  " + "-" * 66)
    print(f"  {'weights':>12}  {'Sharpe_raw*':>11}  {'Sharpe_sub63':>12}  {'ann_ret':>8}  {'max_dd':>8}")
    print("  " + "-" * 66)
    for _, row in sweep_df.iterrows():
        print(f"  {row.name:>12}  {_fmt(row['Sharpe_raw']):>11}  "
              f"{_fmt(row['Sharpe_sub63']):>12}  "
              f"{_fmt(row['ann_ret'], 4):>8}  "
              f"{_fmt(row['max_dd'], 4):>8}")
    print("  " + "-" * 66)
    print("  * raw daily Sharpe is inflated due to overlapping h=63 returns.")
    print()

    valid_sub = sweep_df["Sharpe_sub63"].dropna()
    if len(valid_sub):
        best_w = valid_sub.idxmax()
        best_s = valid_sub.max()
        print(f"  Best blend (sub63 Sharpe):  {best_w}  → Sharpe={_fmt(best_s)}")
        if best_s > COMMODITY_BASELINE_SHARPE:
            print(f"  *** Beats commodity baseline ({COMMODITY_BASELINE_SHARPE}) — update live signal ***")
        else:
            print(f"  Does NOT beat commodity baseline ({COMMODITY_BASELINE_SHARPE}).")
    print()

    # ── Save results JSON for phase17_summary.md ─────────────────────────────
    out_dir = cfg.paths.outputs_reports
    out_dir.mkdir(parents=True, exist_ok=True)

    results_data = {
        "comm_sharpe": float(res_comm.ls_sharpe),
        "equity_sharpe": float(res_equity.ls_sharpe),
        "correlation": corr,
        "div_comment": div_comment,
        "common_start": str(common_idx.min().date()),
        "common_end":   str(common_idx.max().date()),
        "common_n":     int(len(common_idx)),
        "n_dropped_comm":   int(n_dropped_comm),
        "n_dropped_equity": int(n_dropped_equity),
        "sweep": sweep_df.reset_index().to_dict(orient="records"),
    }
    json_path = out_dir / "phase17a_results.json"
    json_path.write_text(json.dumps(results_data, indent=2), encoding="utf-8")
    logger.info("Results saved → %s", json_path)

    # ── Write standalone markdown report ─────────────────────────────────────
    _write_ensemble_report(cfg, results_data, sweep_df)


def _write_ensemble_report(cfg, data: dict, sweep_df: pd.DataFrame) -> None:
    """Write outputs/reports/phase17_ensemble.md."""
    out_dir = cfg.paths.outputs_reports
    path = out_dir / "phase17_ensemble.md"

    corr = data["correlation"]
    comm_sharpe = data["comm_sharpe"]
    equity_sharpe = data["equity_sharpe"]

    theo = (np.sqrt(comm_sharpe**2 + equity_sharpe**2)
            if np.isfinite(comm_sharpe) and np.isfinite(equity_sharpe)
            else float("nan"))

    valid = sweep_df["Sharpe_sub63"].dropna()
    best_w  = valid.idxmax() if len(valid) else "N/A"
    best_s  = valid.max()    if len(valid) else float("nan")
    beats   = np.isfinite(best_s) and best_s > COMMODITY_BASELINE_SHARPE

    tbl_rows = [
        "| weights (comm/eq) | Sharpe_raw† | Sharpe_sub63 | ann_ret | max_dd |",
        "| --- | --- | --- | --- | --- |",
    ]
    for _, row in sweep_df.iterrows():
        tbl_rows.append(
            f"| {row.name} "
            f"| {_fmt(row['Sharpe_raw'])} "
            f"| {_fmt(row['Sharpe_sub63'])} "
            f"| {_fmt(row['ann_ret'], 4)} "
            f"| {_fmt(row['max_dd'], 4)} |"
        )
    tbl = "\n".join(tbl_rows)

    lines = [
        "# Phase 17A — Cross-Asset Ensemble Results",
        "",
        "## Individual signal performance",
        "",
        "| Signal | Sharpe (sub63) | Horizon | Asset class |",
        "| --- | --- | --- | --- |",
        f"| Commodity LightGBM | {_fmt(comm_sharpe)} | 63d | Precious metals futures |",
        f"| Equity MeanReversion | {_fmt(equity_sharpe)} | 63d | SPDR sector ETFs |",
        "",
        "## P&L correlation",
        "",
        f"- Daily P&L correlation: **{corr:+.4f}**",
        f"- Assessment: {data['div_comment']}",
        f"- Theoretical Sharpe if ρ=0: √({_fmt(comm_sharpe)}² + {_fmt(equity_sharpe)}²)"
        f" = **{_fmt(theo)}**",
        "",
        "## OOS alignment",
        "",
        f"- Common period: {data['common_start']} → {data['common_end']}  ({data['common_n']} days)",
        f"- Dates dropped from commodity series: {data['n_dropped_comm']}",
        f"- Dates dropped from equity series: {data['n_dropped_equity']}",
        "",
        "## Weight sweep (Sharpe_sub63 = subsampled every 63 days)",
        "",
        tbl,
        "",
        "† Raw daily Sharpe is inflated by overlapping h=63 returns; use Sharpe_sub63 for comparison.",
        "",
        "## Result",
        "",
        f"- Best blend (sub63): **{best_w}** → Sharpe={_fmt(best_s)}",
        f"- Commodity baseline (Phase 14): {COMMODITY_BASELINE_SHARPE}",
        f"- Beats baseline: **{'YES' if beats else 'NO'}**",
        "",
        "---",
        "*Research tooling only — not investment advice.*",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Ensemble report saved → %s", path)
    print(f"  Report saved → {path}\n")


if __name__ == "__main__":
    main()
