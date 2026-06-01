"""Phase 18C — Three-way ensemble: commodity + equity + forex.

Approach:
  1. Run commodity h=63 LightGBM (no COT, embargo=63)  → Sharpe ~0.79
  2. Run equity sector h=63 LightGBM PredAvg21 (embargo=63)  → Sharpe ~0.48
  3. Run forex best model at best horizon (from Phase 18 sweep)
  4. Align all three P&L series to common date range
  5. Report pairwise correlations (3 pairs)
  6. Sweep equal weights 33/33/33 and report combined Sharpe
  7. Compare to two-way ensemble (Phase 18A)

Usage
-----
    python scripts/run_three_way_ensemble.py [--refresh]
    python scripts/run_three_way_ensemble.py --forex-horizon 63 --forex-model LightGBM_B3

Output
------
    Prints correlation table and weight sweep to stdout.
    Saves phase18_three_way.md to outputs/reports/.
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
from src.pipeline_ranking_forex import run_forex_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_three_way_ensemble")

TRADING_DAYS_PER_YEAR = 252
SUBSAMPLE = 63

TWO_WAY_BEST_SHARPE = 0.97   # Phase 17/18A two-way best (updated after Part A)

WEIGHT_COMBOS_3 = [
    (1.00, 0.00, 0.00),
    (0.00, 1.00, 0.00),
    (0.00, 0.00, 1.00),
    (0.50, 0.50, 0.00),
    (0.50, 0.00, 0.50),
    (0.00, 0.50, 0.50),
    (1/3,  1/3,  1/3),
]


def _sharpe_stats(pnl: np.ndarray, subsample: int = 1) -> tuple[float, float, float]:
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
    parser = argparse.ArgumentParser(description="Phase 18C three-way ensemble")
    parser.add_argument("--refresh", action="store_true", help="Force re-fetch data")
    parser.add_argument("--forex-horizon", type=int, default=63,
                        help="Horizon for forex leg (default: 63)")
    parser.add_argument("--forex-model", choices=["MeanReversion", "LightGBM", "LambdaMART", "LightGBM_B3"],
                        default="LightGBM_B3",
                        help="Forex model (default: LightGBM_B3 = LightGBM + PredAvg21)")
    args = parser.parse_args()

    cfg = load_config()

    forex_pred_avg = 21 if args.forex_model == "LightGBM_B3" else 1
    forex_model_base = "LightGBM" if args.forex_model == "LightGBM_B3" else args.forex_model

    # ── 1. Commodity h=63 LightGBM ───────────────────────────────────────────
    logger.info("Running commodity h=63 LightGBM (no COT, embargo=63) …")
    t0 = time.time()
    results_comm = run_ranking_pipeline(
        cfg, horizon=63, force_refresh=args.refresh,
        use_cot=False, embargo=63, model_names=["LightGBM"],
    )
    logger.info("Commodity done in %.1fs", time.time() - t0)
    res_comm = results_comm["LightGBM"]
    pnl_comm = res_comm.ls_pnl_ts.copy()
    logger.info("Commodity Sharpe=%.2f", res_comm.ls_sharpe)

    # ── 2. Equity h=63 LightGBM PredAvg21 ───────────────────────────────────
    logger.info("Running equity h=63 LightGBM PredAvg21 (embargo=63) …")
    t0 = time.time()
    results_equity = run_sectors_pipeline(
        cfg, horizon=63, force_refresh=False,
        embargo=63, model_names=["LightGBM"], pred_avg_window=21,
    )
    logger.info("Equity done in %.1fs", time.time() - t0)
    res_equity = results_equity["LightGBM"]
    pnl_equity = res_equity.ls_pnl_ts.copy()
    logger.info("Equity Sharpe=%.2f", res_equity.ls_sharpe)

    # ── 3. Forex best model ───────────────────────────────────────────────────
    logger.info(
        "Running forex h=%d %s (pred_avg=%d, embargo=%d) …",
        args.forex_horizon, args.forex_model, forex_pred_avg, args.forex_horizon,
    )
    t0 = time.time()
    results_forex = run_forex_pipeline(
        cfg, horizon=args.forex_horizon, force_refresh=False,
        embargo=args.forex_horizon,
        model_names=[forex_model_base],
        pred_avg_window=forex_pred_avg,
    )
    logger.info("Forex done in %.1fs", time.time() - t0)
    res_forex = results_forex[forex_model_base]
    pnl_forex = res_forex.ls_pnl_ts.copy()
    logger.info("Forex Sharpe=%.2f", res_forex.ls_sharpe)

    # ── 4. Align to common date range ────────────────────────────────────────
    common_idx = pnl_comm.index.intersection(pnl_equity.index).intersection(pnl_forex.index)
    logger.info(
        "Common period: %s → %s  (%d days)",
        common_idx.min().date(), common_idx.max().date(), len(common_idx),
    )

    pnl_c = pnl_comm.loc[common_idx].values.astype(float)
    pnl_e = pnl_equity.loc[common_idx].values.astype(float)
    pnl_f = pnl_forex.loc[common_idx].values.astype(float)

    # ── 5. Pairwise correlations ──────────────────────────────────────────────
    mask = np.isfinite(pnl_c) & np.isfinite(pnl_e) & np.isfinite(pnl_f)
    corr_ce = float(np.corrcoef(pnl_c[mask], pnl_e[mask])[0, 1])
    corr_cf = float(np.corrcoef(pnl_c[mask], pnl_f[mask])[0, 1])
    corr_ef = float(np.corrcoef(pnl_e[mask], pnl_f[mask])[0, 1])
    logger.info(
        "Pairwise P&L correlations — comm/eq: %.4f  comm/fx: %.4f  eq/fx: %.4f",
        corr_ce, corr_cf, corr_ef,
    )

    # ── 6. Weight sweep ───────────────────────────────────────────────────────
    rows = []
    for w_c, w_e, w_f in WEIGHT_COMBOS_3:
        pnl_blend = w_c * pnl_c + w_e * pnl_e + w_f * pnl_f
        sharpe_raw, _, _     = _sharpe_stats(pnl_blend, subsample=1)
        sharpe_63, ann_ret, max_dd = _sharpe_stats(pnl_blend, subsample=SUBSAMPLE)
        label = f"{int(round(w_c*100))}/{int(round(w_e*100))}/{int(round(w_f*100))}"
        rows.append({
            "weights (comm/eq/fx)": label,
            "Sharpe_raw":   sharpe_raw,
            "Sharpe_sub63": sharpe_63,
            "ann_ret":      ann_ret,
            "max_dd":       max_dd,
        })

    sweep_df = pd.DataFrame(rows).set_index("weights (comm/eq/fx)")

    # ── Print results ─────────────────────────────────────────────────────────
    print("\n" + "=" * 75)
    print("  PHASE 18C — Three-Way Ensemble  (commodity + equity + forex)")
    print("=" * 75)
    print(f"\n  Commodity h=63 LightGBM Sharpe (sub63):         {_fmt(res_comm.ls_sharpe)}")
    print(f"  Equity h=63 LightGBM PredAvg21 Sharpe (sub63): {_fmt(res_equity.ls_sharpe)}")
    print(f"  Forex h={args.forex_horizon} {args.forex_model} Sharpe (sub{SUBSAMPLE}): "
          f"{_fmt(res_forex.ls_sharpe)}")
    print(f"\n  Pairwise P&L correlations:")
    print(f"    Commodity vs Equity:  {corr_ce:+.4f}")
    print(f"    Commodity vs Forex:   {corr_cf:+.4f}")
    print(f"    Equity vs Forex:      {corr_ef:+.4f}")

    print(f"\n  Common OOS period: {common_idx.min().date()} → {common_idx.max().date()}"
          f"  ({len(common_idx)} days)\n")

    print("  Weight Sweep Table (comm/eq/fx):")
    print("  " + "-" * 70)
    print(f"  {'weights':>16}  {'Sharpe_raw*':>11}  {'Sharpe_sub63':>12}  {'ann_ret':>8}  {'max_dd':>8}")
    print("  " + "-" * 70)
    for _, row in sweep_df.iterrows():
        print(f"  {row.name:>16}  {_fmt(row['Sharpe_raw']):>11}  "
              f"{_fmt(row['Sharpe_sub63']):>12}  "
              f"{_fmt(row['ann_ret'], 4):>8}  "
              f"{_fmt(row['max_dd'], 4):>8}")
    print("  " + "-" * 70)
    print("  * raw daily Sharpe inflated by overlapping h=63 returns.")
    print()

    valid_sub = sweep_df["Sharpe_sub63"].dropna()
    if len(valid_sub):
        best_w = valid_sub.idxmax()
        best_s = valid_sub.max()
        print(f"  Best blend (sub63 Sharpe):  {best_w}  → Sharpe={_fmt(best_s)}")
        if best_s > TWO_WAY_BEST_SHARPE:
            print(f"  *** Beats two-way ensemble ({TWO_WAY_BEST_SHARPE}) — UPDATE LIVE SIGNAL ***")
        else:
            print(f"  Does NOT beat two-way ensemble ({TWO_WAY_BEST_SHARPE}).")
    print()

    # ── Save JSON ─────────────────────────────────────────────────────────────
    out_dir = cfg.paths.outputs_reports
    out_dir.mkdir(parents=True, exist_ok=True)

    results_data = {
        "comm_sharpe":   float(res_comm.ls_sharpe),
        "equity_sharpe": float(res_equity.ls_sharpe),
        "forex_sharpe":  float(res_forex.ls_sharpe),
        "forex_horizon": args.forex_horizon,
        "forex_model":   args.forex_model,
        "corr_comm_equity": corr_ce,
        "corr_comm_forex":  corr_cf,
        "corr_equity_forex": corr_ef,
        "common_start": str(common_idx.min().date()),
        "common_end":   str(common_idx.max().date()),
        "common_n":     int(len(common_idx)),
        "sweep": sweep_df.reset_index().to_dict(orient="records"),
        "two_way_best_sharpe": TWO_WAY_BEST_SHARPE,
    }
    json_path = out_dir / "phase18c_results.json"
    json_path.write_text(json.dumps(results_data, indent=2), encoding="utf-8")
    logger.info("Results saved → %s", json_path)

    _write_three_way_report(cfg, results_data, sweep_df, args)


def _write_three_way_report(cfg, data: dict, sweep_df: pd.DataFrame, args) -> None:
    out_dir = cfg.paths.outputs_reports
    path = out_dir / "phase18_three_way.md"

    best_w = sweep_df["Sharpe_sub63"].idxmax() if sweep_df["Sharpe_sub63"].notna().any() else "N/A"
    best_s = float(sweep_df["Sharpe_sub63"].max()) if sweep_df["Sharpe_sub63"].notna().any() else float("nan")
    beats = np.isfinite(best_s) and best_s > data["two_way_best_sharpe"]

    tbl_rows = [
        "| weights (comm/eq/fx) | Sharpe_raw† | Sharpe_sub63 | ann_ret | max_dd |",
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
        "# Phase 18C — Three-Way Ensemble Results",
        "",
        "## Individual signal performance",
        "",
        "| Signal | Sharpe (sub63) | Horizon | Asset class |",
        "| --- | --- | --- | --- |",
        f"| Commodity LightGBM | {_fmt(data['comm_sharpe'])} | 63d | Precious metals futures |",
        f"| Equity LightGBM PredAvg21 | {_fmt(data['equity_sharpe'])} | 63d | SPDR sector ETFs |",
        f"| Forex {data['forex_model']} | {_fmt(data['forex_sharpe'])} | {data['forex_horizon']}d | Major USD forex pairs |",
        "",
        "## Pairwise P&L correlations",
        "",
        "| Pair | Correlation |",
        "| --- | --- |",
        f"| Commodity vs Equity | {data['corr_comm_equity']:+.4f} |",
        f"| Commodity vs Forex  | {data['corr_comm_forex']:+.4f} |",
        f"| Equity vs Forex     | {data['corr_equity_forex']:+.4f} |",
        "",
        "## OOS alignment",
        "",
        f"- Common period: {data['common_start']} → {data['common_end']}  ({data['common_n']} days)",
        "",
        "## Weight sweep (Sharpe_sub63 = subsampled every 63 days)",
        "",
        tbl,
        "",
        "† Raw daily Sharpe inflated by overlapping h=63 returns; use Sharpe_sub63 for comparison.",
        "",
        "## Result",
        "",
        f"- Best blend (sub63): **{best_w}** → Sharpe={_fmt(best_s)}",
        f"- Two-way ensemble baseline (Phase 18A): {data['two_way_best_sharpe']}",
        f"- Beats two-way baseline: **{'YES' if beats else 'NO'}**",
        "",
        "---",
        "*Research tooling only — not investment advice.*",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Three-way report saved → %s", path)
    print(f"  Report saved → {path}\n")


if __name__ == "__main__":
    main()
