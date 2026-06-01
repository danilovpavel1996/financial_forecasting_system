"""Phase 18 forex sweep: horizons × models (+ LightGBM B3 PredAvg21 variant).

Runs 10 configurations:
  horizons: [5, 21, 63]
  models:   [MeanReversion, LightGBM, LambdaMART]
  + LightGBM with pred_avg_window=21 at each horizon (B3 variant)

This gives 12 runs total (9 base + 3 B3 variants).

Embargo = horizon for each run.

Output: outputs/reports/phase18_forex_sweep.md

Usage
-----
    python scripts/run_sweep_forex.py
    python scripts/run_sweep_forex.py --refresh
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

import numpy as np

from src.config import load_config
from src.pipeline_ranking_forex import run_forex_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_sweep_forex")

_HORIZONS = [5, 21, 63]
_MODELS = ["MeanReversion", "LightGBM", "LambdaMART"]
_LEAKAGE_CSRIC = 0.15
_LEAKAGE_SHARPE = 2.5


def _fmt(v, decimals: int = 2) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "N/A"
    return f"{v:.{decimals}f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 18 forex sweep — 12 configurations")
    parser.add_argument("--refresh", action="store_true", help="Force re-fetch of data")
    args = parser.parse_args()

    cfg = load_config()

    # Results dict: (model_label, horizon) → (sharpe, cs_ric, turnover, pnl_ts)
    results: dict[tuple[str, int], tuple] = {}
    leakage_flags: list[str] = []

    # ── Base sweep: 9 configs ─────────────────────────────────────────────────
    for horizon in _HORIZONS:
        logger.info("=== Forex: horizon=%d  (embargo=%d) ===", horizon, horizon)
        t0 = time.time()

        try:
            run_results = run_forex_pipeline(
                cfg,
                horizon=horizon,
                force_refresh=args.refresh,
                embargo=horizon,
                model_names=_MODELS,
                pred_avg_window=1,
            )
        except Exception as exc:
            logger.error("Forex pipeline failed for h=%d: %s", horizon, exc)
            for m in _MODELS:
                results[(m, horizon)] = (float("nan"), float("nan"), float("nan"), None)
            continue

        logger.info("  Horizon=%d base finished in %.1fs", horizon, time.time() - t0)

        for model_name in _MODELS:
            r = run_results.get(model_name)
            if r is None:
                results[(model_name, horizon)] = (float("nan"), float("nan"), float("nan"), None)
                continue

            sharpe   = r.ls_sharpe
            cs_ric   = r.mean_cs_ric
            turnover = r.turnover
            results[(model_name, horizon)] = (sharpe, cs_ric, turnover, r.ls_pnl_ts)

            logger.info(
                "  %s (h=%d): Sharpe=%.2f  CS-RIC=%.4f  turnover=%.3f",
                model_name, horizon,
                sharpe if np.isfinite(sharpe) else float("nan"),
                cs_ric if np.isfinite(cs_ric) else float("nan"),
                turnover,
            )

            if np.isfinite(cs_ric) and abs(cs_ric) > _LEAKAGE_CSRIC:
                flag = f"{model_name} h={horizon}: CS-RIC={cs_ric:.4f}"
                leakage_flags.append(flag)
                logger.warning("LEAKAGE FLAG: %s", flag)
            if np.isfinite(sharpe) and sharpe > _LEAKAGE_SHARPE:
                flag = f"{model_name} h={horizon}: Sharpe={sharpe:.2f}"
                leakage_flags.append(flag)
                logger.warning("LEAKAGE FLAG: %s", flag)

    # ── B3 variant: LightGBM + pred_avg_window=21 at each horizon ────────────
    for horizon in _HORIZONS:
        logger.info("=== Forex B3 (PredAvg21): horizon=%d ===", horizon)
        t0 = time.time()

        try:
            run_b3 = run_forex_pipeline(
                cfg,
                horizon=horizon,
                force_refresh=False,
                embargo=horizon,
                model_names=["LightGBM"],
                pred_avg_window=21,
            )
        except Exception as exc:
            logger.error("Forex B3 pipeline failed for h=%d: %s", horizon, exc)
            results[(f"LightGBM_B3", horizon)] = (float("nan"), float("nan"), float("nan"), None)
            continue

        logger.info("  Horizon=%d B3 finished in %.1fs", horizon, time.time() - t0)

        r = run_b3.get("LightGBM")
        if r is None:
            results[("LightGBM_B3", horizon)] = (float("nan"), float("nan"), float("nan"), None)
            continue

        sharpe   = r.ls_sharpe
        cs_ric   = r.mean_cs_ric
        turnover = r.turnover
        results[("LightGBM_B3", horizon)] = (sharpe, cs_ric, turnover, r.ls_pnl_ts)

        logger.info(
            "  LightGBM_B3 (h=%d): Sharpe=%.2f  CS-RIC=%.4f  turnover=%.3f",
            horizon,
            sharpe if np.isfinite(sharpe) else float("nan"),
            cs_ric if np.isfinite(cs_ric) else float("nan"),
            turnover,
        )

        if np.isfinite(cs_ric) and abs(cs_ric) > _LEAKAGE_CSRIC:
            flag = f"LightGBM_B3 h={horizon}: CS-RIC={cs_ric:.4f}"
            leakage_flags.append(flag)
            logger.warning("LEAKAGE FLAG: %s", flag)
        if np.isfinite(sharpe) and sharpe > _LEAKAGE_SHARPE:
            flag = f"LightGBM_B3 h={horizon}: Sharpe={sharpe:.2f}"
            leakage_flags.append(flag)
            logger.warning("LEAKAGE FLAG: %s", flag)

    # ── Build sweep table ─────────────────────────────────────────────────────
    all_model_labels = _MODELS + ["LightGBM_B3"]
    col_headers = [f"h={h}" for h in _HORIZONS]

    sep    = "| " + " | ".join(["---"] * (len(col_headers) + 1)) + " |"
    header = "| Model | " + " | ".join(col_headers) + " |"

    sweep_rows = [header, sep]
    for model in all_model_labels:
        cells = []
        for h in _HORIZONS:
            sharpe, cs_ric, _, _ = results.get((model, h), (float("nan"), float("nan"), float("nan"), None))
            cells.append(f"{_fmt(sharpe, 2)} / {_fmt(cs_ric, 4)}")
        label = model if model != "LightGBM_B3" else "LightGBM PredAvg21"
        sweep_rows.append(f"| {label} | " + " | ".join(cells) + " |")

    sweep_table_md = "\n".join(sweep_rows)

    # ── Identify best config (by Sharpe) ─────────────────────────────────────
    best_sharpe = float("-inf")
    best_key: tuple | None = None
    best_pnl = None

    for (model, h), (sharpe, cs_ric, _, pnl_ts) in results.items():
        if np.isfinite(sharpe) and sharpe > best_sharpe:
            best_sharpe = sharpe
            best_key = (model, h)
            best_pnl = pnl_ts

    best_line = "None found"
    if best_key:
        bm, bh = best_key
        display = bm if bm != "LightGBM_B3" else "LightGBM PredAvg21"
        best_line = f"**{display}** at h={bh} — Sharpe {best_sharpe:.2f}"

    leakage_section = ""
    if leakage_flags:
        leakage_section = "\n\n## Leakage flags\n\n"
        leakage_section += "\n".join(f"- {f}" for f in leakage_flags)

    report = f"""# Phase 18 Forex Sweep Results

*Format: Sharpe (net, non-overlapping) / CS-RIC (mean OOS)*

*Embargo = horizon for each run. All results are out-of-sample walk-forward.*

*Universe: EURUSD=X, GBPUSD=X, USDJPY=X, AUDUSD=X, USDCAD=X, USDCHF=X, NZDUSD=X*

*No COT (not configured for forex). No carry proxy. No late-close lag (all pairs 5pm ET).*

*History start: {cfg.forex.get("start_date", "2005-01-01")}  |  End: {cfg.dates["end"]}*

---

## Forex Sweep Matrix

{sweep_table_md}

---

## Summary

**Forex best:** {best_line}

**Commodity reference (Phase 14):** LightGBM h=63 — Sharpe 0.79

**Equity reference (Phase 17/18):** LightGBM PredAvg21 h=63 — Sharpe 0.48
{leakage_section}
---

## Notes

- Sharpe uses non-overlapping subsampling every `horizon` steps to avoid autocorrelation inflation.
- CS-RIC is mean cross-sectional rank information coefficient across all OOS test dates.
- Embargo = horizon ensures forward return labels cannot leak across the train/test boundary.
- Costs charged at {cfg.cost_bps} bps round-trip on turnover.
- LightGBM PredAvg21 applies rolling 21-day mean to predictions before ranking (B3 variant).
- Quote convention: EURUSD/GBPUSD/AUDUSD/NZDUSD are foreign/USD; USDJPY/USDCAD/USDCHF are USD/foreign.
  No inversion — model learns each pair's behavior via one-hot encoding.

*Research tooling only — not investment advice.*
"""

    out_path = Path(cfg.paths.outputs_reports) / "phase18_forex_sweep.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    logger.info("Forex sweep results saved → %s", out_path)

    print("\n" + "=" * 70)
    print("PHASE 18 FOREX SWEEP COMPLETE")
    print("=" * 70)
    print(sweep_table_md)
    print(f"\nForex best: {best_line}")
    print(f"\nFull report → {out_path}")

    return best_sharpe, best_key, best_pnl


if __name__ == "__main__":
    main()
