"""Phase 13 systematic sweep: horizons × COT configs × models.

Runs all 18 combinations:
  horizons: [5, 21, 63]
  features: [no-COT, with-COT]
  models:   [MeanReversion, LightGBM, LambdaMART]

Embargo = horizon for each run (prevents label leakage).

Output: outputs/reports/phase13_sweep.md

Usage
-----
    python scripts/run_sweep.py
    python scripts/run_sweep.py --refresh    # force re-fetch data
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
from src.pipeline_ranking import run_ranking_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_sweep")

_HORIZONS = [5, 21, 63]
_MODELS = ["MeanReversion", "LightGBM", "LambdaMART"]
_LEAKAGE_CSRIC = 0.15
_LEAKAGE_SHARPE = 2.5


def _fmt(v: float, decimals: int = 2) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "N/A"
    return f"{v:.{decimals}f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 13 sweep: 18-configuration matrix")
    parser.add_argument("--refresh", action="store_true", help="Force re-fetch of data")
    args = parser.parse_args()

    cfg = load_config()

    # Results dict: (model, horizon, cot) → (sharpe, cs_ric)
    results: dict[tuple[str, int, bool], tuple[float, float]] = {}
    leakage_flags: list[str] = []

    total = len(_HORIZONS) * 2 * len(_MODELS)
    run_n = 0

    for horizon in _HORIZONS:
        for use_cot in [False, True]:
            cot_label = "COT" if use_cot else "no-COT"
            logger.info(
                "=== Dataset: horizon=%d  %s  (embargo=%d) ===",
                horizon, cot_label, horizon,
            )
            t0 = time.time()

            try:
                run_results = run_ranking_pipeline(
                    cfg,
                    horizon=horizon,
                    force_refresh=args.refresh,
                    use_cot=use_cot,
                    embargo=horizon,
                    model_names=_MODELS,
                )
            except Exception as exc:
                logger.error(
                    "Pipeline failed for h=%d %s: %s", horizon, cot_label, exc
                )
                for m in _MODELS:
                    results[(m, horizon, use_cot)] = (float("nan"), float("nan"))
                continue

            elapsed = time.time() - t0
            logger.info("  Dataset run finished in %.1fs", elapsed)

            for model_name in _MODELS:
                run_n += 1
                r = run_results.get(model_name)
                if r is None:
                    results[(model_name, horizon, use_cot)] = (float("nan"), float("nan"))
                    logger.warning("  %s: not found in results", model_name)
                    continue

                sharpe = r.ls_sharpe
                cs_ric = r.mean_cs_ric

                results[(model_name, horizon, use_cot)] = (sharpe, cs_ric)
                logger.info(
                    "  [%d/%d] %s (h=%d, %s): Sharpe=%.2f  CS-RIC=%.4f  turnover=%.3f",
                    run_n, total, model_name, horizon, cot_label,
                    sharpe if np.isfinite(sharpe) else float("nan"),
                    cs_ric if np.isfinite(cs_ric) else float("nan"),
                    r.turnover,
                )

                if np.isfinite(cs_ric) and abs(cs_ric) > _LEAKAGE_CSRIC:
                    flag = f"{model_name} h={horizon} {cot_label}: CS-RIC={cs_ric:.4f}"
                    leakage_flags.append(flag)
                    logger.warning("LEAKAGE FLAG: %s", flag)
                if np.isfinite(sharpe) and sharpe > _LEAKAGE_SHARPE:
                    flag = f"{model_name} h={horizon} {cot_label}: Sharpe={sharpe:.2f}"
                    leakage_flags.append(flag)
                    logger.warning("LEAKAGE FLAG: %s", flag)

    # ── Build markdown table ──────────────────────────────────────────────────
    col_headers = []
    col_keys: list[tuple[int, bool]] = []
    for h in _HORIZONS:
        for cot in [False, True]:
            col_headers.append(f"h={h} {'COT' if cot else 'no-COT'}")
            col_keys.append((h, cot))

    # Header row
    sep = "| " + " | ".join(["---"] * (len(col_headers) + 1)) + " |"
    header = "| Model | " + " | ".join(col_headers) + " |"

    rows = [header, sep]
    for model in _MODELS:
        cells = []
        for (h, cot) in col_keys:
            sharpe, cs_ric = results.get((model, h, cot), (float("nan"), float("nan")))
            cells.append(f"{_fmt(sharpe, 2)} / {_fmt(cs_ric, 4)}")
        rows.append(f"| {model} | " + " | ".join(cells) + " |")

    table_md = "\n".join(rows)

    # ── Identify best configuration ───────────────────────────────────────────
    best_sharpe = float("-inf")
    best_key: tuple[str, int, bool] | None = None
    meanrev_h5_sharpe = results.get(("MeanReversion", 5, False), (float("nan"), float("nan")))[0]

    for (model, h, cot), (sharpe, cs_ric) in results.items():
        if np.isfinite(sharpe) and sharpe > best_sharpe:
            best_sharpe = sharpe
            best_key = (model, h, cot)

    best_line = "None found"
    if best_key:
        bm, bh, bcot = best_key
        best_line = (
            f"**{bm}** at h={bh} {'with' if bcot else 'without'} COT "
            f"— Sharpe {best_sharpe:.2f}"
        )

    beats_baseline = (
        np.isfinite(best_sharpe)
        and np.isfinite(meanrev_h5_sharpe)
        and best_sharpe > meanrev_h5_sharpe
    )

    leakage_section = ""
    if leakage_flags:
        leakage_section = "\n\n## Leakage flags\n\n"
        leakage_section += "\n".join(f"- {f}" for f in leakage_flags)

    report = f"""# Phase 13 Sweep Results

*Format: Sharpe (net, non-overlapping) / CS-RIC (mean OOS)*

*Embargo = horizon for each run. All results are out-of-sample walk-forward.*

## Sweep Matrix

{table_md}

## Best Configuration

{best_line}

MeanReversion h=5 no-COT Sharpe (Phase 8 baseline): {_fmt(meanrev_h5_sharpe, 2)}

LambdaMART beats baseline: **{'YES' if beats_baseline else 'NO'}**
{leakage_section}
## Notes

- Sharpe uses non-overlapping subsampling every `horizon` steps to avoid autocorrelation inflation.
- CS-RIC is mean cross-sectional rank information coefficient across all OOS test dates.
- Embargo = horizon ensures forward return labels cannot leak across the train/test boundary.
- Costs charged at {cfg.cost_bps} bps round-trip on turnover.

*Research tooling only — not investment advice.*
"""

    out_path = Path(cfg.paths.outputs_reports) / "phase13_sweep.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    logger.info("Sweep results saved → %s", out_path)

    print("\n" + "=" * 70)
    print("PHASE 13 SWEEP COMPLETE")
    print("=" * 70)
    print(table_md)
    print(f"\nBest: {best_line}")
    print(f"MeanReversion h=5 no-COT baseline: {_fmt(meanrev_h5_sharpe, 2)}")
    print(f"LambdaMART beats baseline: {'YES' if beats_baseline else 'NO'}")
    print(f"\nFull report → {out_path}")

    return best_sharpe, best_key, beats_baseline


if __name__ == "__main__":
    main()
