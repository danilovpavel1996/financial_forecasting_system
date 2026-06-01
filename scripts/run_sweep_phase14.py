"""Phase 14 systematic sweep: new features + extended history (2005–2024).

Runs the same 18 configurations as Phase 13:
  horizons: [5, 21, 63]
  features: [no-COT, with-COT]
  models:   [MeanReversion, LightGBM, LambdaMART]

New vs Phase 13:
  - Data extended from 2010-01-01 to 2005-01-01
  - New features: basis_momentum, carry_proxy, carry_proxy_chg_21d,
    rel_basis_momentum, rel_carry_proxy (carry_features.py)
  - New features: month_sin, month_cos (seasonal_features.py)
  - New features: yield_curve_slope, yield_curve_slope_chg_21d,
    yield_curve_slope_zscore (macro_features.py via DGS2)

Output: outputs/reports/phase14_sweep.md
Side-by-side comparison to Phase 13 matrix is included.

Usage
-----
    python scripts/run_sweep_phase14.py
    python scripts/run_sweep_phase14.py --refresh    # force re-fetch data
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
logger = logging.getLogger("run_sweep_phase14")

_HORIZONS = [5, 21, 63]
_MODELS = ["MeanReversion", "LightGBM", "LambdaMART"]
_LEAKAGE_CSRIC = 0.15
_LEAKAGE_SHARPE = 2.5

# Phase 13 baseline results for comparison
_PHASE13_RESULTS: dict[tuple[str, int, bool], tuple[float, float]] = {
    ("MeanReversion", 5,  False): (0.63,  0.0197),
    ("MeanReversion", 5,  True):  (0.29,  0.0264),
    ("MeanReversion", 21, False): (-0.07, 0.0215),
    ("MeanReversion", 21, True):  (-0.07, 0.0258),
    ("MeanReversion", 63, False): (0.03,  0.0131),
    ("MeanReversion", 63, True):  (-0.12, 0.0149),
    ("LightGBM",      5,  False): (0.43,  0.0109),
    ("LightGBM",      5,  True):  (-0.33, -0.0016),
    ("LightGBM",      21, False): (-0.21, -0.0214),
    ("LightGBM",      21, True):  (0.11,  0.0248),
    ("LightGBM",      63, False): (0.10,  0.0247),
    ("LightGBM",      63, True):  (-0.48, -0.0057),
    ("LambdaMART",    5,  False): (-0.59, -0.0174),
    ("LambdaMART",    5,  True):  (-0.38, -0.0073),
    ("LambdaMART",    21, False): (0.09,  0.0246),
    ("LambdaMART",    21, True):  (0.15,  0.0298),
    ("LambdaMART",    63, False): (0.02,  0.0827),
    ("LambdaMART",    63, True):  (0.27,  0.0768),
}

_PHASE13_BEST_SHARPE = 0.63  # MeanReversion h=5 no-COT


def _fmt(v: float, decimals: int = 2) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "N/A"
    return f"{v:.{decimals}f}"


def _build_matrix_md(
    results: dict[tuple[str, int, bool], tuple[float, float]],
    col_headers: list[str],
    col_keys: list[tuple[int, bool]],
) -> str:
    sep = "| " + " | ".join(["---"] * (len(col_headers) + 1)) + " |"
    header = "| Model | " + " | ".join(col_headers) + " |"
    rows = [header, sep]
    for model in _MODELS:
        cells = []
        for (h, cot) in col_keys:
            sharpe, cs_ric = results.get((model, h, cot), (float("nan"), float("nan")))
            cells.append(f"{_fmt(sharpe, 2)} / {_fmt(cs_ric, 4)}")
        rows.append(f"| {model} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def _build_delta_md(
    p14: dict[tuple[str, int, bool], tuple[float, float]],
    p13: dict[tuple[str, int, bool], tuple[float, float]],
    col_headers: list[str],
    col_keys: list[tuple[int, bool]],
) -> str:
    """Build a delta table showing Phase14 - Phase13 Sharpe changes."""
    sep = "| " + " | ".join(["---"] * (len(col_headers) + 1)) + " |"
    header = "| Model | " + " | ".join(col_headers) + " |"
    rows = [header, sep]
    for model in _MODELS:
        cells = []
        for (h, cot) in col_keys:
            s14, _ = p14.get((model, h, cot), (float("nan"), float("nan")))
            s13, _ = p13.get((model, h, cot), (float("nan"), float("nan")))
            if np.isfinite(s14) and np.isfinite(s13):
                delta = s14 - s13
                sign = "+" if delta >= 0 else ""
                cells.append(f"{sign}{_fmt(delta, 2)}")
            else:
                cells.append("N/A")
        rows.append(f"| {model} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 14 sweep: new features + extended history")
    parser.add_argument("--refresh", action="store_true", help="Force re-fetch of data")
    args = parser.parse_args()

    cfg = load_config()

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

    # ── Build column headers ──────────────────────────────────────────────────
    col_headers = []
    col_keys: list[tuple[int, bool]] = []
    for h in _HORIZONS:
        for cot in [False, True]:
            col_headers.append(f"h={h} {'COT' if cot else 'no-COT'}")
            col_keys.append((h, cot))

    # ── Phase 14 matrix ───────────────────────────────────────────────────────
    p14_table = _build_matrix_md(results, col_headers, col_keys)

    # ── Phase 13 matrix (from hardcoded baseline) ─────────────────────────────
    p13_table = _build_matrix_md(_PHASE13_RESULTS, col_headers, col_keys)

    # ── Delta table ───────────────────────────────────────────────────────────
    delta_table = _build_delta_md(results, _PHASE13_RESULTS, col_headers, col_keys)

    # ── Best configuration ────────────────────────────────────────────────────
    best_sharpe = float("-inf")
    best_key: tuple[str, int, bool] | None = None

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

    p14_meanrev_h5 = results.get(("MeanReversion", 5, False), (float("nan"), float("nan")))[0]
    beats_p13_baseline = np.isfinite(best_sharpe) and best_sharpe > _PHASE13_BEST_SHARPE

    leakage_section = ""
    if leakage_flags:
        leakage_section = "\n\n## Leakage Flags\n\n"
        leakage_section += "\n".join(f"- {f}" for f in leakage_flags)

    report = f"""# Phase 14 Sweep Results

*New features: basis_momentum, carry_proxy, carry_proxy_chg_21d, rel_basis_momentum,
rel_carry_proxy, month_sin, month_cos, yield_curve_slope, yield_curve_slope_chg_21d,
yield_curve_slope_zscore.  History extended: 2005-01-01 → 2024-12-31.*

*Format: Sharpe (net, non-overlapping) / CS-RIC (mean OOS)*

*Embargo = horizon for each run. All results are out-of-sample walk-forward.*

---

## Phase 14 Sweep Matrix

{p14_table}

## Phase 13 Sweep Matrix (baseline)

{p13_table}

## Delta: Phase 14 Sharpe minus Phase 13 Sharpe

{delta_table}

---

## Best Configuration (Phase 14)

{best_line}

MeanReversion h=5 no-COT (Phase 13 baseline): {_fmt(_PHASE13_BEST_SHARPE, 2)}
MeanReversion h=5 no-COT (Phase 14): {_fmt(p14_meanrev_h5, 2)}

Phase 14 best beats Phase 13 baseline (Sharpe {_PHASE13_BEST_SHARPE:.2f}): **{'YES' if beats_p13_baseline else 'NO'}**
{leakage_section}

## Notes

- Sharpe uses non-overlapping subsampling every `horizon` steps to reduce autocorrelation.
- CS-RIC is mean cross-sectional rank information coefficient across all OOS test dates.
- Embargo = horizon; forward return labels cannot leak across the train/test boundary.
- Costs charged at {cfg.cost_bps} bps round-trip on turnover.
- COT data (Disaggregated CFTC) available from ~2006; warm-up starts 2007 for percentile features.
- carry_proxy available for GC=F (vs GLD) and SI=F (vs SLV) only.
- yield_curve_slope requires both DGS10 and DGS2; zscore requires 252-day history warm-up.

*Research tooling only — not investment advice.*
"""

    out_path = Path(cfg.paths.outputs_reports) / "phase14_sweep.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    logger.info("Sweep results saved → %s", out_path)

    print("\n" + "=" * 70)
    print("PHASE 14 SWEEP COMPLETE")
    print("=" * 70)
    print("\n--- Phase 14 Matrix ---")
    print(p14_table)
    print("\n--- Delta vs Phase 13 (Sharpe change) ---")
    print(delta_table)
    print(f"\nBest (Phase 14): {best_line}")
    print(f"Phase 13 baseline (MeanReversion h=5 no-COT): {_fmt(_PHASE13_BEST_SHARPE, 2)}")
    print(f"Beats Phase 13 baseline: {'YES' if beats_p13_baseline else 'NO'}")
    print(f"\nFull report → {out_path}")

    return best_sharpe, best_key, beats_p13_baseline


if __name__ == "__main__":
    main()
