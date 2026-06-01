"""Phase 16 equity sector sweep: horizons × models (no COT variant).

Runs all 9 combinations:
  horizons: [5, 21, 63]
  models:   [MeanReversion, LightGBM, LambdaMART]

Embargo = horizon for each run.

Output: outputs/reports/phase16_sweep.md

Usage
-----
    python scripts/run_sweep_sectors.py
    python scripts/run_sweep_sectors.py --refresh
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
from src.pipeline_ranking_sectors import run_sectors_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_sweep_sectors")

_HORIZONS = [5, 21, 63]
_MODELS = ["MeanReversion", "LightGBM", "LambdaMART"]
_LEAKAGE_CSRIC = 0.15
_LEAKAGE_SHARPE = 2.5

# Commodity reference results from Phase 14 (for comparison table)
_COMMODITY_BEST = {
    ("MeanReversion", 5):  (0.40, 0.020),
    ("MeanReversion", 21): (None, None),
    ("MeanReversion", 63): (None, None),
    ("LightGBM",      5):  (None, None),
    ("LightGBM",      21): (None, None),
    ("LightGBM",      63): (0.79, 0.055),
    ("LambdaMART",    5):  (None, None),
    ("LambdaMART",    21): (None, None),
    ("LambdaMART",    63): (None, None),
}


def _fmt(v, decimals: int = 2) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "N/A"
    return f"{v:.{decimals}f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 16 equity sector sweep — 9 configurations")
    parser.add_argument("--refresh", action="store_true", help="Force re-fetch of data")
    args = parser.parse_args()

    cfg = load_config()

    # Results dict: (model, horizon) → (sharpe, cs_ric, turnover)
    results: dict[tuple[str, int], tuple[float, float, float]] = {}
    leakage_flags: list[str] = []

    total = len(_HORIZONS) * len(_MODELS)
    run_n = 0

    for horizon in _HORIZONS:
        logger.info("=== Equity sectors: horizon=%d  (embargo=%d) ===", horizon, horizon)
        t0 = time.time()

        try:
            run_results = run_sectors_pipeline(
                cfg,
                horizon=horizon,
                force_refresh=args.refresh,
                embargo=horizon,
                model_names=_MODELS,
            )
        except Exception as exc:
            logger.error("Sector pipeline failed for h=%d: %s", horizon, exc)
            for m in _MODELS:
                results[(m, horizon)] = (float("nan"), float("nan"), float("nan"))
            continue

        elapsed = time.time() - t0
        logger.info("  Horizon=%d finished in %.1fs", horizon, elapsed)

        for model_name in _MODELS:
            run_n += 1
            r = run_results.get(model_name)
            if r is None:
                results[(model_name, horizon)] = (float("nan"), float("nan"), float("nan"))
                logger.warning("  %s: not found in results", model_name)
                continue

            sharpe   = r.ls_sharpe
            cs_ric   = r.mean_cs_ric
            turnover = r.turnover

            results[(model_name, horizon)] = (sharpe, cs_ric, turnover)
            logger.info(
                "  [%d/%d] %s (h=%d): Sharpe=%.2f  CS-RIC=%.4f  turnover=%.3f",
                run_n, total, model_name, horizon,
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

    # ── Build equity sweep markdown table ─────────────────────────────────────
    col_headers = [f"h={h}" for h in _HORIZONS]

    sep    = "| " + " | ".join(["---"] * (len(col_headers) + 1)) + " |"
    header = "| Model | " + " | ".join(col_headers) + " |"

    equity_rows = [header, sep]
    for model in _MODELS:
        cells = []
        for h in _HORIZONS:
            sharpe, cs_ric, _ = results.get((model, h), (float("nan"), float("nan"), float("nan")))
            cells.append(f"{_fmt(sharpe, 2)} / {_fmt(cs_ric, 4)}")
        equity_rows.append(f"| {model} | " + " | ".join(cells) + " |")

    equity_table_md = "\n".join(equity_rows)

    # ── Identify equity best and compare to commodity best ────────────────────
    best_sharpe = float("-inf")
    best_key: tuple[str, int] | None = None

    for (model, h), (sharpe, cs_ric, _) in results.items():
        if np.isfinite(sharpe) and sharpe > best_sharpe:
            best_sharpe = sharpe
            best_key = (model, h)

    best_line = "None found"
    if best_key:
        bm, bh = best_key
        best_line = f"**{bm}** at h={bh} — Sharpe {best_sharpe:.2f}"

    beats_commodity = np.isfinite(best_sharpe) and best_sharpe > 0.79
    commodity_best_sharpe = 0.79

    leakage_section = ""
    if leakage_flags:
        leakage_section = "\n\n## Leakage flags\n\n"
        leakage_section += "\n".join(f"- {f}" for f in leakage_flags)

    # ── Combined comparison table: equity vs commodity ─────────────────────────
    compare_header = "| Model | Horizon | Equity Sharpe / CS-RIC | Commodity Sharpe / CS-RIC (Phase 14) |"
    compare_sep    = "| --- | --- | --- | --- |"
    compare_rows   = [compare_header, compare_sep]
    for model in _MODELS:
        for h in _HORIZONS:
            eq_s, eq_r, _ = results.get((model, h), (float("nan"), float("nan"), float("nan")))
            cm_s, cm_r = _COMMODITY_BEST.get((model, h), (None, None))
            compare_rows.append(
                f"| {model} | {h} | {_fmt(eq_s, 2)} / {_fmt(eq_r, 4)} | "
                f"{_fmt(cm_s, 2) if cm_s is not None else 'N/A'} / "
                f"{_fmt(cm_r, 4) if cm_r is not None else 'N/A'} |"
            )

    compare_table_md = "\n".join(compare_rows)

    report = f"""# Phase 16 Equity Sector Sweep Results

*Format: Sharpe (net, non-overlapping) / CS-RIC (mean OOS)*

*Embargo = horizon for each run. All results are out-of-sample walk-forward.*

*Universe: XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY — 9 SPDR sector ETFs*

*No COT features (futures-specific). No carry proxy (futures-specific).*

*History start: 1999-01-01  |  End: {cfg.dates["end"]}*

---

## Equity Sector Sweep Matrix

{equity_table_md}

---

## Cross-Asset Comparison: Equity Sectors vs Commodities (Phase 14)

{compare_table_md}

---

## Summary

**Equity best:** {best_line}

**Commodity best (Phase 14):** LightGBM at h=63 — Sharpe {commodity_best_sharpe:.2f}

**Equity beats commodity best ({commodity_best_sharpe:.2f}):** {'**YES**' if beats_commodity else 'NO'}
{leakage_section}
---

## Notes

- Sharpe uses non-overlapping subsampling every `horizon` steps to avoid autocorrelation inflation.
- CS-RIC is mean cross-sectional rank information coefficient across all OOS test dates.
- Embargo = horizon ensures forward return labels cannot leak across the train/test boundary.
- Costs charged at {cfg.cost_bps} bps round-trip on turnover.
- Equity sector history starts 1999-01-01; longer history than commodity backtest (2005-01-01).

*Research tooling only — not investment advice.*
"""

    out_path = Path(cfg.paths.outputs_reports) / "phase16_sweep.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    logger.info("Sector sweep results saved → %s", out_path)

    print("\n" + "=" * 70)
    print("PHASE 16 EQUITY SECTOR SWEEP COMPLETE")
    print("=" * 70)
    print(equity_table_md)
    print(f"\nEquity best: {best_line}")
    print(f"Commodity best (Phase 14): LightGBM h=63 — Sharpe {commodity_best_sharpe:.2f}")
    print(f"Equity beats commodity: {'YES' if beats_commodity else 'NO'}")
    print(f"\nFull report → {out_path}")

    return best_sharpe, best_key, beats_commodity


if __name__ == "__main__":
    main()
