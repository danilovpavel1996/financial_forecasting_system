"""Phase 20 crypto sweep: 10 tokens, cost_bps=20, horizons × models.

Runs 12 configurations:
  horizons: [5, 21, 63]
  models:   [MeanReversion, LightGBM, LambdaMART]
  + LightGBM with pred_avg_window=21 at each horizon (B3 variant)

All crypto data filtered to business days (Mon-Fri) before training.
Embargo = horizon for each run.

Output: outputs/reports/phase20_crypto_sweep.md

Usage
-----
    python scripts/run_sweep_crypto.py
    python scripts/run_sweep_crypto.py --refresh
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
from src.data import universe
from src.pipeline_ranking_crypto import run_crypto_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_sweep_crypto")

_HORIZONS = [5, 21, 63]
_MODELS = ["MeanReversion", "LightGBM", "LambdaMART"]
_LEAKAGE_CSRIC = 0.15
_LEAKAGE_SHARPE = 2.5


def _fmt(v, decimals: int = 2) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "N/A"
    return f"{v:.{decimals}f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 20 crypto sweep — 10 tokens, 12 configs")
    parser.add_argument("--refresh", action="store_true", help="Force re-fetch of data")
    args = parser.parse_args()

    cfg = load_config()

    n_tokens = len(universe.crypto_tickers(cfg))
    cost_bps  = universe.crypto_cost_bps(cfg)
    start_date = universe.crypto_start_date(cfg)
    logger.info("Crypto universe: %d tokens, cost_bps=%.0f, start=%s", n_tokens, cost_bps, start_date)

    results: dict[tuple[str, int], tuple] = {}
    leakage_flags: list[str] = []

    # ── Base sweep: 9 configs ─────────────────────────────────────────────────
    for horizon in _HORIZONS:
        logger.info("=== Crypto: horizon=%d  (embargo=%d) ===", horizon, horizon)
        t0 = time.time()

        try:
            run_results = run_crypto_pipeline(
                cfg,
                horizon=horizon,
                force_refresh=args.refresh,
                embargo=horizon,
                model_names=_MODELS,
                pred_avg_window=1,
            )
        except Exception as exc:
            logger.error("Crypto pipeline failed for h=%d: %s", horizon, exc)
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

    # ── B3 variant: LightGBM + pred_avg_window=21 ─────────────────────────────
    for horizon in _HORIZONS:
        logger.info("=== Crypto B3 (PredAvg21): horizon=%d ===", horizon)
        t0 = time.time()

        try:
            run_b3 = run_crypto_pipeline(
                cfg,
                horizon=horizon,
                force_refresh=False,
                embargo=horizon,
                model_names=["LightGBM"],
                pred_avg_window=21,
            )
        except Exception as exc:
            logger.error("Crypto B3 pipeline failed for h=%d: %s", horizon, exc)
            results[("LightGBM_B3", horizon)] = (float("nan"), float("nan"), float("nan"), None)
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
            sharpe, cs_ric, turnover, _ = results.get((model, h), (float("nan"), float("nan"), float("nan"), None))
            cells.append(f"{_fmt(sharpe, 2)} / {_fmt(cs_ric, 4)}")
        label = model if model != "LightGBM_B3" else "LightGBM PredAvg21"
        sweep_rows.append(f"| {label} | " + " | ".join(cells) + " |")

    sweep_table_md = "\n".join(sweep_rows)

    # Turnover row
    turn_rows = ["| Model | " + " | ".join(col_headers) + " |", sep]
    for model in all_model_labels:
        cells = []
        for h in _HORIZONS:
            _, _, turnover, _ = results.get((model, h), (float("nan"), float("nan"), float("nan"), None))
            cells.append(_fmt(turnover, 3))
        label = model if model != "LightGBM_B3" else "LightGBM PredAvg21"
        turn_rows.append(f"| {label} | " + " | ".join(cells) + " |")
    turnover_table_md = "\n".join(turn_rows)

    # ── Best config ───────────────────────────────────────────────────────────
    best_sharpe = float("-inf")
    best_key: tuple | None = None

    for (model, h), (sharpe, cs_ric, _, _) in results.items():
        if np.isfinite(sharpe) and sharpe > best_sharpe:
            best_sharpe = sharpe
            best_key = (model, h)

    best_line = "None found"
    if best_key:
        bm, bh = best_key
        display = bm if bm != "LightGBM_B3" else "LightGBM PredAvg21"
        best_line = f"**{display}** at h={bh} — Sharpe {best_sharpe:.2f}"

    leakage_section = ""
    if leakage_flags:
        leakage_section = "\n\n## Leakage flags\n\n"
        leakage_section += "\n".join(f"- {f}" for f in leakage_flags)

    crypto_tickers_str = ", ".join(universe.crypto_tickers(cfg))

    report = f"""# Phase 20 Crypto Sweep Results (10 tokens, cost_bps={cost_bps:.0f})

*Format: Sharpe (net, non-overlapping) / CS-RIC (mean OOS)*

*Embargo = horizon for each run. All results are out-of-sample walk-forward.*

*Universe ({n_tokens} tokens): {crypto_tickers_str}*

*Data filtered to business days (Mon-Fri) — weekend returns absorbed into Monday bar.*

*History start: {start_date}  |  End: {cfg.dates["end"]}*

*Cost: {cost_bps:.0f} bps round-trip (Binance/Kraken ~0.1% per side)*

---

## Crypto Sweep Matrix

{sweep_table_md}

---

## Turnover Matrix

{turnover_table_md}

---

## Summary

**Crypto best:** {best_line}

**Forex reference (Phase 18):** LightGBM B3 h=5 — Sharpe 0.94

**Min viable threshold:** Sharpe > 0.3 after {cost_bps:.0f} bps costs (confirms cross-asset approach)
{leakage_section}
---

## Notes

- Sharpe uses non-overlapping subsampling every `horizon` steps.
- CS-RIC is mean cross-sectional rank IC across all OOS test dates.
- Crypto has ~5 years of data (2020–2024). h=5 is most robust; h=63 has only ~20 independent periods.
- Costs: {cost_bps:.0f} bps round-trip is conservative (Binance Spot: 0.1% maker + 0.1% taker).
- With 10 tokens: top-3 long, bottom-3 short.
- High crypto vol (50-100% ann.) means high absolute P&L variance — interpret Sharpe carefully.
- Weekend data is dropped: Monday bars contain Fri-close to Mon-close move (Sat+Sun included).

*Research tooling only — not investment advice.*
"""

    out_path = Path(cfg.paths.outputs_reports) / "phase20_crypto_sweep.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    logger.info("Crypto sweep results saved → %s", out_path)

    print("\n" + "=" * 70)
    print("PHASE 20 CRYPTO SWEEP COMPLETE")
    print("=" * 70)
    print(sweep_table_md)
    print(f"\nCrypto best: {best_line}")
    print(f"\nFull report → {out_path}")

    return best_sharpe, best_key


if __name__ == "__main__":
    main()
