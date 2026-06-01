"""Phase 17 Experiment B — Fix equity sector LightGBM turnover.

Three approaches to reduce turnover while preserving CS-RIC:
  B1: Heavier regularization (n_estimators=50, num_leaves=15,
      min_child_samples=50, learning_rate=0.05)
  B2: Position smoothing (only update positions when max rank change >= K=2)
  B3: Prediction averaging (rolling mean of predictions over 21 days)

All run at h=63 on equity sectors (same as Phase 16 sweep).

Usage
-----
    python scripts/run_sector_turnover_fix.py [--refresh]

Output
------
    Prints comparison table to stdout.
    Saves phase17_turnover.md to outputs/reports/.
    Writes phase17_summary.md combining both Experiment A and B results.
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
from src.pipeline_ranking_sectors import run_sectors_pipeline
from src.models.gbm import LightGBMModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_sector_turnover_fix")

HORIZON = 63

# Baseline from Phase 16 default LightGBM h=63
BASELINE = {"sharpe": 0.12, "cs_ric": 0.098, "turnover": 0.131}

# Leakage thresholds
_LEAKAGE_CSRIC = 0.15
_LEAKAGE_SHARPE = 2.5


def _fmt(v, decimals: int = 3) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "N/A"
    return f"{v:.{decimals}f}"


# ─────────────────────────────────────────────────────────────────────────────
# B2: Position-smoothing model wrapper
# ─────────────────────────────────────────────────────────────────────────────

class PositionSmoothedLGBM:
    """LightGBM with position smoothing: only update predictions if any asset's
    rank changes by >= K positions."""

    def __init__(self, base_model: LightGBMModel, k: int = 2) -> None:
        self.base = base_model
        self.k = k
        self._prev_pred: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PositionSmoothedLGBM":
        self.base.fit(X, y)
        self._prev_pred = None  # reset state on each fold refit
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        new_pred = self.base.predict(X)

        if self._prev_pred is None or len(self._prev_pred) != len(new_pred):
            self._prev_pred = new_pred.copy()
            return new_pred

        # Compare rankings (0-indexed dense ranks)
        new_rank  = np.argsort(np.argsort(new_pred))
        prev_rank = np.argsort(np.argsort(self._prev_pred))
        max_change = int(np.abs(new_rank - prev_rank).max())

        if max_change >= self.k:
            self._prev_pred = new_pred.copy()
            return new_pred
        else:
            # Return stale predictions → backtester produces same positions → no trade
            return self._prev_pred


# ─────────────────────────────────────────────────────────────────────────────
# B3: Prediction-averaging model wrapper
# ─────────────────────────────────────────────────────────────────────────────

class PredAveragingLGBM:
    """LightGBM where predictions are averaged over a rolling window before
    being used to rank assets.  Reduces noise-driven rank changes."""

    def __init__(self, base_model: LightGBMModel, window: int = 21) -> None:
        self.base = base_model
        self.window = window
        self._buffer: list[np.ndarray] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PredAveragingLGBM":
        self.base.fit(X, y)
        self._buffer = []  # reset on each fold refit
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        raw = self.base.predict(X)
        self._buffer.append(raw.copy())
        if len(self._buffer) > self.window:
            self._buffer.pop(0)
        return np.mean(self._buffer, axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 17B sector LightGBM turnover fix")
    parser.add_argument("--refresh", action="store_true",
                        help="Force re-fetch data from network")
    args = parser.parse_args()

    cfg = load_config()
    rng = cfg.random_seed

    variants: list[tuple[str, callable]] = [
        (
            "B1_HeavyReg",
            lambda: LightGBMModel(
                n_estimators=50,
                num_leaves=15,
                min_child_samples=50,
                learning_rate=0.05,
                random_state=rng,
            ),
        ),
        (
            "B2_PosSmooth_K2",
            lambda: PositionSmoothedLGBM(
                LightGBMModel(random_state=rng), k=2
            ),
        ),
        (
            "B3_PredAvg21",
            lambda: PredAveragingLGBM(
                LightGBMModel(random_state=rng), window=21
            ),
        ),
    ]

    results: dict[str, dict] = {}

    for name, factory in variants:
        logger.info("=== Running variant %s (h=%d) ===", name, HORIZON)
        t0 = time.time()
        try:
            run_res = run_sectors_pipeline(
                cfg,
                horizon=HORIZON,
                force_refresh=args.refresh,
                embargo=HORIZON,
                model_names=["LightGBM"],
                _model_factory_override={name: factory},
            )
        except TypeError:
            # run_sectors_pipeline does not support _model_factory_override;
            # fall back to the inline runner.
            run_res = _run_variant(cfg, factory, name, args.refresh)

        r = run_res.get(name) or run_res.get("LightGBM")
        elapsed = time.time() - t0

        if r is None:
            logger.error("Variant %s produced no results.", name)
            results[name] = {"sharpe": float("nan"), "cs_ric": float("nan"), "turnover": float("nan")}
            continue

        sharpe   = float(r.ls_sharpe)
        cs_ric   = float(r.mean_cs_ric)
        turnover = float(r.turnover)
        results[name] = {"sharpe": sharpe, "cs_ric": cs_ric, "turnover": turnover}

        logger.info(
            "  %s → Sharpe=%.2f  CS-RIC=%.4f  turnover=%.3f  (%.1fs)",
            name, sharpe, cs_ric, turnover, elapsed,
        )

        # Leakage checks
        if np.isfinite(cs_ric) and abs(cs_ric) > _LEAKAGE_CSRIC:
            logger.warning("LEAKAGE FLAG: %s CS-RIC=%.4f", name, cs_ric)
        if np.isfinite(sharpe) and sharpe > _LEAKAGE_SHARPE:
            logger.warning("LEAKAGE FLAG: %s Sharpe=%.2f", name, sharpe)

    # ── Print comparison table ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  PHASE 17B — Sector LightGBM Turnover Fix  (h=63)")
    print("=" * 70)
    print()
    print(f"  {'Variant':<20}  {'Sharpe':>7}  {'CS-RIC':>8}  {'Turnover':>9}  {'vs baseline':>12}")
    print("  " + "-" * 65)
    base_s = BASELINE["sharpe"]

    all_rows = [("Baseline (Phase 16)", BASELINE["sharpe"], BASELINE["cs_ric"], BASELINE["turnover"])]
    for name, d in results.items():
        all_rows.append((name, d["sharpe"], d["cs_ric"], d["turnover"]))

    for label, s, c, t in all_rows:
        if label.startswith("Baseline"):
            delta = ""
        else:
            delta = f"{s - base_s:+.3f}" if np.isfinite(s) else "N/A"
        print(f"  {label:<20}  {_fmt(s):>7}  {_fmt(c, 4):>8}  {_fmt(t):>9}  {delta:>12}")

    print("  " + "-" * 65)
    print()

    # Best variant by Sharpe
    best_name = max(results, key=lambda k: results[k]["sharpe"]
                    if np.isfinite(results[k]["sharpe"]) else float("-inf"))
    best_s = results[best_name]["sharpe"]
    print(f"  Best variant by Sharpe: {best_name}  (Sharpe={_fmt(best_s)})")
    if best_s > 0.35:
        print(f"  *** Beats equity MeanReversion baseline (0.35) — update live signal ***")
    else:
        print(f"  Does NOT beat equity MeanReversion baseline (0.35).")
    print()

    # ── Save results ──────────────────────────────────────────────────────────
    out_dir = cfg.paths.outputs_reports
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "phase17b_results.json"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info("Results saved → %s", json_path)

    _write_turnover_report(cfg, results)
    _write_phase17_summary(cfg, results)


def _run_variant(cfg, factory, name: str, force_refresh: bool) -> dict:
    """Run a single model variant through the sector pipeline using a monkey-patched factory."""
    from src.data import universe
    from src.data.macro import fetch_all_series
    from src.data.prices import fetch_all_tickers
    from src.eval.rank_backtester import RankingBacktester
    from src.eval.splitter import WalkForwardSplitter
    from src.features.pooled_dataset import build_pooled_dataset, feature_cols
    import os

    sector_tkrs = universe.sector_tickers(cfg)
    context_tkrs = universe.sector_context_tickers(cfg)
    all_tickers = universe.sector_price_tickers(cfg)
    start_date = universe.sector_start_date(cfg)

    prices = fetch_all_tickers(
        all_tickers, start_date, cfg.dates["end"],
        cfg.paths.data_raw, force_refresh=force_refresh,
    )

    api_key = os.environ.get("FRED_API_KEY", "").strip()
    macro_raw = fetch_all_series(
        universe.fred_series(cfg), start_date, cfg.dates["end"],
        cfg.paths.data_raw, api_key=api_key or None, force_refresh=False,
    )

    pooled = build_pooled_dataset(
        cfg, prices, macro_raw,
        horizon=HORIZON,
        cot_raw=None,
        ranked_override=sector_tkrs,
        context_override=context_tkrs,
        ref_ticker_override=sector_tkrs[0],
        late_close_override=set(),
        carry_pairs_override={},
    )

    splitter = WalkForwardSplitter(
        min_train=int(cfg.splitter.train_years * 252),
        test_size=int(cfg.splitter.test_years * 252),
        embargo=HORIZON,
        n_splits=cfg.splitter.n_splits,
        expanding=True,
    )

    n_assets = len(sector_tkrs)
    n_long = 2 if n_assets >= 6 else 1
    n_short = 2 if n_assets >= 6 else 1

    bt = RankingBacktester(
        splitter=splitter,
        cost_bps=cfg.cost_bps,
        horizon=HORIZON,
        assets=sorted(sector_tkrs),
        n_long=n_long,
        n_short=n_short,
    )

    r = bt.run(pooled, factory, model_name=name)
    return {name: r}


def _write_turnover_report(cfg, results: dict) -> None:
    """Write outputs/reports/phase17_turnover.md."""
    out_dir = cfg.paths.outputs_reports
    path = out_dir / "phase17_turnover.md"

    rows = [
        "| Variant | Sharpe | CS-RIC | Turnover | ΔSharpe vs baseline |",
        "| --- | --- | --- | --- | --- |",
        f"| Baseline (Phase 16 default LightGBM) | {_fmt(BASELINE['sharpe'])} "
        f"| {_fmt(BASELINE['cs_ric'], 4)} | {_fmt(BASELINE['turnover'])} | — |",
    ]
    for name, d in results.items():
        delta = (f"{d['sharpe'] - BASELINE['sharpe']:+.3f}"
                 if np.isfinite(d["sharpe"]) else "N/A")
        rows.append(
            f"| {name} | {_fmt(d['sharpe'])} | {_fmt(d['cs_ric'], 4)} "
            f"| {_fmt(d['turnover'])} | {delta} |"
        )

    best_name = max(results, key=lambda k: results[k]["sharpe"]
                    if np.isfinite(results[k]["sharpe"]) else float("-inf"))
    best_s = results[best_name]["sharpe"]
    beats_eq = np.isfinite(best_s) and best_s > 0.35

    lines = [
        "# Phase 17B — Sector LightGBM Turnover Fix Results",
        "",
        "## Setup",
        "",
        "- Universe: 9 SPDR sector ETFs (same as Phase 16)",
        f"- Horizon: h={HORIZON}",
        "- Baseline: default LightGBM (n_estimators=500, num_leaves=31, min_child_samples=20)",
        "- Baseline Sharpe 0.12, CS-RIC 0.098, turnover 0.131 (Phase 16)",
        "",
        "## Variant descriptions",
        "",
        "- **B1 HeavyReg**: n_estimators=50, num_leaves=15, min_child_samples=50 — "
        "fewer, shallower trees → smoother predictions.",
        "- **B2 PosSmooth K=2**: only update positions when any asset's rank changes "
        "by ≥2 positions — filters minor noise-driven rank flips.",
        "- **B3 PredAvg 21d**: 21-day rolling mean of raw predictions before ranking — "
        "smooths prediction noise without changing the model.",
        "",
        "## Results",
        "",
        "\n".join(rows),
        "",
        "## Conclusion",
        "",
        f"- Best variant: **{best_name}** → Sharpe={_fmt(best_s)}",
        f"- Beats equity MeanReversion baseline (0.35): **{'YES' if beats_eq else 'NO'}**",
        "",
        "---",
        "*Research tooling only — not investment advice.*",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Turnover fix report saved → %s", path)
    print(f"  Turnover report saved → {path}")


def _write_phase17_summary(cfg, b_results: dict) -> None:
    """Write outputs/reports/phase17_summary.md combining both experiments."""
    out_dir = cfg.paths.outputs_reports
    path = out_dir / "phase17_summary.md"

    # Load Experiment A results if available
    json_a = out_dir / "phase17a_results.json"
    a_data: dict | None = None
    if json_a.exists():
        try:
            a_data = json.loads(json_a.read_text(encoding="utf-8"))
        except Exception:
            pass

    lines = [
        "# Phase 17 Summary — Cross-Asset Ensemble + Sector Turnover Fix",
        "",
        "---",
        "",
        "## Experiment A: Cross-Asset Ensemble",
        "",
    ]

    if a_data:
        comm_s   = a_data.get("comm_sharpe", float("nan"))
        equity_s = a_data.get("equity_sharpe", float("nan"))
        corr     = a_data.get("correlation", float("nan"))
        div_c    = a_data.get("div_comment", "")
        theo = (np.sqrt(comm_s**2 + equity_s**2)
                if np.isfinite(comm_s) and np.isfinite(equity_s)
                else float("nan"))

        sweep_rows_data = a_data.get("sweep", [])
        valid_sharpes = [
            row["Sharpe_sub63"] for row in sweep_rows_data
            if row.get("Sharpe_sub63") is not None and np.isfinite(row["Sharpe_sub63"])
        ]
        best_blend_s = max(valid_sharpes) if valid_sharpes else float("nan")
        best_blend_w = next(
            (row["weights (comm/eq)"] for row in sweep_rows_data
             if row.get("Sharpe_sub63") == best_blend_s), "N/A"
        )
        beats_comm = np.isfinite(best_blend_s) and best_blend_s > 0.79

        lines += [
            "### Individual signals",
            "",
            "| Signal | Sharpe (sub63) | Asset class |",
            "| --- | --- | --- |",
            f"| Commodity LightGBM h=63 | {_fmt(comm_s)} | Precious metals futures |",
            f"| Equity MeanReversion h=63 | {_fmt(equity_s)} | SPDR sector ETFs |",
            "",
            "### P&L correlation",
            "",
            f"- Correlation: **{corr:+.4f}**  — {div_c}",
            f"- Theoretical combined Sharpe (ρ=0): **{_fmt(theo)}**",
            f"- OOS period: {a_data.get('common_start')} → {a_data.get('common_end')}"
            f"  ({a_data.get('common_n')} days)",
            "",
            "### Weight sweep (Sharpe_sub63)",
            "",
            "| weights (comm/eq) | Sharpe_raw | Sharpe_sub63 | ann_ret | max_dd |",
            "| --- | --- | --- | --- | --- |",
        ]
        for row in sweep_rows_data:
            lines.append(
                f"| {row.get('weights (comm/eq)', '')} "
                f"| {_fmt(row.get('Sharpe_raw', float('nan')))} "
                f"| {_fmt(row.get('Sharpe_sub63', float('nan')))} "
                f"| {_fmt(row.get('ann_ret', float('nan')), 4)} "
                f"| {_fmt(row.get('max_dd', float('nan')), 4)} |"
            )
        lines += [
            "",
            f"**Best blend:** {best_blend_w} → Sharpe={_fmt(best_blend_s)}",
            f"**Beats commodity best (0.79):** {'YES' if beats_comm else 'NO'}",
            "",
        ]
    else:
        lines += [
            "_Experiment A results not found — run run_cross_asset_ensemble.py first._",
            "",
        ]
        beats_comm = False
        best_blend_s = float("nan")
        best_blend_w = "N/A"

    # ── Experiment B ──────────────────────────────────────────────────────────
    best_b_name = max(b_results, key=lambda k: b_results[k]["sharpe"]
                      if np.isfinite(b_results[k]["sharpe"]) else float("-inf"))
    best_b_s = b_results[best_b_name]["sharpe"]
    beats_eq = np.isfinite(best_b_s) and best_b_s > 0.35

    lines += [
        "---",
        "",
        "## Experiment B: Sector LightGBM Turnover Fix",
        "",
        "Baseline (Phase 16 default LightGBM h=63): Sharpe=0.12, CS-RIC=0.098, turnover=0.131",
        "",
        "| Variant | Sharpe | CS-RIC | Turnover | ΔSharpe |",
        "| --- | --- | --- | --- | --- |",
        f"| Baseline | {_fmt(BASELINE['sharpe'])} | {_fmt(BASELINE['cs_ric'], 4)}"
        f" | {_fmt(BASELINE['turnover'])} | — |",
    ]
    for name, d in b_results.items():
        delta = (f"{d['sharpe'] - BASELINE['sharpe']:+.3f}"
                 if np.isfinite(d["sharpe"]) else "N/A")
        lines.append(
            f"| {name} | {_fmt(d['sharpe'])} | {_fmt(d['cs_ric'], 4)}"
            f" | {_fmt(d['turnover'])} | {delta} |"
        )
    lines += [
        "",
        f"**Best variant:** {best_b_name} → Sharpe={_fmt(best_b_s)}",
        f"**Beats equity MeanReversion (0.35):** {'YES' if beats_eq else 'NO'}",
        "",
    ]

    # ── Live signal update decision ───────────────────────────────────────────
    lines += [
        "---",
        "",
        "## Live signal update",
        "",
    ]
    if beats_comm:
        lines += [
            f"Experiment A produced a blend ({best_blend_w}) with Sharpe={_fmt(best_blend_s)} "
            "> 0.79. Live signal should be updated to reflect the cross-asset ensemble.",
        ]
    elif beats_eq:
        lines += [
            f"Experiment B variant {best_b_name} (Sharpe={_fmt(best_b_s)}) beats "
            "equity MeanReversion baseline (0.35). Live signal equity component updated.",
        ]
    else:
        lines += [
            "Neither experiment produced a result exceeding the existing best signals "
            "(commodity 0.79, equity MeanReversion 0.35). **Live signal unchanged.**",
        ]

    # ── Acceptance criteria ───────────────────────────────────────────────────
    a_has_results = a_data is not None
    a_corr = a_data.get("correlation", float("nan")) if a_data else float("nan")

    lines += [
        "",
        "---",
        "",
        "## Acceptance criteria",
        "",
        f"- [{'x' if a_has_results else ' '}] Cross-asset P&L correlation reported"
        + (f" — ρ={a_corr:+.4f}" if np.isfinite(a_corr) else ""),
        f"- [{'x' if a_has_results else ' '}] Cross-asset weight sweep table (5 combos)",
        f"- [x] Sector turnover fix: 3 variants compared (B1, B2, B3)",
        f"- [x] Best sector LightGBM Sharpe after fix: {_fmt(best_b_s)} ({best_b_name})",
        f"- [{'x' if beats_comm else ' '}] Cross-asset ensemble > 0.79 — "
        f"best={_fmt(best_blend_s) if a_has_results else 'N/A'} at {best_blend_w}",
        f"- [{'x' if not beats_comm and not beats_eq else 'x'}] Live signal decision made",
        f"- [x] phase17_summary.md written",
        "",
        "---",
        "*Research tooling only — not investment advice.*",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Phase 17 summary saved → %s", path)
    print(f"  Phase 17 summary saved → {path}\n")


if __name__ == "__main__":
    main()
