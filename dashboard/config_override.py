"""Translate sidebar widget values into run_ranking_pipeline() kwargs.

This module is the only place that knows the mapping between UI controls and
pipeline parameters. No Streamlit imports — this is pure data transformation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class RunConfig:
    """Parameters for one backtest run (derived from sidebar selections)."""
    horizon: int
    use_cot: bool
    vol_target: Optional[float]
    max_leverage: float
    vol_lookback: int
    models: list[str]            # subset of pipeline model names to display
    force_refresh: bool = False
    universe: str = "Commodities"
    pred_avg_window: int = 1     # 1 = no averaging

    def pipeline_kwargs(self) -> dict:
        """Return kwargs for run_ranking_pipeline() (commodity pipeline)."""
        return dict(
            horizon=self.horizon,
            use_cot=self.use_cot,
            vol_target=self.vol_target,
            max_leverage=self.max_leverage,
            vol_lookback=self.vol_lookback,
            force_refresh=self.force_refresh,
            embargo=self.horizon,  # must equal horizon to prevent label leakage
        )

    def forex_kwargs(self) -> dict:
        """Return kwargs for run_forex_pipeline()."""
        return dict(
            horizon=self.horizon,
            force_refresh=self.force_refresh,
            vol_target=self.vol_target,
            max_leverage=self.max_leverage,
            vol_lookback=self.vol_lookback,
            embargo=self.horizon,
            model_names=self.models,
            pred_avg_window=self.pred_avg_window,
        )

    def sectors_kwargs(self) -> dict:
        """Return kwargs for run_sectors_pipeline()."""
        return dict(
            horizon=self.horizon,
            force_refresh=self.force_refresh,
            vol_target=self.vol_target,
            max_leverage=self.max_leverage,
            vol_lookback=self.vol_lookback,
            embargo=self.horizon,
            model_names=self.models,
            pred_avg_window=self.pred_avg_window,
        )

    def label(self) -> str:
        """Short human-readable label for session state / report filename."""
        parts = [self.universe.lower().replace(" ", "_"), f"h{self.horizon}"]
        if self.use_cot:
            parts.append("cot")
        if self.vol_target is not None:
            parts.append(f"vt{self.vol_target:.0%}")
        if self.pred_avg_window > 1:
            parts.append(f"pa{self.pred_avg_window}")
        return "_".join(parts)


# Model name → pipeline factory name mapping (must match pipeline_ranking.py keys)
ALL_MODELS = ["EqualWeight", "MomentumRank", "MeanReversion", "ElasticNet", "LightGBM", "LambdaMART"]
DEFAULT_MODELS = ["MeanReversion", "MomentumRank", "LightGBM"]

# Universe → human-readable label and baseline reference
UNIVERSE_META = {
    "Commodities": {
        "label": "9 commodity futures (GC=F, SI=F, PL=F, PA=F, CL=F, NG=F, HG=F, ZC=F, ZS=F)",
        "baseline_text": "Phase 14 OOS baseline — MeanReversion Sharpe ≈ 0.28, LightGBM h=63 Sharpe ≈ 0.79",
    },
    "Equity Sectors": {
        "label": "9 SPDR sector ETFs (XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY)",
        "baseline_text": "Phase 17 OOS baseline — MeanReversion Sharpe ≈ 0.15, LightGBM B3 h=63 Sharpe ≈ 0.41",
    },
    "Forex": {
        "label": "7 major USD pairs (EURUSD=X, GBPUSD=X, USDJPY=X, AUDUSD=X, USDCAD=X, USDCHF=X, NZDUSD=X)",
        "baseline_text": "Phase 18 OOS baseline — MeanReversion Sharpe ≈ 0.05, LightGBM PredAvg21 h=5 Sharpe ≈ 0.94",
    },
}
