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

    def pipeline_kwargs(self) -> dict:
        """Return kwargs for run_ranking_pipeline()."""
        return dict(
            horizon=self.horizon,
            use_cot=self.use_cot,
            vol_target=self.vol_target,
            max_leverage=self.max_leverage,
            vol_lookback=self.vol_lookback,
            force_refresh=self.force_refresh,
        )

    def label(self) -> str:
        """Short human-readable label for session state / report filename."""
        parts = [f"h{self.horizon}"]
        if self.use_cot:
            parts.append("cot")
        if self.vol_target is not None:
            parts.append(f"vt{self.vol_target:.0%}")
        return "_".join(parts)


# Model name → pipeline factory name mapping (must match pipeline_ranking.py keys)
ALL_MODELS = ["EqualWeight", "MomentumRank", "MeanReversion", "ElasticNet", "LightGBM"]
DEFAULT_MODELS = ["MeanReversion", "MomentumRank", "LightGBM"]
