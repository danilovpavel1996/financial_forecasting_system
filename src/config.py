"""Load and validate config/config.yaml."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import yaml


ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "config.yaml"


@dataclass
class SplitterConfig:
    n_splits: int
    train_years: int
    test_years: int
    embargo: int  # trading days; must be >= horizon


@dataclass
class PathsConfig:
    data_raw: Path
    data_processed: Path
    outputs_figures: Path
    outputs_reports: Path
    outputs_models: Path


@dataclass
class Config:
    universe: dict  # {"metals": [...], "etfs": [...], "macro_context": [...]}
    fred_series: List[str]
    dates: dict    # {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}
    horizon: int
    cost_bps: float
    splitter: SplitterConfig
    random_seed: int
    paths: PathsConfig
    cftc: Dict = field(default_factory=dict)  # {"report_type": ..., "codes": {...}}

    def all_price_tickers(self) -> List[str]:
        """Return the flat list of all price tickers (metals + ETFs + macro context)."""
        tickers: List[str] = []
        for group in ("metals", "etfs", "macro_context"):
            tickers.extend(self.universe.get(group, []))
        return tickers


def load_config(path: Path = CONFIG_PATH) -> Config:
    """Load config.yaml and return a validated Config dataclass."""
    with open(path) as f:
        raw = yaml.safe_load(f)

    _validate_raw(raw)

    splitter_raw = raw["splitter"]
    splitter = SplitterConfig(
        n_splits=int(splitter_raw["n_splits"]),
        train_years=int(splitter_raw["train_years"]),
        test_years=int(splitter_raw["test_years"]),
        embargo=int(splitter_raw["embargo"]),
    )

    paths_raw = raw["paths"]
    paths = PathsConfig(
        data_raw=ROOT / paths_raw["data_raw"],
        data_processed=ROOT / paths_raw["data_processed"],
        outputs_figures=ROOT / paths_raw["outputs_figures"],
        outputs_reports=ROOT / paths_raw["outputs_reports"],
        outputs_models=ROOT / paths_raw["outputs_models"],
    )

    cfg = Config(
        universe=raw["universe"],
        fred_series=raw["fred_series"],
        dates=raw["dates"],
        horizon=int(raw["horizon"]),
        cost_bps=float(raw["cost_bps"]),
        splitter=splitter,
        random_seed=int(raw["random_seed"]),
        paths=paths,
        cftc=raw.get("cftc", {}),
    )

    _validate_config(cfg)
    return cfg


def _validate_raw(raw: dict) -> None:
    required = ("universe", "fred_series", "dates", "horizon", "cost_bps", "splitter", "random_seed", "paths")
    missing = [k for k in required if k not in raw]
    if missing:
        raise ValueError(f"config.yaml is missing required keys: {missing}")

    required_dates = ("start", "end")
    for k in required_dates:
        if k not in raw["dates"]:
            raise ValueError(f"config.yaml dates is missing key: {k}")

    required_splitter = ("n_splits", "train_years", "test_years", "embargo")
    for k in required_splitter:
        if k not in raw["splitter"]:
            raise ValueError(f"config.yaml splitter is missing key: {k}")


def _validate_config(cfg: Config) -> None:
    if cfg.horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {cfg.horizon}")

    if cfg.splitter.embargo < cfg.horizon:
        raise ValueError(
            f"embargo ({cfg.splitter.embargo}) must be >= horizon ({cfg.horizon}) "
            "to prevent label leakage across the train/test boundary."
        )

    if cfg.cost_bps < 0:
        raise ValueError(f"cost_bps must be >= 0, got {cfg.cost_bps}")

    if not cfg.all_price_tickers():
        raise ValueError("Universe must contain at least one price ticker.")

    if not cfg.fred_series:
        raise ValueError("fred_series must contain at least one series.")


if __name__ == "__main__":
    cfg = load_config()
    print(cfg)
