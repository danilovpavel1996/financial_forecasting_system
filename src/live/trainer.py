"""Train ranking models on the full historical dataset for live prediction.

Unlike the walk-forward backtest, the live model uses ALL available
historical data as training — including what was the "test" set in the
backtest. This maximises the information used for the live signal.

The out-of-sample Sharpe from the backtest (0.63) remains the honest
estimate of strategy quality. The live model just has more training data.

Saved model artefact:  outputs/models/live_lgbm_{timestamp}.pkl
  Contains: booster, feature_cols, training_date, horizon, n_training_rows
"""
from __future__ import annotations

import datetime
import glob
import joblib
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.config import Config
from src.data import universe
from src.features.pooled_dataset import build_pooled_dataset, feature_cols
from src.live.data import LiveData
from src.models.gbm import LightGBMModel

logger = logging.getLogger(__name__)

_MODEL_GLOB = "live_lgbm_*.pkl"


@dataclass
class TrainedModel:
    model: LightGBMModel
    feature_cols: list[str]
    trained_at: datetime.datetime
    horizon: int
    n_rows: int
    path: Path


def train_live_model(
    cfg: Config,
    live_data: LiveData,
    horizon: int,
    use_cot: bool = False,
) -> TrainedModel:
    """Train LightGBM on the full available history; save to outputs/models/.

    Uses build_pooled_dataset() — the exact same code path as the backtest —
    so features are identical and the model can predict on live feature vectors.
    """
    logger.info("Building pooled dataset for live training (horizon=%d) …", horizon)
    pooled = build_pooled_dataset(
        cfg,
        live_data.prices,
        live_data.macro_raw,
        horizon=horizon,
        cot_raw=live_data.cot_raw if use_cot else None,
    )

    fcols = feature_cols(pooled)
    X = pooled[fcols].values.astype(float)
    y = pooled["target"].values.astype(float)

    ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    X_train, y_train = X[ok], y[ok]

    logger.info("Training LightGBM on %d rows × %d features …", len(X_train), len(fcols))
    model = LightGBMModel(random_state=cfg.random_seed)
    model.fit(X_train, y_train)

    # Save artefact
    ts = datetime.datetime.now()
    ts_str = ts.strftime("%Y%m%d_%H%M%S")
    out_path = cfg.paths.outputs_models / f"live_lgbm_{ts_str}.pkl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model": model,
        "feature_cols": fcols,
        "trained_at": ts,
        "horizon": horizon,
        "n_rows": int(len(X_train)),
        "use_cot": use_cot,
    }
    joblib.dump(payload, out_path)
    logger.info("Model saved → %s", out_path)

    return TrainedModel(
        model=model,
        feature_cols=fcols,
        trained_at=ts,
        horizon=horizon,
        n_rows=int(len(X_train)),
        path=out_path,
    )


def load_latest_model(cfg: Config) -> TrainedModel | None:
    """Load the most recently saved live model, or None if none exist."""
    pattern = str(cfg.paths.outputs_models / _MODEL_GLOB)
    candidates = sorted(glob.glob(pattern))
    if not candidates:
        return None
    latest = candidates[-1]
    try:
        payload = joblib.load(latest)
        return TrainedModel(
            model=payload["model"],
            feature_cols=payload["feature_cols"],
            trained_at=payload["trained_at"],
            horizon=payload["horizon"],
            n_rows=payload["n_rows"],
            path=Path(latest),
        )
    except Exception as exc:
        logger.warning("Failed to load model %s: %s", latest, exc)
        return None


def model_is_stale(trained_model: TrainedModel | None, staleness_days: int) -> bool:
    """Return True if the model is older than staleness_days or doesn't exist."""
    if trained_model is None:
        return True
    age = datetime.datetime.now() - trained_model.trained_at
    return age.days >= staleness_days
