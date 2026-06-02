"""Live signal generator — today's commodity ranking.

Produces a LiveSignal dataclass containing:
  - Rankings (best → worst predicted forward return)
  - Position sheet (LONG / SHORT / FLAT)
  - Confidence context (vol regime, backtest stats, model staleness)

Two models supported:
  MeanReversion — ranks by negative 5d trailing return (no model file needed)
  LightGBM      — uses the trained model from trainer.py
"""
from __future__ import annotations

import dataclasses
import datetime
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.data import universe
from src.config import Config
from src.live.data import LiveData
from src.live.features import align_to_training_cols, build_live_features, get_feature_date
from src.live.trainer import TrainedModel
from src.models.vol_forecast import EWMAVolForecast, realized_vol

logger = logging.getLogger(__name__)

# Ticker → human-readable name
_NAMES: dict[str, str] = {
    "GC=F": "Gold",
    "SI=F": "Silver",
    "PL=F": "Platinum",
    "PA=F": "Palladium",
    "CL=F": "Crude Oil",
    "NG=F": "Nat Gas",
    "HG=F": "Copper",
    "ZC=F": "Corn",
    "ZS=F": "Soybeans",
    # Equity sector ETFs
    "XLB": "Materials",
    "XLE": "Energy",
    "XLF": "Financials",
    "XLI": "Industrials",
    "XLK": "Technology",
    "XLP": "Staples",
    "XLU": "Utilities",
    "XLV": "Health Care",
    "XLY": "Cons. Disc.",
}

def _n_long_short(n_assets: int) -> tuple[int, int]:
    """Return (n_long, n_short) matching the backtest pipeline rule.

    ≥9 assets → 3/3  (15-pair forex, 10-token crypto)
    ≥6 assets → 2/2  (7-pair forex, 9-asset commodity/sector)
    <6 assets → 1/1
    """
    if n_assets >= 9:
        return 3, 3
    if n_assets >= 6:
        return 2, 2
    return 1, 1


@dataclasses.dataclass
class AssetRanking:
    ticker: str
    name: str
    predicted_return: float
    rank: int                 # 1 = best, N = worst
    position: str             # "LONG", "SHORT", "FLAT"
    recent_mom_5d: float      # trailing 5-day return (context only)
    current_vol: float        # annualised EWMA vol (context only)


@dataclasses.dataclass
class SignalConfidence:
    backtest_sharpe: float    # out-of-sample Sharpe from most recent backtest
    backtest_cs_ric: float    # mean CS-RIC from most recent backtest
    days_since_retrain: int
    current_vol_regime: str   # "low" / "normal" / "high"
    vol_scale: float          # vol-targeting position scale (1.0 = no scaling)


@dataclasses.dataclass
class LiveSignal:
    date: datetime.date
    feature_date: datetime.date   # date of the feature row used
    horizon: int
    model_name: str
    model_trained_on: Optional[datetime.date]
    rankings: list[AssetRanking]
    confidence: SignalConfidence


def generate_signal(
    cfg: Config,
    live_data: LiveData,
    trained_model: Optional[TrainedModel],
    horizon: int,
    model_name: str = "MeanReversion",
    use_cot: bool = False,
    backtest_sharpe: float = 0.63,
    backtest_cs_ric: float = 0.020,
    vol_target: Optional[float] = 0.10,
    ranked_override: Optional[list[str]] = None,
    context_override: Optional[list[str]] = None,
    late_close_override: Optional[set[str]] = None,
    ref_ticker_override: Optional[str] = None,
) -> LiveSignal:
    """Build today's ranked signal for all 9 commodities.

    Parameters
    ----------
    cfg:            project config.
    live_data:      today's prices/macro/COT from fetch_live_data().
    trained_model:  loaded or freshly trained LightGBM; None → MeanReversion.
    horizon:        forecast horizon (trading days).
    model_name:     "MeanReversion" or "LightGBM".
    use_cot:        must match training config.
    backtest_sharpe/cs_ric: from the most recent walk-forward backtest.
    vol_target:     for vol-targeting scale computation (None = no targeting).
    """
    ranked = ranked_override if ranked_override is not None else universe.ranked_tickers(cfg)
    prices = live_data.prices

    # ── Build live feature vectors ────────────────────────────────────────────
    live_feats = build_live_features(
        cfg, live_data, use_cot=use_cot,
        ranked_override=ranked_override,
        context_override=context_override,
        late_close_override=late_close_override,
        ref_ticker_override=ref_ticker_override,
    )
    if not live_feats:
        raise RuntimeError("No live features could be built — check data availability.")

    feature_date = get_feature_date(live_feats).date()

    # ── Compute per-asset EWMA vol and momentum ───────────────────────────────
    ewma_model = EWMAVolForecast(lam=0.94)
    asset_vols: dict[str, float] = {}
    asset_mom5: dict[str, float] = {}

    for ticker in ranked:
        if ticker not in prices:
            asset_vols[ticker] = np.nan
            asset_mom5[ticker] = np.nan
            continue
        close = prices[ticker]["Close"].dropna()
        log_ret = np.log(close / close.shift(1)).dropna()
        ewma = ewma_model.fit_predict(log_ret).dropna()
        asset_vols[ticker] = float(ewma.iloc[-1]) if len(ewma) else np.nan
        # 5-day momentum from last available closes
        if len(close) >= 6:
            asset_mom5[ticker] = float(np.log(close.iloc[-1] / close.iloc[-6]))
        else:
            asset_mom5[ticker] = np.nan

    # ── Predictions ───────────────────────────────────────────────────────────
    preds: dict[str, float] = {}

    if model_name == "MeanReversion" or trained_model is None:
        # Rank by negative 5-day return (mean-reversion heuristic)
        for ticker in ranked:
            mom5 = asset_mom5.get(ticker, np.nan)
            preds[ticker] = -mom5 if np.isfinite(mom5) else 0.0
        model_name = "MeanReversion"
        model_trained_on = None

    else:
        # LightGBM: align feature vectors to training column order
        aligned = align_to_training_cols(live_feats, trained_model.feature_cols)
        for ticker in ranked:
            if ticker not in aligned:
                preds[ticker] = 0.0
                continue
            x = aligned[ticker].reshape(1, -1)
            if not np.all(np.isfinite(x)):
                n_nan = int(np.isnan(x).sum())
                logger.warning("%s: %d NaN features → prediction may be unreliable.", ticker, n_nan)
            preds[ticker] = float(trained_model.model.predict(x)[0])
        model_trained_on = trained_model.trained_at.date()

    # ── Ranking and positions ─────────────────────────────────────────────────
    sorted_tickers = sorted(ranked, key=lambda t: preds[t], reverse=True)
    n_long, n_short = _n_long_short(len(ranked))
    long_set  = set(sorted_tickers[:n_long])
    short_set = set(sorted_tickers[-n_short:])

    rankings: list[AssetRanking] = []
    for i, ticker in enumerate(sorted_tickers):
        pos = "LONG" if ticker in long_set else ("SHORT" if ticker in short_set else "FLAT")
        rankings.append(AssetRanking(
            ticker=ticker,
            name=_NAMES.get(ticker, ticker),
            predicted_return=preds[ticker],
            rank=i + 1,
            position=pos,
            recent_mom_5d=asset_mom5.get(ticker, np.nan),
            current_vol=asset_vols.get(ticker, np.nan),
        ))

    # ── Vol regime and targeting ──────────────────────────────────────────────
    vols = [v for v in asset_vols.values() if np.isfinite(v)]
    avg_vol = float(np.mean(vols)) if vols else np.nan

    if np.isfinite(avg_vol):
        if avg_vol < 0.18:
            vol_regime = "low"
        elif avg_vol > 0.30:
            vol_regime = "high"
        else:
            vol_regime = "normal"
    else:
        vol_regime = "unknown"

    vol_scale = 1.0
    if vol_target is not None and np.isfinite(avg_vol) and avg_vol > 1e-6:
        raw_scale = vol_target / avg_vol
        vol_scale = float(np.clip(raw_scale, 0.0, 2.0))

    days_since = (
        (datetime.datetime.now() - trained_model.trained_at).days
        if trained_model is not None else 0
    )

    confidence = SignalConfidence(
        backtest_sharpe=backtest_sharpe,
        backtest_cs_ric=backtest_cs_ric,
        days_since_retrain=days_since,
        current_vol_regime=vol_regime,
        vol_scale=vol_scale,
    )

    return LiveSignal(
        date=live_data.as_of,
        feature_date=feature_date,
        horizon=horizon,
        model_name=model_name,
        model_trained_on=model_trained_on if trained_model else None,
        rankings=rankings,
        confidence=confidence,
    )


def save_signal(signal: LiveSignal, output_dir: Path) -> Path:
    """Persist signal as JSON to outputs/signals/signal_YYYY-MM-DD.json."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"signal_{signal.date.strftime('%Y-%m-%d')}.json"

    payload = {
        "date": signal.date.isoformat(),
        "feature_date": signal.feature_date.isoformat(),
        "horizon": signal.horizon,
        "model_name": signal.model_name,
        "model_trained_on": signal.model_trained_on.isoformat() if signal.model_trained_on else None,
        "rankings": [
            {
                "ticker": r.ticker,
                "name": r.name,
                "predicted_return": round(r.predicted_return, 6),
                "rank": r.rank,
                "position": r.position,
                "recent_mom_5d": round(r.recent_mom_5d, 6) if np.isfinite(r.recent_mom_5d) else None,
                "current_vol": round(r.current_vol, 4) if np.isfinite(r.current_vol) else None,
            }
            for r in signal.rankings
        ],
        "confidence": {
            "backtest_sharpe": signal.confidence.backtest_sharpe,
            "backtest_cs_ric": signal.confidence.backtest_cs_ric,
            "days_since_retrain": signal.confidence.days_since_retrain,
            "current_vol_regime": signal.confidence.current_vol_regime,
            "vol_scale": round(signal.confidence.vol_scale, 3),
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Signal saved → %s", path)
    return path


def format_signal(signal: LiveSignal) -> str:
    """Return a formatted multi-line string suitable for CLI printing."""
    w = 58
    border = "═" * w
    lines = [
        border,
        f"  LIVE SIGNAL — {signal.date}  —  horizon: {signal.horizon} trading days",
        f"  Model: {signal.model_name}" + (
            f"  (trained: {signal.model_trained_on})" if signal.model_trained_on else ""
        ),
        f"  Features as of: {signal.feature_date}",
        border,
        "",
    ]

    longs  = [r for r in signal.rankings if r.position == "LONG"]
    shorts = [r for r in signal.rankings if r.position == "SHORT"]
    flats  = [r for r in signal.rankings if r.position == "FLAT"]

    def _row(r: AssetRanking) -> str:
        pred = f"{r.predicted_return:+.4f}" if np.isfinite(r.predicted_return) else "   N/A"
        mom  = f"{r.recent_mom_5d:+.2%}"   if np.isfinite(r.recent_mom_5d)  else "  N/A"
        vol  = f"{r.current_vol:.0%}"        if np.isfinite(r.current_vol)     else "N/A"
        return (f"    {r.rank:>2}. {r.ticker:<6}  ({r.name:<10})  "
                f"pred: {pred}   mom_5d: {mom}   vol: {vol}")

    if longs:
        lines.append("  ▲ LONG:")
        lines.extend(_row(r) for r in longs)
        lines.append("")

    if shorts:
        lines.append("  ▼ SHORT:")
        lines.extend(_row(r) for r in shorts)
        lines.append("")

    if flats:
        ticker_list = ", ".join(r.ticker for r in flats)
        lines.append(f"  ── FLAT: {ticker_list}")
        lines.append("")

    c = signal.confidence
    lines += [
        "  ── Context " + "─" * (w - 13),
        f"  Vol regime: {c.current_vol_regime:<8}  vol_scale: {c.vol_scale:.2f}×",
        f"  Backtest Sharpe: {c.backtest_sharpe:.2f}   CS-RIC: {c.backtest_cs_ric:+.4f}",
        f"  Model staleness: {c.days_since_retrain} day(s)",
        "",
        "  ⚠  Research signal only — not investment advice.",
        border,
    ]
    return "\n".join(lines)
