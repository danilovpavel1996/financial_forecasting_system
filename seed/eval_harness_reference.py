"""
Layer 0 — Evaluation Harness for daily financial return forecasting.

This is the *moat*. Everything else (features, models, regime layers) plugs
into this. Its only job is to make it very hard to fool yourself:

  - Targets are FORWARD LOG RETURNS, never price levels.
  - Walk-forward (expanding) splits with an EMBARGO gap, so a model is never
    trained on labels whose look-ahead window overlaps the test set.
  - Every model is scored against the random-walk and drift baselines.
  - Strategy P&L is reported NET of transaction costs, with turnover, so a
    high-IC model that churns itself to death is exposed as such.

Design intent: each piece is swappable. Models only need a sklearn-style
.fit(X, y) / .predict(X). Drop in ElasticNet, LightGBM, or your TFT later
without touching the harness.

Author scaffold for Pavel — grow it from here.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable, Iterator, Protocol

import numpy as np
import pandas as pd
from scipy import stats


# --------------------------------------------------------------------------- #
# Target construction
# --------------------------------------------------------------------------- #
def forward_log_return(close: pd.Series, horizon: int = 1) -> pd.Series:
    """
    Forward log return over `horizon` days. Label at time t looks at prices
    in (t, t+horizon] -> it 'knows the future' up to t+horizon. The harness
    accounts for that leakage via the embargo below.

        r_t = log(close_{t+horizon} / close_t)

    The last `horizon` rows are NaN (no future available) and get dropped.
    """
    return np.log(close.shift(-horizon) / close)


# --------------------------------------------------------------------------- #
# Walk-forward splitting with embargo (anti-leakage)
# --------------------------------------------------------------------------- #
@dataclass
class WalkForwardSplitter:
    """
    Expanding-window walk-forward splits.

    The critical bit is `embargo`: because a label at time t uses prices up to
    t+horizon, the last `horizon` training labels straddle the train/test
    boundary and secretly contain test-period information. We drop a gap of
    `embargo` (>= horizon) samples between the end of training and the start of
    testing to kill that leak.

        min_train : minimum samples before the first test fold
        test_size : samples per test fold
        embargo   : gap between train end and test start (set >= horizon)
        expanding : True = expanding train window; False = fixed rolling window
        train_window : size of rolling window if expanding=False
    """

    min_train: int = 500
    test_size: int = 60
    embargo: int = 1
    expanding: bool = True
    train_window: int | None = None

    def split(self, n: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        if not self.expanding and self.train_window is None:
            raise ValueError("rolling mode (expanding=False) needs train_window")

        test_start = self.min_train + self.embargo
        while test_start + self.test_size <= n:
            train_end = test_start - self.embargo  # exclusive
            if self.expanding:
                train_start = 0
            else:
                train_start = max(0, train_end - self.train_window)

            train_idx = np.arange(train_start, train_end)
            test_idx = np.arange(test_start, test_start + self.test_size)
            yield train_idx, test_idx
            test_start += self.test_size


# --------------------------------------------------------------------------- #
# Model protocol + baselines you MUST beat
# --------------------------------------------------------------------------- #
class Model(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray) -> "Model": ...
    def predict(self, X: np.ndarray) -> np.ndarray: ...


class RandomWalk:
    """Predict zero return. The null hypothesis. If you can't beat this, stop."""

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.zeros(len(X))


class Drift:
    """Predict the trailing mean return seen in training."""

    def __init__(self):
        self.mu_ = 0.0

    def fit(self, X, y):
        self.mu_ = float(np.nanmean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mu_)


class Momentum:
    """
    Predict the sign/scale of the most recent feature column, assumed to be a
    momentum signal. Pure heuristic baseline — no fitting. Expects the LAST
    feature column to be a trailing return / momentum proxy.
    """

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.sign(X[:, -1]) * np.nanstd(X[:, -1])


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _safe_corr(pred: np.ndarray, realized: np.ndarray, kind: str) -> float:
    """Correlation that returns NaN (silently) when either side is constant."""
    m = np.isfinite(pred) & np.isfinite(realized)
    if m.sum() < 3 or np.std(pred[m]) == 0 or np.std(realized[m]) == 0:
        return np.nan
    fn = stats.pearsonr if kind == "pearson" else stats.spearmanr
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # near-constant preds -> NaN, expected
        try:
            r = float(fn(pred[m], realized[m])[0])
        except Exception:
            return np.nan
    return r if np.isfinite(r) else np.nan


def information_coefficient(pred: np.ndarray, realized: np.ndarray) -> float:
    """Pearson corr between predicted and realized returns."""
    return _safe_corr(pred, realized, "pearson")


def rank_ic(pred: np.ndarray, realized: np.ndarray) -> float:
    """Spearman (rank) IC — robust to outliers, what quants actually watch."""
    return _safe_corr(pred, realized, "spearman")


def directional_accuracy(pred: np.ndarray, realized: np.ndarray) -> float:
    """Hit rate on sign. 0.5 is a coin flip."""
    m = np.isfinite(pred) & np.isfinite(realized) & (pred != 0)
    if m.sum() == 0:
        return np.nan
    return float(np.mean(np.sign(pred[m]) == np.sign(realized[m])))


@dataclass
class StrategyStats:
    sharpe: float
    ann_return: float
    ann_vol: float
    max_drawdown: float
    turnover: float
    hit_rate: float


def strategy_pnl(
    pred: np.ndarray,
    realized: np.ndarray,
    cost_bps: float = 2.0,
    mode: str = "sign",
    periods_per_year: int = 252,
) -> StrategyStats:
    """
    Convert predictions to positions and compute NET-of-cost performance.

        mode = "sign"  -> +1 / -1 / 0 position (binary conviction)
        mode = "scaled"-> position proportional to prediction, clipped to [-1, 1]

    cost_bps is charged on |position change| (turnover). This is where naive
    daily strategies go to die: a 55%-hit-rate signal that flips every day pays
    its edge straight to the broker.
    """
    pred = np.asarray(pred, float)
    realized = np.asarray(realized, float)

    if mode == "sign":
        pos = np.sign(pred)
    elif mode == "scaled":
        s = np.nanstd(pred)
        pos = np.clip(pred / s, -1, 1) if s > 0 else np.zeros_like(pred)
    else:
        raise ValueError(mode)

    pos = np.nan_to_num(pos)
    turnover = np.abs(np.diff(pos, prepend=0.0))
    cost = turnover * (cost_bps / 1e4)

    net = pos * realized - cost
    net = net[np.isfinite(net)]
    if len(net) == 0 or np.std(net) == 0:
        return StrategyStats(np.nan, np.nan, np.nan, np.nan, float(turnover.mean()), np.nan)

    ann_return = float(np.mean(net) * periods_per_year)
    ann_vol = float(np.std(net) * np.sqrt(periods_per_year))
    sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan

    curve = np.cumprod(1 + net)
    peak = np.maximum.accumulate(curve)
    max_dd = float(np.min(curve / peak - 1))

    return StrategyStats(
        sharpe=sharpe,
        ann_return=ann_return,
        ann_vol=ann_vol,
        max_drawdown=max_dd,
        turnover=float(turnover.mean()),
        hit_rate=directional_accuracy(pred, realized),
    )


# --------------------------------------------------------------------------- #
# Backtester orchestrator
# --------------------------------------------------------------------------- #
@dataclass
class FoldResult:
    ic: float
    rank_ic: float
    hit: float
    pred: np.ndarray
    realized: np.ndarray


@dataclass
class Backtester:
    splitter: WalkForwardSplitter
    cost_bps: float = 2.0
    pnl_mode: str = "sign"

    def run(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_factory: Callable[[], Model],
    ) -> dict:
        """
        model_factory: a zero-arg callable returning a FRESH model each fold
        (so no state leaks across folds). e.g. lambda: ElasticNet(alpha=0.01)
        """
        folds: list[FoldResult] = []
        all_pred, all_real = [], []

        for tr, te in self.splitter.split(len(X)):
            Xtr, ytr = X[tr], y[tr]
            ok = np.isfinite(ytr) & np.all(np.isfinite(Xtr), axis=1)
            model = model_factory().fit(Xtr[ok], ytr[ok])
            p = model.predict(X[te])

            folds.append(
                FoldResult(
                    ic=information_coefficient(p, y[te]),
                    rank_ic=rank_ic(p, y[te]),
                    hit=directional_accuracy(p, y[te]),
                    pred=p,
                    realized=y[te],
                )
            )
            all_pred.append(p)
            all_real.append(y[te])

        all_pred = np.concatenate(all_pred)
        all_real = np.concatenate(all_real)
        strat = strategy_pnl(all_pred, all_real, self.cost_bps, self.pnl_mode)

        ics = [f.ic for f in folds if np.isfinite(f.ic)]
        rics = [f.rank_ic for f in folds if np.isfinite(f.rank_ic)]
        return {
            "n_folds": len(folds),
            "mean_ic": float(np.mean(ics)) if ics else np.nan,
            "mean_rank_ic": float(np.mean(rics)) if rics else np.nan,
            "ic_stability": (  # fraction of folds with positive IC
                float(np.mean([i > 0 for i in ics])) if ics else np.nan
            ),
            "pooled_ic": information_coefficient(all_pred, all_real),
            "strategy": strat,
        }


def compare(results: dict[str, dict]) -> pd.DataFrame:
    """Tabulate models side by side. Baselines included so the comparison is honest."""
    rows = []
    for name, r in results.items():
        s: StrategyStats = r["strategy"]
        rows.append(
            {
                "model": name,
                "mean_IC": round(r["mean_ic"], 4),
                "rank_IC": round(r["mean_rank_ic"], 4),
                "IC_stability": round(r["ic_stability"], 2),
                "hit_rate": round(s.hit_rate, 3) if np.isfinite(s.hit_rate) else np.nan,
                "Sharpe_net": round(s.sharpe, 2) if np.isfinite(s.sharpe) else np.nan,
                "ann_ret": round(s.ann_return, 3) if np.isfinite(s.ann_return) else np.nan,
                "max_dd": round(s.max_drawdown, 3) if np.isfinite(s.max_drawdown) else np.nan,
                "turnover": round(s.turnover, 3),
            }
        )
    return pd.DataFrame(rows).set_index("model")


# --------------------------------------------------------------------------- #
# Demo (synthetic data so it runs with zero dependencies / no network)
# --------------------------------------------------------------------------- #
def _demo():
    rng = np.random.default_rng(7)
    n = 2600  # ~10 years daily

    # Synthetic price with weak, decaying momentum signal + heavy noise,
    # so baselines are genuinely hard to beat (as in real markets).
    rets = np.zeros(n)
    for t in range(1, n):
        momentum = 0.04 * rets[t - 1]          # tiny real autocorrelation
        rets[t] = momentum + rng.normal(0, 0.012)
    close = pd.Series(100 * np.exp(np.cumsum(rets)))

    # Features: a few lagged returns + a trailing momentum (last column = momentum)
    df = pd.DataFrame({"close": close})
    df["r1"] = np.log(df.close / df.close.shift(1))
    df["r5"] = np.log(df.close / df.close.shift(5))
    df["mom10"] = df["r1"].rolling(10).mean()   # momentum proxy -> last col
    feats = ["r1", "r5", "mom10"]

    horizon = 1
    df["y"] = forward_log_return(df.close, horizon)
    df = df.dropna()

    X = df[feats].to_numpy()
    y = df["y"].to_numpy()

    splitter = WalkForwardSplitter(min_train=500, test_size=60, embargo=horizon)
    bt = Backtester(splitter, cost_bps=2.0, pnl_mode="sign")

    factories = {
        "RandomWalk": RandomWalk,
        "Drift": Drift,
        "Momentum": Momentum,
    }
    # Optional: add a real model if sklearn is around
    try:
        from sklearn.linear_model import ElasticNet

        factories["ElasticNet"] = lambda: ElasticNet(alpha=1e-4, l1_ratio=0.5)
    except ImportError:
        pass

    results = {name: bt.run(X, y, fac) for name, fac in factories.items()}
    print(compare(results).to_string())
    print(
        "\nRead it like a skeptic: a model only 'works' if its rank_IC and "
        "Sharpe_net clear the baselines AND IC_stability is well above 0.5.\n"
        "On this synthetic series the real signal is deliberately tiny, so "
        "edges should look marginal — that's the honest baseline feel of daily data."
    )


if __name__ == "__main__":
    _demo()
