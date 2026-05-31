"""Tests for src/models/linear.py and src/models/gbm.py."""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from src.models.base import Model
from src.models.linear import ElasticNetModel
from src.models.gbm import LightGBMModel, LightGBMQuantileModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_regression_data(n_train=300, n_test=60, n_features=10, seed=0):
    rng = np.random.default_rng(seed)
    # Weak but non-zero signal in first two features
    w = np.zeros(n_features)
    w[:2] = [0.3, -0.2]
    X = rng.standard_normal((n_train + n_test, n_features))
    y = X @ w + 0.5 * rng.standard_normal(n_train + n_test)
    return (
        X[:n_train], y[:n_train],
        X[n_train:], y[n_train:],
    )


# ---------------------------------------------------------------------------
# ElasticNetModel
# ---------------------------------------------------------------------------

class TestElasticNetModel:
    def test_protocol_compliance(self):
        assert isinstance(ElasticNetModel(), Model)

    def test_predict_shape(self):
        Xtr, ytr, Xte, _ = _make_regression_data()
        m = ElasticNetModel()
        m.fit(Xtr, ytr)
        assert m.predict(Xte).shape == (len(Xte),)

    def test_predict_dtype_float(self):
        Xtr, ytr, Xte, _ = _make_regression_data()
        m = ElasticNetModel()
        m.fit(Xtr, ytr)
        assert m.predict(Xte).dtype == float

    def test_scaler_fit_on_train_only(self):
        """Scaler.mean_ must match train distribution, not test distribution."""
        rng = np.random.default_rng(7)
        # Train distribution: mean ~100; test distribution: mean ~-100
        X_train = rng.standard_normal((200, 5)) + 100.0
        X_test = rng.standard_normal((50, 5)) - 100.0
        y_train = rng.standard_normal(200)

        m = ElasticNetModel()
        m.fit(X_train, y_train)

        scaler_mean = m.scaler_.mean_
        train_mean = X_train.mean(axis=0)
        test_mean = X_test.mean(axis=0)

        assert np.allclose(scaler_mean, train_mean, atol=1e-10), (
            "Scaler must be fit on training data"
        )
        assert not np.allclose(scaler_mean, test_mean, atol=5.0), (
            "Scaler must NOT reflect test data statistics"
        )

    def test_scaler_does_not_see_test_data(self):
        """Predict on test data must not change the fitted scaler parameters."""
        Xtr, ytr, Xte, _ = _make_regression_data()
        m = ElasticNetModel()
        m.fit(Xtr, ytr)

        mean_before = m.scaler_.mean_.copy()
        _ = m.predict(Xte)
        assert np.array_equal(m.scaler_.mean_, mean_before)

    def test_predictions_not_all_zero_on_signal(self):
        """A model with a real signal should not predict zeros everywhere."""
        Xtr, ytr, Xte, _ = _make_regression_data(n_train=500)
        m = ElasticNetModel(alpha=0.001)
        m.fit(Xtr, ytr)
        assert not np.allclose(m.predict(Xte), 0.0, atol=1e-6)

    def test_coef_shape(self):
        Xtr, ytr, _, _ = _make_regression_data(n_features=8)
        m = ElasticNetModel()
        m.fit(Xtr, ytr)
        assert m.coef_.shape == (8,)

    def test_fit_returns_self(self):
        Xtr, ytr, _, _ = _make_regression_data()
        m = ElasticNetModel()
        assert m.fit(Xtr, ytr) is m

    def test_different_alpha_values(self):
        Xtr, ytr, Xte, _ = _make_regression_data()
        p_low = ElasticNetModel(alpha=0.0001).fit(Xtr, ytr).predict(Xte)
        p_high = ElasticNetModel(alpha=10.0).fit(Xtr, ytr).predict(Xte)
        # High alpha → heavy shrinkage → predictions closer to zero
        assert np.std(p_high) < np.std(p_low)

    def test_l1_ratio_pure_ridge(self):
        """l1_ratio=0 is Ridge; should still fit and predict."""
        Xtr, ytr, Xte, _ = _make_regression_data()
        m = ElasticNetModel(alpha=0.01, l1_ratio=0.0)
        m.fit(Xtr, ytr)
        assert m.predict(Xte).shape == (len(Xte),)

    def test_repr(self):
        m = ElasticNetModel(alpha=0.05, l1_ratio=0.3)
        assert "0.05" in repr(m)
        assert "0.3" in repr(m)


# ---------------------------------------------------------------------------
# LightGBMModel — internal split logic
# ---------------------------------------------------------------------------

class TestLightGBMSplit:
    def test_val_is_tail_not_random(self):
        """val_start must come after fit_end."""
        m = LightGBMModel(val_fraction=0.2, inner_embargo=5)
        fit_end, val_start = m._split_train_val(200)
        assert fit_end < val_start, "fit portion must precede val portion"
        assert fit_end >= 0

    def test_split_sizes(self):
        m = LightGBMModel(val_fraction=0.2, inner_embargo=5)
        n = 200
        fit_end, val_start = m._split_train_val(n)
        expected_val_size = max(20, int(n * 0.2))
        expected_val_start = n - expected_val_size
        expected_fit_end = expected_val_start - 5
        assert val_start == expected_val_start
        assert fit_end == expected_fit_end

    def test_inner_embargo_respected(self):
        """Gap between fit_end and val_start must equal inner_embargo."""
        for embargo in [1, 5, 10]:
            m = LightGBMModel(inner_embargo=embargo)
            fit_end, val_start = m._split_train_val(300)
            assert val_start - fit_end == embargo, (
                f"inner_embargo={embargo}: gap={val_start - fit_end}"
            )

    def test_degenerate_returns_neg1(self):
        """Too little data → (-1, -1) → fall back to no ES."""
        m = LightGBMModel(val_fraction=0.5, inner_embargo=10)
        fit_end, val_start = m._split_train_val(30)
        assert fit_end == -1
        assert val_start == -1

    def test_val_indices_are_strictly_after_fit_indices(self):
        """No overlap between fit data and validation data."""
        m = LightGBMModel(val_fraction=0.15, inner_embargo=5)
        n = 500
        fit_end, val_start = m._split_train_val(n)
        fit_idx = set(range(0, fit_end))
        val_idx = set(range(val_start, n))
        assert fit_idx.isdisjoint(val_idx)


# ---------------------------------------------------------------------------
# LightGBMModel — fit / predict
# ---------------------------------------------------------------------------

class TestLightGBMModel:
    def test_protocol_compliance(self):
        assert isinstance(LightGBMModel(), Model)

    def test_predict_shape(self):
        Xtr, ytr, Xte, _ = _make_regression_data()
        m = LightGBMModel(n_estimators=20)
        m.fit(Xtr, ytr)
        assert m.predict(Xte).shape == (len(Xte),)

    def test_predict_before_fit_raises(self):
        m = LightGBMModel()
        with pytest.raises(RuntimeError, match="before fit"):
            m.predict(np.ones((10, 5)))

    def test_fit_returns_self(self):
        Xtr, ytr, _, _ = _make_regression_data()
        m = LightGBMModel(n_estimators=20)
        assert m.fit(Xtr, ytr) is m

    def test_fallback_when_data_too_small(self):
        """Small dataset should not crash — falls back to no early stopping."""
        rng = np.random.default_rng(1)
        X = rng.standard_normal((50, 5))
        y = rng.standard_normal(50)
        m = LightGBMModel(n_estimators=10, val_fraction=0.5, inner_embargo=10)
        m.fit(X, y)
        assert m.predict(X[:5]).shape == (5,)

    def test_predictions_vary(self):
        """Model must produce non-constant predictions on a signal dataset."""
        Xtr, ytr, Xte, _ = _make_regression_data(n_train=500)
        m = LightGBMModel(n_estimators=50)
        m.fit(Xtr, ytr)
        preds = m.predict(Xte)
        assert preds.std() > 0.0

    def test_booster_is_set_after_fit(self):
        Xtr, ytr, _, _ = _make_regression_data()
        m = LightGBMModel(n_estimators=10)
        assert m._booster is None
        m.fit(Xtr, ytr)
        assert m._booster is not None

    def test_repr(self):
        m = LightGBMModel(n_estimators=200, num_leaves=15)
        r = repr(m)
        assert "200" in r
        assert "15" in r


# ---------------------------------------------------------------------------
# LightGBMQuantileModel
# ---------------------------------------------------------------------------

class TestLightGBMQuantileModel:
    def test_protocol_compliance(self):
        assert isinstance(LightGBMQuantileModel(), Model)

    def test_predict_shape(self):
        Xtr, ytr, Xte, _ = _make_regression_data()
        m = LightGBMQuantileModel(n_estimators=20)
        m.fit(Xtr, ytr)
        assert m.predict(Xte).shape == (len(Xte),)

    def test_quantile_objective_in_params(self):
        m = LightGBMQuantileModel(quantile_alpha=0.9)
        assert m._params["objective"] == "quantile"
        assert m._params["alpha"] == 0.9

    def test_quantile_differs_from_regression(self):
        """q90 predictions should differ from regression predictions."""
        Xtr, ytr, Xte, _ = _make_regression_data(n_train=500)
        m_reg = LightGBMModel(n_estimators=50)
        m_q90 = LightGBMQuantileModel(quantile_alpha=0.9, n_estimators=50)
        m_reg.fit(Xtr, ytr)
        m_q90.fit(Xtr, ytr)
        # q90 predictions should be higher on average than point estimates
        assert m_q90.predict(Xte).mean() > m_reg.predict(Xte).mean()

    def test_fit_returns_self(self):
        Xtr, ytr, _, _ = _make_regression_data()
        m = LightGBMQuantileModel(n_estimators=10)
        assert m.fit(Xtr, ytr) is m


# ---------------------------------------------------------------------------
# Integration: models run through backtester
# ---------------------------------------------------------------------------

class TestModelsInBacktester:
    def test_elasticnet_in_backtester(self):
        from src.eval.splitter import WalkForwardSplitter
        from src.eval.backtester import Backtester, comparison_table

        rng = np.random.default_rng(42)
        n = 600
        X = rng.standard_normal((n, 5))
        y = rng.standard_normal(n)

        splitter = WalkForwardSplitter(min_train=100, test_size=60, embargo=5, n_splits=3)
        bt = Backtester(splitter=splitter, cost_bps=5.0)
        result = bt.run(X, y, ElasticNetModel, model_name="ElasticNet")

        assert result.n_folds() == 3
        assert len(result.oos_pred()) == 180
        assert np.isfinite(result.pooled_ic) or np.isnan(result.pooled_ic)

    def test_lightgbm_in_backtester(self):
        from src.eval.splitter import WalkForwardSplitter
        from src.eval.backtester import Backtester

        rng = np.random.default_rng(42)
        n = 600
        X = rng.standard_normal((n, 5))
        y = rng.standard_normal(n)

        splitter = WalkForwardSplitter(min_train=100, test_size=60, embargo=5, n_splits=3)
        bt = Backtester(splitter=splitter, cost_bps=5.0)
        factory = lambda: LightGBMModel(n_estimators=20)
        result = bt.run(X, y, factory, model_name="LightGBM")

        assert result.n_folds() == 3
        assert len(result.oos_pred()) == 180

    def test_comparison_table_has_all_models(self):
        from src.eval.splitter import WalkForwardSplitter
        from src.eval.backtester import Backtester, comparison_table
        from src.models.baselines import RandomWalk, Drift

        rng = np.random.default_rng(0)
        n = 400
        X = rng.standard_normal((n, 4))
        y = rng.standard_normal(n)

        splitter = WalkForwardSplitter(min_train=80, test_size=40, embargo=5, n_splits=2)
        bt = Backtester(splitter=splitter, cost_bps=5.0)

        results = {
            "RandomWalk": bt.run(X, y, RandomWalk, "RandomWalk"),
            "ElasticNet": bt.run(X, y, ElasticNetModel, "ElasticNet"),
            "LightGBM": bt.run(X, y, lambda: LightGBMModel(n_estimators=10), "LightGBM"),
        }
        tbl = comparison_table(results)
        assert set(tbl.index) == {"RandomWalk", "ElasticNet", "LightGBM"}
