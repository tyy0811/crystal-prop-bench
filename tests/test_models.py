# tests/test_models.py
import numpy as np
import pytest

from crystal_prop_bench.models.lgbm_baseline import train_lgbm


class TestTrainLGBM:
    @pytest.fixture
    def synthetic_data(self):
        rng = np.random.RandomState(42)
        n_train, n_cal, n_features = 200, 50, 10
        X_train = rng.randn(n_train, n_features)
        y_train = X_train[:, 0] * 2 + rng.randn(n_train) * 0.1
        X_cal = rng.randn(n_cal, n_features)
        y_cal = X_cal[:, 0] * 2 + rng.randn(n_cal) * 0.1
        return X_train, y_train, X_cal, y_cal

    def test_returns_model_and_residuals(self, synthetic_data):
        X_train, y_train, X_cal, y_cal = synthetic_data
        model, cal_residuals = train_lgbm(
            X_train, y_train, X_cal, y_cal, seed=42
        )
        assert model is not None
        assert len(cal_residuals) == len(y_cal)

    def test_residuals_are_absolute(self, synthetic_data):
        X_train, y_train, X_cal, y_cal = synthetic_data
        _, cal_residuals = train_lgbm(
            X_train, y_train, X_cal, y_cal, seed=42
        )
        assert (cal_residuals >= 0).all()

    def test_model_predicts_correct_shape(self, synthetic_data):
        X_train, y_train, X_cal, y_cal = synthetic_data
        model, _ = train_lgbm(X_train, y_train, X_cal, y_cal, seed=42)
        preds = model.predict(X_cal)
        assert preds.shape == (len(X_cal),)

    def test_model_learns_linear_pattern(self, synthetic_data):
        X_train, y_train, X_cal, y_cal = synthetic_data
        model, _ = train_lgbm(X_train, y_train, X_cal, y_cal, seed=42)
        preds = model.predict(X_cal)
        mae = np.mean(np.abs(preds - y_cal))
        # Should achieve < 0.5 MAE on this simple linear problem
        assert mae < 0.5

    def test_deterministic_with_same_seed(self, synthetic_data):
        X_train, y_train, X_cal, y_cal = synthetic_data
        _, res1 = train_lgbm(X_train, y_train, X_cal, y_cal, seed=42)
        _, res2 = train_lgbm(X_train, y_train, X_cal, y_cal, seed=42)
        np.testing.assert_array_almost_equal(res1, res2)
