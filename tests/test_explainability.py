# tests/test_explainability.py
import numpy as np
import pytest


class TestSHAPExplainer:
    def test_shap_values_shape(self):
        """SHAP values should match input feature shape."""
        from crystal_prop_bench.evaluation.explainability import compute_shap_values

        # Train a tiny model
        from crystal_prop_bench.models.lgbm_baseline import train_lgbm

        rng = np.random.RandomState(42)
        X = rng.randn(100, 5)
        y = X[:, 0] * 2 + rng.randn(100) * 0.1
        model, _ = train_lgbm(X[:80], y[:80], X[80:], y[80:], seed=42)

        shap_values = compute_shap_values(model, X[:10])
        assert shap_values.shape == (10, 5)

    def test_feature_importance_ranking(self):
        """Most important feature should be feature 0 (it determines y)."""
        from crystal_prop_bench.evaluation.explainability import (
            compute_shap_values,
            global_feature_importance,
        )
        from crystal_prop_bench.models.lgbm_baseline import train_lgbm

        rng = np.random.RandomState(42)
        X = rng.randn(200, 5)
        y = X[:, 0] * 5 + rng.randn(200) * 0.01
        model, _ = train_lgbm(X[:160], y[:160], X[160:], y[160:], seed=42)

        shap_values = compute_shap_values(model, X[:50])
        importance = global_feature_importance(shap_values, ["f0", "f1", "f2", "f3", "f4"])
        assert importance[0][0] == "f0"  # first element is most important


class TestFailureCases:
    def test_extracts_worst_predictions(self):
        from crystal_prop_bench.evaluation.explainability import extract_failure_cases

        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 10.0])  # last one is worst
        ids = np.array(["a", "b", "c", "d", "e"])

        failures = extract_failure_cases(y_true, y_pred, ids, n=2)
        assert len(failures) == 2
        assert failures.iloc[0]["material_id"] == "e"
