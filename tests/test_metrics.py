# tests/test_metrics.py
import numpy as np
import pytest

from crystal_prop_bench.evaluation.metrics import (
    aggregate_seeds,
    compute_metrics,
    compute_per_family_metrics,
)


class TestComputeMetrics:
    def test_returns_expected_keys(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.1, 3.1])
        result = compute_metrics(y_true, y_pred)
        assert "mae" in result
        assert "rmse" in result
        assert "r2" in result

    def test_perfect_prediction(self):
        y = np.array([1.0, 2.0, 3.0])
        result = compute_metrics(y, y)
        assert result["mae"] == pytest.approx(0.0)
        assert result["rmse"] == pytest.approx(0.0)
        assert result["r2"] == pytest.approx(1.0)

    def test_known_mae(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.5, 2.5, 3.5])
        result = compute_metrics(y_true, y_pred)
        assert result["mae"] == pytest.approx(0.5)


class TestPerFamilyMetrics:
    def test_returns_all_families(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.1, 2.1, 3.1, 4.1])
        families = np.array(["oxide", "oxide", "sulfide", "sulfide"])
        result = compute_per_family_metrics(y_true, y_pred, families)
        assert "oxide" in result
        assert "sulfide" in result

    def test_per_family_correct_values(self):
        y_true = np.array([1.0, 2.0, 10.0, 20.0])
        y_pred = np.array([1.0, 2.0, 11.0, 21.0])  # oxide perfect, sulfide MAE=1
        families = np.array(["oxide", "oxide", "sulfide", "sulfide"])
        result = compute_per_family_metrics(y_true, y_pred, families)
        assert result["oxide"]["mae"] == pytest.approx(0.0)
        assert result["sulfide"]["mae"] == pytest.approx(1.0)


class TestAggregateSeeds:
    def test_mean_and_std(self):
        seed_results = [
            {"mae": 1.0, "rmse": 2.0},
            {"mae": 2.0, "rmse": 3.0},
            {"mae": 3.0, "rmse": 4.0},
        ]
        result = aggregate_seeds(seed_results)
        assert result["mae_mean"] == pytest.approx(2.0)
        assert result["mae_std"] == pytest.approx(np.std([1.0, 2.0, 3.0], ddof=1))
        assert result["rmse_mean"] == pytest.approx(3.0)

    def test_single_seed(self):
        seed_results = [{"mae": 1.5}]
        result = aggregate_seeds(seed_results)
        assert result["mae_mean"] == pytest.approx(1.5)
        assert result["mae_std"] == pytest.approx(0.0)
