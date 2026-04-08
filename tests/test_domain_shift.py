# tests/test_domain_shift.py
import pytest

from crystal_prop_bench.evaluation.domain_shift import compute_degradation_ratios


class TestDegradationRatios:
    def test_returns_expected_keys(self):
        id_metrics = {"mae": 0.1, "rmse": 0.2}
        ood_metrics = {
            "sulfide": {"mae": 0.3, "rmse": 0.5},
            "nitride": {"mae": 0.4, "rmse": 0.6},
        }
        result = compute_degradation_ratios(id_metrics, ood_metrics)
        assert "sulfide" in result
        assert "nitride" in result
        assert "mae_ratio" in result["sulfide"]

    def test_correct_ratio(self):
        id_metrics = {"mae": 0.1}
        ood_metrics = {"sulfide": {"mae": 0.3}}
        result = compute_degradation_ratios(id_metrics, ood_metrics)
        assert result["sulfide"]["mae_ratio"] == pytest.approx(3.0)

    def test_delta_mae(self):
        id_metrics = {"mae": 0.1}
        ood_metrics = {"sulfide": {"mae": 0.3}}
        result = compute_degradation_ratios(id_metrics, ood_metrics)
        assert result["sulfide"]["mae_delta"] == pytest.approx(0.2)
