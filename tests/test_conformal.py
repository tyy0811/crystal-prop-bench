# tests/test_conformal.py
import numpy as np
import pytest

from crystal_prop_bench.evaluation.conformal import (
    conformal_regression_interval,
    evaluate_conformal_coverage,
)


class TestConformalRegressionInterval:
    def test_returns_lower_upper(self):
        cal_residuals = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        test_preds = np.array([1.0, 2.0])
        lower, upper = conformal_regression_interval(cal_residuals, test_preds, alpha=0.20)
        assert lower.shape == (2,)
        assert upper.shape == (2,)
        assert (upper > lower).all()

    def test_intervals_centered_on_predictions(self):
        cal_residuals = np.array([0.1, 0.2, 0.3])
        test_preds = np.array([5.0])
        lower, upper = conformal_regression_interval(cal_residuals, test_preds, alpha=0.10)
        midpoint = (lower[0] + upper[0]) / 2
        assert midpoint == pytest.approx(5.0)

    def test_coverage_on_uniform_residuals(self):
        """With enough uniform residuals, coverage should be close to 1-alpha."""
        rng = np.random.RandomState(42)
        n_cal = 10000
        cal_residuals = rng.uniform(0, 1, n_cal)
        test_preds = rng.randn(5000)
        test_true = test_preds + rng.uniform(-1, 1, 5000)

        lower, upper = conformal_regression_interval(cal_residuals, test_preds, alpha=0.10)
        covered = ((test_true >= lower) & (test_true <= upper)).mean()
        assert covered >= 0.88  # some slack for finite sample

    def test_zero_residuals_give_zero_width(self):
        cal_residuals = np.zeros(100)
        test_preds = np.array([1.0, 2.0])
        lower, upper = conformal_regression_interval(cal_residuals, test_preds, alpha=0.10)
        np.testing.assert_array_almost_equal(lower, test_preds)
        np.testing.assert_array_almost_equal(upper, test_preds)

    def test_alpha_affects_width(self):
        cal_residuals = np.linspace(0, 1, 100)
        test_preds = np.array([0.0])
        _, upper_10 = conformal_regression_interval(cal_residuals, test_preds, alpha=0.10)
        _, upper_30 = conformal_regression_interval(cal_residuals, test_preds, alpha=0.30)
        # Tighter alpha -> wider interval
        assert upper_10[0] > upper_30[0]


class TestEvaluateConformalCoverage:
    def test_returns_expected_keys(self):
        cal_residuals = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.05, 2.1, 3.15])
        result = evaluate_conformal_coverage(
            cal_residuals, y_true, y_pred, alphas=[0.10, 0.20]
        )
        assert len(result) == 2
        assert "alpha" in result[0]
        assert "coverage" in result[0]
        assert "mean_width" in result[0]

    def test_perfect_predictions_have_full_coverage(self):
        cal_residuals = np.array([0.1, 0.2, 0.3])
        y = np.array([1.0, 2.0, 3.0])
        result = evaluate_conformal_coverage(
            cal_residuals, y, y, alphas=[0.10]
        )
        assert result[0]["coverage"] == pytest.approx(1.0)
