"""Split conformal regression intervals.

Implements Vovk et al. / Lei et al. split conformal prediction
for regression. Guarantees marginal coverage >= 1 - alpha on
exchangeable data.
"""

from __future__ import annotations

import numpy as np


def conformal_regression_interval(
    cal_residuals: np.ndarray,
    test_predictions: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute split conformal regression intervals.

    Parameters
    ----------
    cal_residuals : array of shape (n_cal,)
        Absolute residuals |y_cal - y_pred_cal| on calibration set.
    test_predictions : array of shape (n_test,)
        Point predictions on test set.
    alpha : float
        Miscoverage level. Intervals target 1 - alpha coverage.

    Returns
    -------
    (lower, upper) arrays of shape (n_test,).
    """
    n = len(cal_residuals)
    q = np.ceil((n + 1) * (1 - alpha)) / n
    q_hat = float(np.quantile(np.abs(cal_residuals), min(q, 1.0)))
    return test_predictions - q_hat, test_predictions + q_hat


def evaluate_conformal_coverage(
    cal_residuals: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    alphas: list[float] | None = None,
) -> list[dict[str, float]]:
    """Evaluate conformal coverage at multiple alpha levels.

    Returns list of dicts with keys: alpha, coverage, mean_width.
    """
    if alphas is None:
        alphas = [0.10, 0.20, 0.30]

    results = []
    for alpha in alphas:
        lower, upper = conformal_regression_interval(cal_residuals, y_pred, alpha)
        covered = ((y_true >= lower) & (y_true <= upper)).mean()
        width = (upper - lower).mean()
        results.append({
            "alpha": alpha,
            "coverage": float(covered),
            "mean_width": float(width),
        })
    return results
