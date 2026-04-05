"""Evaluation metrics: MAE, RMSE, R², per-family breakdown, seed aggregation."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute MAE, RMSE, R² for a single run."""
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def compute_per_family_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    families: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Compute metrics broken down by chemistry family."""
    result = {}
    for family in np.unique(families):
        mask = families == family
        result[family] = compute_metrics(y_true[mask], y_pred[mask])
    return result


def aggregate_seeds(
    seed_results: list[dict[str, float]],
) -> dict[str, float]:
    """Aggregate per-seed metric dicts into mean +/- std.

    Input: [{"mae": 1.0, "rmse": 2.0}, {"mae": 2.0, "rmse": 3.0}, ...]
    Output: {"mae_mean": 1.5, "mae_std": 0.5, "rmse_mean": 2.5, "rmse_std": 0.5, ...}
    """
    if len(seed_results) == 0:
        return {}

    keys = seed_results[0].keys()
    result = {}
    for key in keys:
        values = [r[key] for r in seed_results]
        result[f"{key}_mean"] = float(np.mean(values))
        result[f"{key}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return result
