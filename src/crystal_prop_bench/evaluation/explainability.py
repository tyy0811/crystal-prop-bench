"""SHAP explainability and failure-case analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap


def compute_shap_values(
    model: object,
    X: np.ndarray,
) -> np.ndarray:
    """Compute SHAP values using TreeExplainer.

    Returns array of shape (n_samples, n_features).
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    return np.array(shap_values)


def global_feature_importance(
    shap_values: np.ndarray,
    feature_names: list[str],
) -> list[tuple[str, float]]:
    """Rank features by mean |SHAP value|.

    Returns list of (feature_name, mean_abs_shap) sorted descending.
    """
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    ranked = sorted(
        zip(feature_names, mean_abs, strict=True),
        key=lambda x: x[1],
        reverse=True,
    )
    return ranked


def extract_failure_cases(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    material_ids: np.ndarray,
    n: int = 50,
    metadata: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Extract the N worst-predicted crystals.

    Returns DataFrame sorted by absolute error descending.
    """
    errors = np.abs(y_true - y_pred)
    indices = np.argsort(errors)[::-1][:n]

    result = pd.DataFrame({
        "material_id": material_ids[indices],
        "y_true": y_true[indices],
        "y_pred": y_pred[indices],
        "abs_error": errors[indices],
    })

    if metadata is not None:
        result = result.merge(metadata, on="material_id", how="left")

    return result.reset_index(drop=True)
