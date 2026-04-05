"""Domain-shift degradation analysis."""

from __future__ import annotations


def compute_degradation_ratios(
    id_metrics: dict[str, float],
    ood_metrics: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    """Compute degradation ratios and deltas for OOD families.

    Parameters
    ----------
    id_metrics : dict
        In-distribution metrics (e.g., {"mae": 0.1, "rmse": 0.2}).
    ood_metrics : dict of dict
        Per-family OOD metrics (e.g., {"sulfide": {"mae": 0.3}}).

    Returns
    -------
    Dict mapping family -> {"mae_ratio", "mae_delta", "rmse_ratio", "rmse_delta", ...}.
    """
    result = {}
    for family, metrics in ood_metrics.items():
        family_result = {}
        for key in metrics:
            if key in id_metrics and id_metrics[key] > 0:
                family_result[f"{key}_ratio"] = metrics[key] / id_metrics[key]
                family_result[f"{key}_delta"] = metrics[key] - id_metrics[key]
        result[family] = family_result
    return result
