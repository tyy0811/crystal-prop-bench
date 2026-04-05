"""Model utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PREDICTIONS_DIR = Path("results/predictions")
MODELS_DIR = Path("results/models")


def save_predictions(
    material_ids: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    families: np.ndarray,
    split_label: str,
    filename: str,
) -> None:
    """Save prediction parquet with standard columns."""
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "material_id": material_ids,
        "y_true": y_true,
        "y_pred": y_pred,
        "chemistry_family": families,
        "split": split_label,
    }).to_parquet(PREDICTIONS_DIR / filename, index=False)
