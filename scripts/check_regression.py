"""CI regression gate: train on fixture, check MAE <= threshold.

Catches broken pipelines, not hyperparameter drift.
Threshold is set generously (~2x initial MAE).
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pandas as pd

from crystal_prop_bench.data.splits import standard_split
from crystal_prop_bench.evaluation.metrics import compute_metrics
from crystal_prop_bench.models.lgbm_baseline import train_lgbm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FIXTURES_DIR = Path("tests/fixtures")
THRESHOLDS_PATH = FIXTURES_DIR / "regression_thresholds.json"


def main() -> int:
    df = pd.read_parquet(FIXTURES_DIR / "fixture_crystals.parquet")
    features = pd.read_parquet(FIXTURES_DIR / "fixture_magpie_features.parquet")

    with open(THRESHOLDS_PATH) as f:
        thresholds = json.load(f)

    feature_cols = [c for c in features.columns if c != "material_id"]
    failures = []

    for target, threshold in thresholds.items():
        train_df, cal_df, test_df = standard_split(df, seed=42)

        train_m = train_df.merge(features, on="material_id")
        cal_m = cal_df.merge(features, on="material_id")
        test_m = test_df.merge(features, on="material_id")

        model, _ = train_lgbm(
            train_m[feature_cols].values, train_m[target].values,
            cal_m[feature_cols].values, cal_m[target].values,
            seed=42,
        )
        preds = model.predict(test_m[feature_cols].values)
        metrics = compute_metrics(test_m[target].values, preds)

        if metrics["mae"] > threshold:
            failures.append(
                f"{target}: MAE={metrics['mae']:.4f} > threshold={threshold}"
            )
            logger.error("REGRESSION: %s", failures[-1])
        else:
            logger.info(
                "PASS: %s MAE=%.4f <= %.4f", target, metrics["mae"], threshold
            )

    if failures:
        logger.error("Regression gate FAILED:\n%s", "\n".join(failures))
        return 1

    logger.info("Regression gate PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
