"""Tier 2 training: Voronoi structural features + LightGBM.

Also runs Tier 1 on the Voronoi-survivable subset for bias check.
Saves predictions to results/predictions/.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
import yaml

from crystal_prop_bench.data.featurizers import (
    compute_magpie_features,
    compute_voronoi_features,
)
from crystal_prop_bench.data.mp_adapter import MPAdapter
from crystal_prop_bench.data.splits import domain_shift_split, standard_split
from crystal_prop_bench.evaluation.metrics import compute_metrics
from crystal_prop_bench.models.lgbm_baseline import train_lgbm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results")
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
MODELS_DIR = RESULTS_DIR / "models"


def save_predictions(
    material_ids: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    families: np.ndarray,
    split_label: str,
    filename: str,
) -> None:
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "material_id": material_ids,
        "y_true": y_true,
        "y_pred": y_pred,
        "chemistry_family": families,
        "split": split_label,
    }).to_parquet(PREDICTIONS_DIR / filename, index=False)


def train_and_predict(
    train_df: pd.DataFrame,
    cal_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: pd.DataFrame,
    target: str,
    seed: int,
    tier_label: str,
    split_label: str,
    test_split_key: str,
) -> None:
    """Generic train-predict-save loop."""
    feature_cols = [c for c in features.columns if c != "material_id"]

    train_m = train_df.merge(features, on="material_id")
    cal_m = cal_df.merge(features, on="material_id")
    test_m = test_df.merge(features, on="material_id")

    if len(train_m) == 0 or len(test_m) == 0:
        return

    model, _ = train_lgbm(
        train_m[feature_cols].values, train_m[target].values,
        cal_m[feature_cols].values, cal_m[target].values,
        seed=seed,
    )

    test_preds = model.predict(test_m[feature_cols].values)
    target_short = "ef" if target == "formation_energy_per_atom" else "bg"

    save_predictions(
        test_m["material_id"].values,
        test_m[target].values, test_preds,
        test_m["chemistry_family"].values,
        f"{split_label}_{test_split_key}",
        f"{tier_label}_{split_label}_seed{seed}_{target_short}_{test_split_key}.parquet",
    )

    # Save cal predictions
    cal_preds = model.predict(cal_m[feature_cols].values)
    save_predictions(
        cal_m["material_id"].values,
        cal_m[target].values, cal_preds,
        cal_m["chemistry_family"].values,
        f"{split_label}_cal",
        f"{tier_label}_{split_label}_seed{seed}_{target_short}_cal.parquet",
    )

    metrics = compute_metrics(test_m[target].values, test_preds)
    mlflow.log_metrics({f"{test_split_key}_{k}": v for k, v in metrics.items()})
    logger.info("%s %s %s %s seed=%d: %s", tier_label, split_label, target, test_split_key, seed, metrics)

    if test_split_key in ("test", "test_id"):
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, MODELS_DIR / f"{tier_label}_{split_label}_seed{seed}_{target_short}.joblib")


def main() -> None:
    with open("configs/base.yaml") as f:
        config = yaml.safe_load(f)

    seeds = config["evaluation"]["seeds"]
    targets = config["evaluation"]["targets"]

    # Load data and structures
    adapter = MPAdapter(cache_dir=Path("data/mp"))
    df = adapter.load()

    with open(adapter.cache_path() / "structures.pkl", "rb") as f:
        structures = pickle.load(f)

    # Compute Voronoi features (includes Magpie)
    voronoi_features = compute_voronoi_features(
        df, structures, cache_path=Path("data/mp/voronoi_features.parquet")
    )

    # Also get pure Magpie features for bias check
    magpie_features = compute_magpie_features(
        df, cache_path=Path("data/mp/magpie_features.parquet")
    )

    # Voronoi-survivable subset
    voronoi_ids = set(voronoi_features["material_id"])
    df_voronoi = df[df["material_id"].isin(voronoi_ids)].copy()
    magpie_voronoi = magpie_features[magpie_features["material_id"].isin(voronoi_ids)]

    logger.info(
        "Voronoi subset: %d/%d crystals (%.1f%%)",
        len(df_voronoi), len(df),
        100.0 * len(df_voronoi) / len(df),
    )

    mlflow.set_experiment("crystal-prop-bench-tier2")

    for target in targets:
        for seed in seeds:
            # --- Standard split ---
            train, cal, test = standard_split(df_voronoi, seed=seed)

            with mlflow.start_run(run_name=f"tier2_standard_{target}_seed{seed}"):
                mlflow.log_params({"tier": 2, "split": "standard", "target": target, "seed": seed})
                train_and_predict(train, cal, test, voronoi_features, target, seed, "tier2", "standard", "test")

            # Bias check: Tier 1 on Voronoi subset
            with mlflow.start_run(run_name=f"tier1_voronoi_subset_standard_{target}_seed{seed}"):
                mlflow.log_params({"tier": "1_voronoi_subset", "split": "standard", "target": target, "seed": seed})
                train_and_predict(train, cal, test, magpie_voronoi, target, seed, "tier1sub", "standard", "test")

            # --- Domain-shift split ---
            splits = domain_shift_split(df_voronoi, seed=seed)

            with mlflow.start_run(run_name=f"tier2_domshift_{target}_seed{seed}"):
                mlflow.log_params({"tier": 2, "split": "domain_shift", "target": target, "seed": seed})
                for split_key in ["test_id", "test_ood_sulfide", "test_ood_nitride", "test_ood_halide"]:
                    train_and_predict(
                        splits["train"], splits["cal"], splits[split_key],
                        voronoi_features, target, seed, "tier2", "domshift", split_key,
                    )


if __name__ == "__main__":
    main()
