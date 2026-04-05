"""Tier 1 training: Magpie composition features + LightGBM.

Trains 3 seeds x 2 targets x 2 splits (standard + domain-shift).
Saves predictions to results/predictions/.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
import yaml

from crystal_prop_bench.data.featurizers import compute_magpie_features
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
    """Save prediction parquet with standard columns."""
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "material_id": material_ids,
        "y_true": y_true,
        "y_pred": y_pred,
        "chemistry_family": families,
        "split": split_label,
    }).to_parquet(PREDICTIONS_DIR / filename, index=False)


def run_standard_split(
    df: pd.DataFrame,
    features: pd.DataFrame,
    target: str,
    seeds: list[int],
) -> None:
    """Train and evaluate on standard split."""
    for seed in seeds:
        train_df, cal_df, test_df = standard_split(df, seed=seed)

        # Align features
        train_feat = features[features["material_id"].isin(train_df["material_id"])]
        cal_feat = features[features["material_id"].isin(cal_df["material_id"])]
        test_feat = features[features["material_id"].isin(test_df["material_id"])]

        # Merge to align rows
        train_merged = train_df.merge(train_feat, on="material_id")
        cal_merged = cal_df.merge(cal_feat, on="material_id")
        test_merged = test_df.merge(test_feat, on="material_id")

        feature_cols = [c for c in features.columns if c != "material_id"]

        X_train = train_merged[feature_cols].values
        y_train = train_merged[target].values
        X_cal = cal_merged[feature_cols].values
        y_cal = cal_merged[target].values
        X_test = test_merged[feature_cols].values
        y_test = test_merged[target].values

        with mlflow.start_run(run_name=f"tier1_standard_{target}_seed{seed}"):
            mlflow.log_params({"tier": 1, "split": "standard", "target": target, "seed": seed})

            model, cal_residuals = train_lgbm(X_train, y_train, X_cal, y_cal, seed=seed)
            test_preds = model.predict(X_test)

            metrics = compute_metrics(y_test, test_preds)
            mlflow.log_metrics(metrics)
            logger.info("Tier 1 standard %s seed=%d: %s", target, seed, metrics)

            # Save predictions
            target_short = "ef" if target == "formation_energy_per_atom" else "bg"
            save_predictions(
                test_merged["material_id"].values,
                y_test, test_preds,
                test_merged["chemistry_family"].values,
                "standard_test",
                f"tier1_standard_seed{seed}_{target_short}.parquet",
            )
            # Also save cal predictions for conformal
            cal_preds = model.predict(X_cal)
            save_predictions(
                cal_merged["material_id"].values,
                y_cal, cal_preds,
                cal_merged["chemistry_family"].values,
                "standard_cal",
                f"tier1_standard_seed{seed}_{target_short}_cal.parquet",
            )

            # Save model for SHAP
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, MODELS_DIR / f"tier1_standard_seed{seed}_{target_short}.joblib")


def run_domain_shift(
    df: pd.DataFrame,
    features: pd.DataFrame,
    target: str,
    seeds: list[int],
) -> None:
    """Train and evaluate on domain-shift split."""
    for seed in seeds:
        splits = domain_shift_split(df, seed=seed)

        train_feat = features[features["material_id"].isin(splits["train"]["material_id"])]
        cal_feat = features[features["material_id"].isin(splits["cal"]["material_id"])]

        train_merged = splits["train"].merge(train_feat, on="material_id")
        cal_merged = splits["cal"].merge(cal_feat, on="material_id")

        feature_cols = [c for c in features.columns if c != "material_id"]

        X_train = train_merged[feature_cols].values
        y_train = train_merged[target].values
        X_cal = cal_merged[feature_cols].values
        y_cal = cal_merged[target].values

        with mlflow.start_run(run_name=f"tier1_domshift_{target}_seed{seed}"):
            mlflow.log_params({"tier": 1, "split": "domain_shift", "target": target, "seed": seed})

            model, cal_residuals = train_lgbm(X_train, y_train, X_cal, y_cal, seed=seed)

            target_short = "ef" if target == "formation_energy_per_atom" else "bg"

            # Save cal predictions
            cal_preds = model.predict(X_cal)
            save_predictions(
                cal_merged["material_id"].values,
                y_cal, cal_preds,
                cal_merged["chemistry_family"].values,
                "domshift_cal",
                f"tier1_domshift_seed{seed}_{target_short}_cal.parquet",
            )

            # Predict on each test set
            for split_key in ["test_id", "test_ood_sulfide", "test_ood_nitride", "test_ood_halide"]:
                test_df = splits[split_key]
                test_feat = features[features["material_id"].isin(test_df["material_id"])]
                test_merged = test_df.merge(test_feat, on="material_id")

                if len(test_merged) == 0:
                    continue

                X_test = test_merged[feature_cols].values
                y_test = test_merged[target].values
                test_preds = model.predict(X_test)

                metrics = compute_metrics(y_test, test_preds)
                mlflow.log_metrics({f"{split_key}_{k}": v for k, v in metrics.items()})
                logger.info("Tier 1 domshift %s %s seed=%d: %s", target, split_key, seed, metrics)

                save_predictions(
                    test_merged["material_id"].values,
                    y_test, test_preds,
                    test_merged["chemistry_family"].values,
                    f"domshift_{split_key}",
                    f"tier1_domshift_seed{seed}_{target_short}_{split_key}.parquet",
                )

            # Save model
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, MODELS_DIR / f"tier1_domshift_seed{seed}_{target_short}.joblib")


def main() -> None:
    with open("configs/base.yaml") as f:
        config = yaml.safe_load(f)

    seeds = config["evaluation"]["seeds"]
    targets = config["evaluation"]["targets"]

    # Load data
    adapter = MPAdapter(cache_dir=Path("data/mp"))
    df = adapter.load()

    # Compute features
    features = compute_magpie_features(
        df, cache_path=Path("data/mp/magpie_features.parquet")
    )

    mlflow.set_experiment("crystal-prop-bench-tier1")

    for target in targets:
        run_standard_split(df, features, target, seeds)
        run_domain_shift(df, features, target, seeds)


if __name__ == "__main__":
    main()
