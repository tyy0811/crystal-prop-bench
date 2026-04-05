"""Tier 1 training: Magpie composition features + LightGBM.

Trains 3 seeds x 2 targets x 3 splits (standard + domain-shift + mixed-train).
Saves predictions to results/predictions/.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import mlflow
import numpy as np
import yaml

from crystal_prop_bench.data.featurizers import compute_magpie_features
from crystal_prop_bench.data.mp_adapter import MPAdapter
from crystal_prop_bench.data.splits import domain_shift_split, mixed_train_split, standard_split
from crystal_prop_bench.evaluation.metrics import compute_metrics
from crystal_prop_bench.models import MODELS_DIR, save_predictions
from crystal_prop_bench.models.lgbm_baseline import train_lgbm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _merge_features(df, features, feature_cols, target):
    """Merge split DataFrame with features, return X, y, and merged df."""
    merged = df.merge(features, on="material_id")
    return merged[feature_cols].values, merged[target].values, merged


def run_standard_split(df, features, target, seeds, model_params):
    """Train and evaluate on standard 4-way split."""
    feature_cols = [c for c in features.columns if c != "material_id"]

    for seed in seeds:
        train_df, val_df, cal_df, test_df = standard_split(df, seed=seed)

        X_train, y_train, _ = _merge_features(train_df, features, feature_cols, target)
        X_val, y_val, _ = _merge_features(val_df, features, feature_cols, target)
        X_cal, y_cal, cal_merged = _merge_features(cal_df, features, feature_cols, target)
        X_test, y_test, test_merged = _merge_features(test_df, features, feature_cols, target)

        target_short = "ef" if target == "formation_energy_per_atom" else "bg"

        with mlflow.start_run(run_name=f"tier1_standard_{target}_seed{seed}"):
            mlflow.log_params({"tier": 1, "split": "standard", "target": target, "seed": seed})

            model, cal_residuals = train_lgbm(
                X_train, y_train, X_val, y_val, X_cal, y_cal, seed=seed, **model_params,
            )
            test_preds = model.predict(X_test)

            metrics = compute_metrics(y_test, test_preds)
            mlflow.log_metrics(metrics)
            logger.info("Tier 1 standard %s seed=%d: %s", target, seed, metrics)

            save_predictions(
                test_merged["material_id"].values, y_test, test_preds,
                test_merged["chemistry_family"].values,
                "standard_test", f"tier1_standard_seed{seed}_{target_short}_test.parquet",
            )
            cal_preds = model.predict(X_cal)
            save_predictions(
                cal_merged["material_id"].values, y_cal, cal_preds,
                cal_merged["chemistry_family"].values,
                "standard_cal", f"tier1_standard_seed{seed}_{target_short}_cal.parquet",
            )

            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, MODELS_DIR / f"tier1_standard_seed{seed}_{target_short}.joblib")


def run_domain_shift(df, features, target, seeds, model_params):
    """Train and evaluate on domain-shift split."""
    feature_cols = [c for c in features.columns if c != "material_id"]

    for seed in seeds:
        splits = domain_shift_split(df, seed=seed)

        X_train, y_train, _ = _merge_features(splits["train"], features, feature_cols, target)
        X_val, y_val, _ = _merge_features(splits["val"], features, feature_cols, target)
        X_cal, y_cal, cal_merged = _merge_features(splits["cal"], features, feature_cols, target)

        target_short = "ef" if target == "formation_energy_per_atom" else "bg"

        with mlflow.start_run(run_name=f"tier1_domshift_{target}_seed{seed}"):
            mlflow.log_params({"tier": 1, "split": "domain_shift", "target": target, "seed": seed})

            model, cal_residuals = train_lgbm(
                X_train, y_train, X_val, y_val, X_cal, y_cal, seed=seed, **model_params,
            )

            cal_preds = model.predict(X_cal)
            save_predictions(
                cal_merged["material_id"].values, y_cal, cal_preds,
                cal_merged["chemistry_family"].values,
                "domshift_cal", f"tier1_domshift_seed{seed}_{target_short}_cal.parquet",
            )

            for split_key in ["test_id", "test_ood_sulfide", "test_ood_nitride", "test_ood_halide"]:
                test_df = splits[split_key]
                X_test, y_test, test_merged = _merge_features(test_df, features, feature_cols, target)
                if len(test_merged) == 0:
                    continue

                test_preds = model.predict(X_test)
                metrics = compute_metrics(y_test, test_preds)
                mlflow.log_metrics({f"{split_key}_{k}": v for k, v in metrics.items()})
                logger.info("Tier 1 domshift %s %s seed=%d: %s", target, split_key, seed, metrics)

                save_predictions(
                    test_merged["material_id"].values, y_test, test_preds,
                    test_merged["chemistry_family"].values,
                    f"domshift_{split_key}",
                    f"tier1_domshift_seed{seed}_{target_short}_{split_key}.parquet",
                )

            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, MODELS_DIR / f"tier1_domshift_seed{seed}_{target_short}.joblib")


def run_mixed_train(df, features, target, seeds, model_params):
    """Train on all families, evaluate per family (domain randomization comparison)."""
    feature_cols = [c for c in features.columns if c != "material_id"]

    for seed in seeds:
        splits = mixed_train_split(df, seed=seed)

        X_train, y_train, _ = _merge_features(splits["train"], features, feature_cols, target)
        X_val, y_val, _ = _merge_features(splits["val"], features, feature_cols, target)
        X_cal, y_cal, cal_merged = _merge_features(splits["cal"], features, feature_cols, target)

        target_short = "ef" if target == "formation_energy_per_atom" else "bg"

        with mlflow.start_run(run_name=f"tier1_mixed_{target}_seed{seed}"):
            mlflow.log_params({"tier": 1, "split": "mixed_train", "target": target, "seed": seed})

            model, cal_residuals = train_lgbm(
                X_train, y_train, X_val, y_val, X_cal, y_cal, seed=seed, **model_params,
            )

            cal_preds = model.predict(X_cal)
            save_predictions(
                cal_merged["material_id"].values, y_cal, cal_preds,
                cal_merged["chemistry_family"].values,
                "mixed_cal", f"tier1_mixed_seed{seed}_{target_short}_cal.parquet",
            )

            # Per-family test evaluation
            for key in [k for k in splits if k.startswith("test")]:
                test_df = splits[key]
                X_test, y_test, test_merged = _merge_features(test_df, features, feature_cols, target)
                if len(test_merged) == 0:
                    continue

                test_preds = model.predict(X_test)
                metrics = compute_metrics(y_test, test_preds)
                mlflow.log_metrics({f"{key}_{k}": v for k, v in metrics.items()})
                logger.info("Tier 1 mixed %s %s seed=%d: %s", target, key, seed, metrics)

                save_predictions(
                    test_merged["material_id"].values, y_test, test_preds,
                    test_merged["chemistry_family"].values,
                    f"mixed_{key}",
                    f"tier1_mixed_seed{seed}_{target_short}_{key}.parquet",
                )


def main() -> None:
    with open("configs/base.yaml") as f:
        config = yaml.safe_load(f)

    seeds = config["evaluation"]["seeds"]
    targets = config["evaluation"]["targets"]
    model_params = {
        "n_estimators": config["model"]["n_estimators"],
        "learning_rate": config["model"]["learning_rate"],
        "num_leaves": config["model"]["num_leaves"],
        "min_child_samples": config["model"]["min_child_samples"],
        "subsample": config["model"]["subsample"],
        "colsample_bytree": config["model"]["colsample_bytree"],
        "early_stopping_rounds": config["model"]["early_stopping_rounds"],
    }

    adapter = MPAdapter(cache_dir=Path("data/mp"))
    df = adapter.load()

    features = compute_magpie_features(
        df, cache_path=Path("data/mp/magpie_features.parquet")
    )

    mlflow.set_experiment("crystal-prop-bench-tier1")

    for target in targets:
        run_standard_split(df, features, target, seeds, model_params)
        run_domain_shift(df, features, target, seeds, model_params)
        run_mixed_train(df, features, target, seeds, model_params)


if __name__ == "__main__":
    main()
