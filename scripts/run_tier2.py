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
import pandas as pd
import yaml

from crystal_prop_bench.data.featurizers import (
    compute_magpie_features,
    compute_voronoi_features,
)
from crystal_prop_bench.data.mp_adapter import MPAdapter
from crystal_prop_bench.data.splits import domain_shift_split, standard_split
from crystal_prop_bench.evaluation.metrics import compute_metrics
from crystal_prop_bench.models import MODELS_DIR, save_predictions
from crystal_prop_bench.models.lgbm_baseline import train_lgbm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _merge_features(df, features, feature_cols, target):
    merged = df.merge(features, on="material_id")
    return merged[feature_cols].values, merged[target].values, merged


def train_and_predict(
    train_df, val_df, cal_df, test_df, features, target, seed,
    tier_label, split_label, test_split_key, model_params,
):
    """Generic train-predict-save loop."""
    feature_cols = [c for c in features.columns if c != "material_id"]

    X_train, y_train, _ = _merge_features(train_df, features, feature_cols, target)
    X_val, y_val, _ = _merge_features(val_df, features, feature_cols, target)
    X_cal, y_cal, cal_m = _merge_features(cal_df, features, feature_cols, target)
    X_test, y_test, test_m = _merge_features(test_df, features, feature_cols, target)

    if len(X_train) == 0 or len(X_test) == 0:
        return

    model, _ = train_lgbm(
        X_train, y_train, X_val, y_val, X_cal, y_cal, seed=seed, **model_params,
    )

    test_preds = model.predict(X_test)
    target_short = "ef" if target == "formation_energy_per_atom" else "bg"

    save_predictions(
        test_m["material_id"].values, test_m[target].values, test_preds,
        test_m["chemistry_family"].values,
        f"{split_label}_{test_split_key}",
        f"{tier_label}_{split_label}_seed{seed}_{target_short}_{test_split_key}.parquet",
    )

    cal_preds = model.predict(X_cal)
    save_predictions(
        cal_m["material_id"].values, cal_m[target].values, cal_preds,
        cal_m["chemistry_family"].values,
        f"{split_label}_cal",
        f"{tier_label}_{split_label}_seed{seed}_{target_short}_cal.parquet",
    )

    metrics = compute_metrics(test_m[target].values, test_preds)
    mlflow.log_metrics({f"{test_split_key}_{k}": v for k, v in metrics.items()})
    logger.info(
        "%s %s %s %s seed=%d: %s",
        tier_label, split_label, target, test_split_key, seed, metrics,
    )

    if test_split_key in ("test", "test_id"):
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            model,
            MODELS_DIR / f"{tier_label}_{split_label}_seed{seed}_{target_short}.joblib",
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
    df_full = adapter.load()

    # Subsample oxides to 15K; keep all minority families intact.
    # LightGBM convergence is unchanged at this scale, and it reduces
    # Voronoi featurization from ~80 hours to ~2 hours.
    OXIDE_SUBSAMPLE = 25000
    oxides = df_full[df_full["chemistry_family"] == "oxide"]
    minorities = df_full[df_full["chemistry_family"] != "oxide"]
    if len(oxides) > OXIDE_SUBSAMPLE:
        oxides = oxides.sample(n=OXIDE_SUBSAMPLE, random_state=42)
        logger.info(
            "Subsampled oxides to %d (from %d) for Tier 2 featurization",
            len(oxides), (df_full["chemistry_family"] == "oxide").sum(),
        )
    df = pd.concat([oxides, minorities], ignore_index=True)
    logger.info("Tier 2 dataset: %d crystals", len(df))

    with open(adapter.cache_path() / "structures.pkl", "rb") as f:
        structures = pickle.load(f)

    voronoi_features = compute_voronoi_features(
        df, structures, cache_path=Path("data/mp/voronoi_features_sub.parquet")
    )
    magpie_features = compute_magpie_features(
        df, cache_path=Path("data/mp/magpie_features.parquet")
    )

    # Encode non-numeric columns from GlobalSymmetryFeatures
    if "crystal_system" in voronoi_features.columns:
        voronoi_features["crystal_system"] = voronoi_features["crystal_system"].astype(
            "category"
        ).cat.codes
    if "is_centrosymmetric" in voronoi_features.columns:
        voronoi_features["is_centrosymmetric"] = voronoi_features[
            "is_centrosymmetric"
        ].astype(int)

    voronoi_ids = set(voronoi_features["material_id"])
    df_voronoi = df[df["material_id"].isin(voronoi_ids)].copy()
    magpie_voronoi = magpie_features[magpie_features["material_id"].isin(voronoi_ids)]

    logger.info(
        "Voronoi subset: %d/%d crystals (%.1f%%)",
        len(df_voronoi), len(df), 100.0 * len(df_voronoi) / len(df),
    )

    mlflow.set_experiment("crystal-prop-bench-tier2")

    for target in targets:
        for seed in seeds:
            train, val, cal, test = standard_split(df_voronoi, seed=seed)

            with mlflow.start_run(run_name=f"tier2_standard_{target}_seed{seed}"):
                mlflow.log_params({"tier": 2, "split": "standard", "target": target, "seed": seed})
                train_and_predict(
                    train, val, cal, test, voronoi_features, target, seed,
                    "tier2", "standard", "test", model_params,
                )

            with mlflow.start_run(run_name=f"tier1sub_standard_{target}_seed{seed}"):
                mlflow.log_params({
                    "tier": "1_voronoi_subset", "split": "standard",
                    "target": target, "seed": seed,
                })
                train_and_predict(
                    train, val, cal, test, magpie_voronoi, target, seed,
                    "tier1sub", "standard", "test", model_params,
                )

            splits = domain_shift_split(df_voronoi, seed=seed, stratify_col=target)

            with mlflow.start_run(run_name=f"tier2_domshift_{target}_seed{seed}"):
                mlflow.log_params({
                    "tier": 2, "split": "domain_shift",
                    "target": target, "seed": seed,
                })
                for split_key in [
                    "test_id", "test_ood_sulfide",
                    "test_ood_nitride", "test_ood_halide",
                ]:
                    train_and_predict(
                        splits["train"], splits["val"], splits["cal"],
                        splits[split_key], voronoi_features, target, seed,
                        "tier2", "domshift", split_key, model_params,
                    )


if __name__ == "__main__":
    main()
