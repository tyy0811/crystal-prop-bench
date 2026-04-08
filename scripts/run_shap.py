"""SHAP analysis on Tier 1 and Tier 2 LightGBM models.

Reads models from results/models/ and predictions from results/predictions/.
Writes SHAP data to results/tables/ for plotting.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import pandas as pd
import yaml

from crystal_prop_bench.data.featurizers import compute_magpie_features
from crystal_prop_bench.data.mp_adapter import MPAdapter

VORONOI_CACHE = Path("data/mp/voronoi_features_sub.parquet")
from crystal_prop_bench.evaluation.explainability import (  # noqa: E402
    compute_shap_values,
    extract_failure_cases,
    global_feature_importance,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODELS_DIR = Path("results/models")
PREDICTIONS_DIR = Path("results/predictions")
TABLES_DIR = Path("results/tables")


def main() -> None:
    with open("configs/base.yaml") as f:
        config = yaml.safe_load(f)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    # Load data for feature names and metadata
    adapter = MPAdapter(cache_dir=Path("data/mp"))
    df = adapter.load()
    features = compute_magpie_features(
        df, cache_path=Path("data/mp/magpie_features.parquet")
    )
    _feature_cols = [c for c in features.columns if c != "material_id"]

    # Use seed 42, standard split, formation energy as representative
    seed = config["evaluation"]["seeds"][0]

    # Load Voronoi features for Tier 2 if available
    voronoi_features = None
    if VORONOI_CACHE.exists():
        voronoi_features = pd.read_parquet(VORONOI_CACHE)
        if "crystal_system" in voronoi_features.columns:
            voronoi_features["crystal_system"] = voronoi_features[
                "crystal_system"
            ].astype("category").cat.codes
        if "is_centrosymmetric" in voronoi_features.columns:
            voronoi_features["is_centrosymmetric"] = voronoi_features[
                "is_centrosymmetric"
            ].astype(int)

    for tier in ["tier1", "tier2"]:
        model_path = MODELS_DIR / f"{tier}_standard_seed{seed}_ef.joblib"
        if not model_path.exists():
            logger.warning("Model not found: %s", model_path)
            continue

        model = joblib.load(model_path)

        # Select features matching the tier
        if tier == "tier2" and voronoi_features is not None:
            tier_features = voronoi_features
        else:
            tier_features = features
        tier_feature_cols = [c for c in tier_features.columns if c != "material_id"]

        # Load test predictions
        pred_files = list(PREDICTIONS_DIR.glob(f"{tier}_standard_seed{seed}_ef_test.parquet"))
        if not pred_files:
            continue
        pred_df = pd.read_parquet(pred_files[0])

        # Get features for test set
        test_features = tier_features[tier_features["material_id"].isin(pred_df["material_id"])]
        test_features = test_features.merge(
            pred_df[["material_id"]], on="material_id"
        )

        X_test = test_features[tier_feature_cols].values

        # Compute SHAP (subsample for speed if large)
        n_shap = min(len(X_test), 5000)
        shap_values = compute_shap_values(model, X_test[:n_shap])

        # Global importance
        importance = global_feature_importance(shap_values, tier_feature_cols)
        importance_df = pd.DataFrame(importance, columns=["feature", "mean_abs_shap"])
        importance_df["tier"] = tier
        importance_df.to_csv(
            TABLES_DIR / f"shap_importance_{tier}.csv", index=False
        )
        logger.info("Top 10 %s features: %s", tier, importance[:10])

        # Per-family SHAP comparison
        test_with_family = test_features.merge(
            pred_df[["material_id", "chemistry_family"]], on="material_id"
        )
        for family in test_with_family["chemistry_family"].unique():
            mask = test_with_family["chemistry_family"].values[:n_shap] == family
            if mask.sum() > 0:
                family_importance = global_feature_importance(
                    shap_values[mask], tier_feature_cols
                )
                family_df = pd.DataFrame(
                    family_importance, columns=["feature", "mean_abs_shap"]
                )
                family_df["tier"] = tier
                family_df["family"] = family
                family_df.to_csv(
                    TABLES_DIR / f"shap_importance_{tier}_{family}.csv", index=False
                )

        # Failure cases
        failures = extract_failure_cases(
            pred_df["y_true"].values,
            pred_df["y_pred"].values,
            pred_df["material_id"].values,
            n=50,
            metadata=df[["material_id", "chemistry_family", "spacegroup_number", "nsites"]],
        )
        failures["tier"] = tier
        failures.to_csv(TABLES_DIR / f"failure_cases_{tier}.csv", index=False)
        logger.info(
            "%s failure cases — family distribution: %s",
            tier,
            failures["chemistry_family"].value_counts().to_dict(),
        )

    # Cross-tier failure overlap
    tier1_failures = TABLES_DIR / "failure_cases_tier1.csv"
    tier2_failures = TABLES_DIR / "failure_cases_tier2.csv"
    if tier1_failures.exists() and tier2_failures.exists():
        t1 = pd.read_csv(tier1_failures)
        t2 = pd.read_csv(tier2_failures)
        overlap = set(t1["material_id"]) & set(t2["material_id"])
        logger.info(
            "Cross-tier failure overlap: %d/%d (%.1f%%)",
            len(overlap),
            min(len(t1), len(t2)),
            100.0 * len(overlap) / min(len(t1), len(t2)) if min(len(t1), len(t2)) > 0 else 0,
        )


if __name__ == "__main__":
    main()
