"""End-to-end integration test on 100-crystal fixture.

Runs: load fixture -> featurize (from cache) -> split -> train -> predict
-> conformal -> metrics. All on CPU, no network, uses pre-computed features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from crystal_prop_bench.data.splits import standard_split
from crystal_prop_bench.evaluation.conformal import evaluate_conformal_coverage
from crystal_prop_bench.evaluation.metrics import aggregate_seeds, compute_metrics
from crystal_prop_bench.models.lgbm_baseline import train_lgbm


class TestIntegration:
    def test_full_pipeline_tier1(
        self,
        fixture_crystals: pd.DataFrame,
        fixture_magpie_features: pd.DataFrame,
    ) -> None:
        """Full Tier 1 pipeline on 100-crystal fixture."""
        df = fixture_crystals
        features = fixture_magpie_features
        target = "formation_energy_per_atom"

        # Split (4-way: train/val/cal/test)
        train_df, val_df, cal_df, test_df = standard_split(df, seed=42)

        # Align features
        feature_cols = [c for c in features.columns if c != "material_id"]

        train_m = train_df.merge(features, on="material_id")
        val_m = val_df.merge(features, on="material_id")
        cal_m = cal_df.merge(features, on="material_id")
        test_m = test_df.merge(features, on="material_id")

        assert len(train_m) > 0
        assert len(val_m) > 0
        assert len(cal_m) > 0
        assert len(test_m) > 0

        # Train (val for early stopping, cal for conformal only)
        model, cal_residuals = train_lgbm(
            train_m[feature_cols].values,
            train_m[target].values,
            val_m[feature_cols].values,
            val_m[target].values,
            cal_m[feature_cols].values,
            cal_m[target].values,
            seed=42,
        )

        # Predict
        test_preds = model.predict(test_m[feature_cols].values)
        assert test_preds.shape == (len(test_m),)

        # Metrics
        metrics = compute_metrics(test_m[target].values, test_preds)
        assert "mae" in metrics
        assert "rmse" in metrics
        assert "r2" in metrics
        assert metrics["mae"] >= 0
        assert metrics["rmse"] >= 0

        # Conformal
        coverage_results = evaluate_conformal_coverage(
            cal_residuals,
            test_m[target].values,
            test_preds,
            alphas=[0.10, 0.20],
        )
        assert len(coverage_results) == 2
        for cr in coverage_results:
            assert 0.0 <= cr["coverage"] <= 1.0
            assert cr["mean_width"] >= 0.0

    def test_seed_aggregation(
        self,
        fixture_crystals: pd.DataFrame,
        fixture_magpie_features: pd.DataFrame,
    ) -> None:
        """Verify seed aggregation works across multiple seeds."""
        df = fixture_crystals
        features = fixture_magpie_features
        target = "formation_energy_per_atom"
        feature_cols = [c for c in features.columns if c != "material_id"]

        seed_results = []
        for seed in [42, 123]:
            train_df, val_df, cal_df, test_df = standard_split(df, seed=seed)
            train_m = train_df.merge(features, on="material_id")
            val_m = val_df.merge(features, on="material_id")
            cal_m = cal_df.merge(features, on="material_id")
            test_m = test_df.merge(features, on="material_id")

            model, _ = train_lgbm(
                train_m[feature_cols].values, train_m[target].values,
                val_m[feature_cols].values, val_m[target].values,
                cal_m[feature_cols].values, cal_m[target].values,
                seed=seed,
            )
            preds = model.predict(test_m[feature_cols].values)
            seed_results.append(compute_metrics(test_m[target].values, preds))

        agg = aggregate_seeds(seed_results)
        assert "mae_mean" in agg
        assert "mae_std" in agg
        assert agg["mae_mean"] > 0
