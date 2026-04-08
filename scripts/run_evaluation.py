"""Cross-cutting evaluation: conformal, domain shift, calibration sweep, bias check.

Reads prediction parquets from results/predictions/.
Writes result CSVs to results/tables/.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from crystal_prop_bench.data.splits import ood_calibration_sweep
from crystal_prop_bench.evaluation.conformal import (
    conformal_regression_interval,
    evaluate_conformal_coverage,
)
from crystal_prop_bench.evaluation.domain_shift import compute_degradation_ratios
from crystal_prop_bench.evaluation.metrics import (
    aggregate_seeds,
    compute_metrics,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PREDICTIONS_DIR = Path("results/predictions")
TABLES_DIR = Path("results/tables")


def build_benchmark_table(config: dict) -> pd.DataFrame:
    """Build the main benchmark table: tier x split x target x MAE +/- std."""
    rows = []
    targets = config["evaluation"]["targets"]

    for tier in ["tier1", "tier2", "tier1sub", "tier3"]:
        for split in ["standard", "domshift"]:
            for target in targets:
                target_short = "ef" if target == "formation_energy_per_atom" else "bg"
                test_key = "test" if split == "standard" else "test_id"

                seed_metrics = []
                for seed in config["evaluation"]["seeds"]:
                    pattern = f"{tier}_{split}_seed{seed}_{target_short}_{test_key}.parquet"
                    files = list(PREDICTIONS_DIR.glob(pattern))
                    if not files:
                        continue
                    pred_df = pd.read_parquet(files[0])
                    metrics = compute_metrics(
                        pred_df["y_true"].values, pred_df["y_pred"].values
                    )
                    seed_metrics.append(metrics)

                if seed_metrics:
                    agg = aggregate_seeds(seed_metrics)
                    rows.append({
                        "tier": tier,
                        "split": split,
                        "target": target_short,
                        **agg,
                    })

    return pd.DataFrame(rows)


def build_domain_shift_table(config: dict) -> pd.DataFrame:
    """Build domain-shift degradation table."""
    rows = []
    targets = config["evaluation"]["targets"]

    for tier in ["tier1", "tier2", "tier3"]:
        for target in targets:
            target_short = "ef" if target == "formation_energy_per_atom" else "bg"

            for seed in config["evaluation"]["seeds"]:
                # ID metrics
                id_files = list(PREDICTIONS_DIR.glob(
                    f"{tier}_domshift_seed{seed}_{target_short}_test_id.parquet"
                ))
                if not id_files:
                    continue
                id_df = pd.read_parquet(id_files[0])
                id_metrics = compute_metrics(id_df["y_true"].values, id_df["y_pred"].values)

                # OOD metrics per family
                ood_metrics = {}
                for family in ["sulfide", "nitride", "halide"]:
                    ood_files = list(PREDICTIONS_DIR.glob(
                        f"{tier}_domshift_seed{seed}_{target_short}_test_ood_{family}.parquet"
                    ))
                    if ood_files:
                        ood_df = pd.read_parquet(ood_files[0])
                        ood_metrics[family] = compute_metrics(
                            ood_df["y_true"].values, ood_df["y_pred"].values
                        )

                ratios = compute_degradation_ratios(id_metrics, ood_metrics)
                for family, ratio_dict in ratios.items():
                    rows.append({
                        "tier": tier,
                        "target": target_short,
                        "seed": seed,
                        "ood_family": family,
                        "id_mae": id_metrics["mae"],
                        "ood_mae": ood_metrics[family]["mae"],
                        **ratio_dict,
                    })

    return pd.DataFrame(rows)


def build_conformal_table(config: dict) -> pd.DataFrame:
    """Build conformal coverage table for all tiers and splits."""
    rows = []
    alphas = config["evaluation"]["alphas"]
    targets = config["evaluation"]["targets"]

    for tier in ["tier1", "tier2", "tier3"]:
        for split in ["standard", "domshift"]:
            for target in targets:
                target_short = "ef" if target == "formation_energy_per_atom" else "bg"

                for seed in config["evaluation"]["seeds"]:
                    # Load cal residuals
                    cal_files = list(PREDICTIONS_DIR.glob(
                        f"{tier}_{split}_seed{seed}_{target_short}_cal.parquet"
                    ))
                    if not cal_files:
                        continue
                    cal_df = pd.read_parquet(cal_files[0])
                    cal_residuals = np.abs(
                        cal_df["y_true"].values - cal_df["y_pred"].values
                    )

                    # Evaluate on test sets
                    if split == "standard":
                        test_keys = ["test"]
                    else:
                        test_keys = ["test_id", "test_ood_sulfide", "test_ood_nitride", "test_ood_halide"]

                    for test_key in test_keys:
                        test_files = list(PREDICTIONS_DIR.glob(
                            f"{tier}_{split}_seed{seed}_{target_short}_{test_key}.parquet"
                        ))
                        if not test_files:
                            continue
                        test_df = pd.read_parquet(test_files[0])

                        coverage_results = evaluate_conformal_coverage(
                            cal_residuals,
                            test_df["y_true"].values,
                            test_df["y_pred"].values,
                            alphas=alphas,
                        )

                        for cr in coverage_results:
                            rows.append({
                                "tier": tier,
                                "split": split,
                                "target": target_short,
                                "seed": seed,
                                "test_set": test_key,
                                **cr,
                            })

    return pd.DataFrame(rows)


def build_calibration_sweep_table(config: dict) -> pd.DataFrame:
    """Build calibration sweep table: coverage vs. cal budget per OOD family."""
    rows = []
    cal_sizes = config["evaluation"]["cal_sizes"]
    alphas = config["evaluation"]["alphas"]
    targets = config["evaluation"]["targets"]

    for tier in ["tier1", "tier2", "tier3"]:
        for target in targets:
            target_short = "ef" if target == "formation_energy_per_atom" else "bg"

            for seed in config["evaluation"]["seeds"]:
                # Load the model's OOD predictions for each family
                for family in ["sulfide", "nitride", "halide"]:
                    ood_files = list(PREDICTIONS_DIR.glob(
                        f"{tier}_domshift_seed{seed}_{target_short}_test_ood_{family}.parquet"
                    ))
                    if not ood_files:
                        continue
                    ood_df = pd.read_parquet(ood_files[0])

                    # Sweep calibration sizes
                    try:
                        pairs = ood_calibration_sweep(ood_df, cal_sizes=cal_sizes, seed=seed)
                    except ValueError:
                        continue

                    for (cal_subset, test_remainder), cal_size in zip(pairs, cal_sizes):
                        cal_resid = np.abs(
                            cal_subset["y_true"].values - cal_subset["y_pred"].values
                        )

                        for alpha in alphas:
                            lower, upper = conformal_regression_interval(
                                cal_resid,
                                test_remainder["y_pred"].values,
                                alpha,
                            )
                            covered = (
                                (test_remainder["y_true"].values >= lower)
                                & (test_remainder["y_true"].values <= upper)
                            ).mean()
                            width = (upper - lower).mean()

                            rows.append({
                                "tier": tier,
                                "target": target_short,
                                "seed": seed,
                                "ood_family": family,
                                "cal_size": cal_size,
                                "alpha": alpha,
                                "coverage": float(covered),
                                "mean_width": float(width),
                            })

    return pd.DataFrame(rows)


def build_bias_check_table(config: dict) -> pd.DataFrame:
    """Compare Tier 1 on full set vs. Voronoi subset."""
    rows = []
    targets = config["evaluation"]["targets"]

    for target in targets:
        target_short = "ef" if target == "formation_energy_per_atom" else "bg"

        for seed in config["evaluation"]["seeds"]:
            for tier_label, label in [("tier1", "tier1_full"), ("tier1sub", "tier1_voronoi_subset")]:
                files = list(PREDICTIONS_DIR.glob(
                    f"{tier_label}_standard_seed{seed}_{target_short}_test.parquet"
                ))
                if not files:
                    continue
                df = pd.read_parquet(files[0])
                metrics = compute_metrics(df["y_true"].values, df["y_pred"].values)
                rows.append({
                    "variant": label,
                    "target": target_short,
                    "seed": seed,
                    **metrics,
                })

    return pd.DataFrame(rows)


def main() -> None:
    with open("configs/base.yaml") as f:
        config = yaml.safe_load(f)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    benchmark = build_benchmark_table(config)
    benchmark.to_csv(TABLES_DIR / "benchmark.csv", index=False)
    logger.info("Wrote benchmark.csv (%d rows)", len(benchmark))

    domain_shift = build_domain_shift_table(config)
    domain_shift.to_csv(TABLES_DIR / "domain_shift.csv", index=False)
    logger.info("Wrote domain_shift.csv (%d rows)", len(domain_shift))

    conformal = build_conformal_table(config)
    conformal.to_csv(TABLES_DIR / "conformal_coverage.csv", index=False)
    logger.info("Wrote conformal_coverage.csv (%d rows)", len(conformal))

    cal_sweep = build_calibration_sweep_table(config)
    cal_sweep.to_csv(TABLES_DIR / "calibration_sweep.csv", index=False)
    logger.info("Wrote calibration_sweep.csv (%d rows)", len(cal_sweep))

    bias_check = build_bias_check_table(config)
    bias_check.to_csv(TABLES_DIR / "bias_check.csv", index=False)
    logger.info("Wrote bias_check.csv (%d rows)", len(bias_check))


if __name__ == "__main__":
    main()
