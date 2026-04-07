"""Plotting functions. Read CSVs, write PNGs. No model or evaluation logic."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_domain_shift_bars(
    domain_shift_csv: Path,
    output_path: Path,
) -> None:
    """Bar chart: MAE per family, ID vs each OOD family."""
    df = pd.read_csv(domain_shift_csv)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, target in zip(axes, ["ef", "bg"]):
        target_df = df[df["target"] == target]
        tiers = sorted(target_df["tier"].unique())
        families = sorted(target_df["ood_family"].unique())
        n_tiers = len(tiers)
        x = np.arange(len(families))
        width = 0.8 / max(n_tiers, 1)
        colors = {"tier1": "#4C72B0", "tier2": "#DD8452", "tier3": "#55A868"}

        for i, tier in enumerate(tiers):
            tier_df = target_df[target_df["tier"] == tier]
            agg = tier_df.groupby("ood_family").agg(
                ood_mae=("ood_mae", "mean"),
            ).reindex(families)
            offset = (i - n_tiers / 2 + 0.5) * width
            ax.bar(x + offset, agg["ood_mae"], width,
                   label=tier, color=colors.get(tier, "#999999"))

        ax.set_xticks(x)
        ax.set_xticklabels(families)
        ax.set_ylabel("MAE")
        ax.set_title(f"{'Formation Energy' if target == 'ef' else 'Band Gap'}")
        ax.legend()

    fig.suptitle("Domain-Shift Degradation")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_conformal_coverage(
    conformal_csv: Path,
    output_path: Path,
) -> None:
    """Coverage vs interval width at multiple alpha levels."""
    df = pd.read_csv(conformal_csv)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, target in zip(axes, ["ef", "bg"]):
        target_df = df[df["target"] == target]
        for tier in target_df["tier"].unique():
            tier_df = target_df[target_df["tier"] == tier]
            agg = tier_df.groupby("alpha").agg(
                coverage=("coverage", "mean"),
                width=("mean_width", "mean"),
            ).reset_index()
            ax.plot(agg["width"], agg["coverage"], "o-", label=tier)

        ax.axhline(y=0.9, color="gray", linestyle="--", alpha=0.5, label="90% target")
        ax.set_xlabel("Mean Interval Width")
        ax.set_ylabel("Empirical Coverage")
        ax.set_title(f"{'Formation Energy' if target == 'ef' else 'Band Gap'}")
        ax.legend()

    fig.suptitle("Conformal Coverage vs. Interval Width")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_calibration_sweep(
    sweep_csv: Path,
    output_path: Path,
) -> None:
    """Coverage vs calibration budget per OOD family — the standout figure."""
    df = pd.read_csv(sweep_csv)

    families = df["ood_family"].unique()
    targets = df["target"].unique()

    fig, axes = plt.subplots(
        len(targets), len(families),
        figsize=(5 * len(families), 4 * len(targets)),
        squeeze=False,
    )

    for i, target in enumerate(targets):
        for j, family in enumerate(families):
            ax = axes[i][j]
            subset = df[(df["target"] == target) & (df["ood_family"] == family)]

            for alpha in sorted(subset["alpha"].unique()):
                alpha_df = subset[subset["alpha"] == alpha]
                agg = alpha_df.groupby("cal_size")["coverage"].agg(["mean", "std"]).reset_index()
                ax.errorbar(
                    agg["cal_size"], agg["mean"], yerr=agg["std"],
                    fmt="o-", label=f"alpha={alpha:.2f}", capsize=3,
                )
                ax.axhline(y=1 - alpha, color="gray", linestyle="--", alpha=0.3)

            ax.set_xlabel("Calibration Set Size")
            ax.set_ylabel("Empirical Coverage")
            ax.set_title(f"{family.title()} — {'Ef' if target == 'ef' else 'Bg'}")
            ax.legend(fontsize=8)

    fig.suptitle("Coverage vs. Calibration Budget (OOD Families)", fontsize=14)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_shap_summary(
    tables_dir: Path,
    output_path: Path,
    top_n: int = 15,
) -> None:
    """Horizontal bar chart of top-N global SHAP importance per tier."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, tier in zip(axes, ["tier1", "tier2"]):
        csv_path = tables_dir / f"shap_importance_{tier}.csv"
        if not csv_path.exists():
            ax.set_title(f"{tier} — not available")
            continue

        df = pd.read_csv(csv_path)
        top = df.nlargest(top_n, "mean_abs_shap")

        ax.barh(range(len(top)), top["mean_abs_shap"].values, color="#4C72B0")
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(top["feature"].values, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title(f"{'Composition (Tier 1)' if tier == 'tier1' else 'Structure (Tier 2)'}")

    fig.suptitle("Global Feature Importance (SHAP)")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
