"""Tier 3 training: ALIGNN crystal graph model.

Builds ALIGNN graphs from cached structures, trains on standard +
domain-shift + mixed-train splits, saves predictions to results/predictions/.
Mirrors the run_tier1.py / run_tier2.py pattern.

Supports resumption: completed runs are tracked in a checkpoint file.
On restart, already-completed runs are skipped.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import mlflow
import pandas as pd
import torch
import yaml

from crystal_prop_bench.data.graph_builder import build_alignn_graphs
from crystal_prop_bench.data.mp_adapter import MPAdapter
from crystal_prop_bench.data.splits import domain_shift_split, mixed_train_split, standard_split
from crystal_prop_bench.evaluation.metrics import compute_metrics
from crystal_prop_bench.models import MODELS_DIR, save_predictions
from crystal_prop_bench.models.alignn_model import build_alignn, predict_alignn, train_alignn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = Path("results/tier3_checkpoint.json")


def _load_checkpoint() -> set[str]:
    """Load set of completed run keys from checkpoint file."""
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH) as f:
            data = json.load(f)
        completed = set(data.get("completed", []))
        logger.info("Resuming: %d runs already completed", len(completed))
        return completed
    return set()


def _save_checkpoint(completed: set[str]) -> None:
    """Save completed run keys to checkpoint file."""
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump({"completed": sorted(completed)}, f, indent=2)


def _run_key(split_name: str, target: str, seed: int) -> str:
    """Unique key for a training run."""
    return f"{split_name}_{target}_{seed}"


def _filter_graphs(
    df: pd.DataFrame,
    graphs: dict[str, tuple],
) -> tuple[list[str], pd.DataFrame]:
    """Filter df to rows with available graphs. Return (graph_ids, filtered_df)."""
    available = set(graphs.keys())
    mask = df["material_id"].isin(available)
    filtered = df[mask].reset_index(drop=True)
    graph_ids = filtered["material_id"].tolist()
    return graph_ids, filtered


def run_split(
    split_name: str,
    split_dfs: dict[str, pd.DataFrame],
    graphs: dict[str, tuple],
    target: str,
    seed: int,
    alignn_config: dict,
    test_keys: list[str],
) -> None:
    """Train on one split configuration and predict on test sets."""
    target_short = "ef" if target == "formation_energy_per_atom" else "bg"

    train_ids, train_df = _filter_graphs(split_dfs["train"], graphs)
    val_ids, val_df = _filter_graphs(split_dfs["val"], graphs)
    cal_ids, cal_df = _filter_graphs(split_dfs["cal"], graphs)

    if len(train_ids) < 10 or len(val_ids) < 2 or len(cal_ids) < 2:
        logger.warning("Skipping %s %s seed=%d: insufficient data", split_name, target, seed)
        return

    with mlflow.start_run(run_name=f"tier3_{split_name}_{target}_seed{seed}"):
        mlflow.log_params({
            "tier": 3, "split": split_name, "target": target, "seed": seed,
            "device": DEVICE, "n_train": len(train_ids),
        })

        model = build_alignn(
            alignn_layers=alignn_config["alignn_layers"],
            gcn_layers=alignn_config["gcn_layers"],
            embedding_features=alignn_config["embedding_features"],
            hidden_features=alignn_config["hidden_features"],
            output_features=alignn_config["output_features"],
        )

        model, cal_residuals, cal_preds = train_alignn(
            model=model,
            graphs=graphs,
            train_ids=train_ids,
            train_targets=train_df[target].values,
            val_ids=val_ids,
            val_targets=val_df[target].values,
            cal_ids=cal_ids,
            cal_targets=cal_df[target].values,
            seed=seed,
            epochs=alignn_config["epochs"],
            lr=alignn_config["lr"],
            weight_decay=alignn_config["weight_decay"],
            patience=alignn_config["patience"],
            scheduler_patience=alignn_config["scheduler_patience"],
            batch_size=alignn_config["batch_size"],
            device=DEVICE,
            epoch_checkpoint_path=Path(
                f"results/epoch_ckpt/tier3_{split_name}_seed{seed}_{target_short}.pt"
            ),
        )

        # Save cal predictions (reuse from train_alignn, no redundant forward pass)
        save_predictions(
            cal_df["material_id"].values, cal_df[target].values, cal_preds,
            cal_df["chemistry_family"].values,
            f"{split_name}_cal",
            f"tier3_{split_name}_seed{seed}_{target_short}_cal.parquet",
        )

        # Predict and save for each test set
        for test_key in test_keys:
            test_df_raw = split_dfs.get(test_key)
            if test_df_raw is None or len(test_df_raw) == 0:
                continue
            test_ids, test_df = _filter_graphs(test_df_raw, graphs)
            if len(test_ids) == 0:
                continue

            test_preds = predict_alignn(model, graphs, test_ids, device=DEVICE)
            metrics = compute_metrics(test_df[target].values, test_preds)
            mlflow.log_metrics({f"{test_key}_{k}": v for k, v in metrics.items()})
            logger.info("tier3 %s %s %s seed=%d: %s", split_name, target, test_key, seed, metrics)

            save_predictions(
                test_df["material_id"].values, test_df[target].values, test_preds,
                test_df["chemistry_family"].values,
                f"{split_name}_{test_key}",
                f"tier3_{split_name}_seed{seed}_{target_short}_{test_key}.parquet",
            )

        # Save model checkpoint
        if any(k in test_keys for k in ("test", "test_id")):
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(
                model.state_dict(),
                MODELS_DIR / f"tier3_{split_name}_seed{seed}_{target_short}.pt",
            )


def main(config_overrides: dict | None = None) -> None:
    with open("configs/base.yaml") as f:
        config = yaml.safe_load(f)

    if config_overrides:
        for key, val in config_overrides.items():
            if key in config.get("alignn", {}):
                config["alignn"][key] = val

    seeds = config["evaluation"]["seeds"]
    targets = config["evaluation"]["targets"]
    alignn_config = config["alignn"]

    # Load data
    adapter = MPAdapter(cache_dir=Path("data/mp"))
    df_full = adapter.load()

    # Subsample oxides (same strategy as Tier 2, smaller N due to GPU cost)
    OXIDE_SUBSAMPLE = 15000
    oxides = df_full[df_full["chemistry_family"] == "oxide"]
    minorities = df_full[df_full["chemistry_family"] != "oxide"]
    if len(oxides) > OXIDE_SUBSAMPLE:
        oxides = oxides.sample(n=OXIDE_SUBSAMPLE, random_state=42)
        logger.info("Subsampled oxides to %d for Tier 3", len(oxides))
    df = pd.concat([oxides, minorities], ignore_index=True)
    logger.info("Tier 3 dataset: %d crystals", len(df))

    # Load structures and build graphs
    with open(adapter.cache_path() / "structures.pkl", "rb") as f:
        structures = pickle.load(f)

    graphs = build_alignn_graphs(
        df, structures,
        cutoff=alignn_config["cutoff"],
        max_neighbors=alignn_config["max_neighbors"],
        cache_path=Path("data/mp/alignn_graphs.pkl"),
    )
    logger.info("Available graphs: %d / %d", len(graphs), len(df))

    # Filter df to graph-available rows
    graph_ids = set(graphs.keys())
    df = df[df["material_id"].isin(graph_ids)].reset_index(drop=True)
    logger.info("Tier 3 dataset after graph filter: %d crystals", len(df))

    # Resume support
    completed = _load_checkpoint()
    total_runs = len(seeds) * len(targets) * 3  # 3 split strategies
    logger.info("Total runs: %d, already completed: %d", total_runs, len(completed))

    mlflow.set_experiment("crystal-prop-bench-tier3")

    for target in targets:
        for seed in seeds:
            # Standard split
            key = _run_key("standard", target, seed)
            if key in completed:
                logger.info("Skipping %s (already completed)", key)
            else:
                train, val, cal, test = standard_split(df, seed=seed)
                run_split(
                    "standard",
                    {"train": train, "val": val, "cal": cal, "test": test},
                    graphs, target, seed, alignn_config,
                    test_keys=["test"],
                )
                completed.add(key)
                _save_checkpoint(completed)
                logger.info("Checkpoint saved: %d/%d runs complete", len(completed), total_runs)

            # Domain-shift split
            key = _run_key("domshift", target, seed)
            if key in completed:
                logger.info("Skipping %s (already completed)", key)
            else:
                splits = domain_shift_split(df, seed=seed, stratify_col=target)
                run_split(
                    "domshift", splits, graphs, target, seed, alignn_config,
                    test_keys=[
                        "test_id", "test_ood_sulfide", "test_ood_nitride", "test_ood_halide",
                    ],
                )
                completed.add(key)
                _save_checkpoint(completed)
                logger.info("Checkpoint saved: %d/%d runs complete", len(completed), total_runs)

            # Mixed-train split
            key = _run_key("mixed", target, seed)
            if key in completed:
                logger.info("Skipping %s (already completed)", key)
            else:
                mixed = mixed_train_split(df, seed=seed)
                mixed_test_keys = [k for k in mixed if k.startswith("test")]
                run_split(
                    "mixed", mixed, graphs, target, seed, alignn_config,
                    test_keys=mixed_test_keys,
                )
                completed.add(key)
                _save_checkpoint(completed)
                logger.info("Checkpoint saved: %d/%d runs complete", len(completed), total_runs)

    logger.info("All %d runs complete!", total_runs)


if __name__ == "__main__":
    main()
