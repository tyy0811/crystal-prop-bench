"""Modal entrypoint for Tier 3 ALIGNN training on A10G GPU.

Usage:
    # First time: upload local data to Modal volume
    modal run scripts/run_tier3_modal.py::upload_data

    # Train (default: 200 epochs)
    modal run scripts/run_tier3_modal.py
    modal run scripts/run_tier3_modal.py --epochs 100 --lr 0.0005

    # Download results back to local machine
    modal run scripts/run_tier3_modal.py::download_results
"""

from __future__ import annotations

from pathlib import Path

import modal

app = modal.App("crystal-prop-bench-alignn")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.2.0",
        "dgl>=2.1.0",
        "alignn>=2024.5.0",
        "pymatgen>=2024.2.0",
        "matminer>=0.9.0",
        "mp-api>=0.39.0",
        "lightgbm>=4.3.0",
        "scikit-learn>=1.4.0",
        "joblib>=1.3.0",
        "pandas>=2.2.0",
        "pandera>=0.18.0",
        "mlflow>=2.11.0",
        "pyyaml>=6.0",
    )
)

volume = modal.Volume.from_name("crystal-prop-bench", create_if_missing=True)
REMOTE_ROOT = "/vol"


@app.function(image=image, timeout=300)
def upload_data() -> str:
    """Upload local data + configs + source to the Modal volume."""
    # This runs remotely but we use modal.Volume put_file from local entrypoint
    return "Use the local entrypoint upload_data instead"


@app.local_entrypoint()
def upload() -> None:
    """Upload local data, configs, and source code to the Modal volume.

    Run: modal run scripts/run_tier3_modal.py::upload
    """
    import subprocess

    local_root = Path(".")

    # Upload data directory
    print("Uploading data/mp/ ...")
    subprocess.run(
        ["modal", "volume", "put", "crystal-prop-bench",
         "data/mp/crystals.parquet", "data/mp/crystals.parquet"],
        check=True,
    )
    subprocess.run(
        ["modal", "volume", "put", "crystal-prop-bench",
         "data/mp/structures.pkl", "data/mp/structures.pkl"],
        check=True,
    )

    # Upload configs
    print("Uploading configs/ ...")
    subprocess.run(
        ["modal", "volume", "put", "crystal-prop-bench",
         "configs/base.yaml", "configs/base.yaml"],
        check=True,
    )

    # Upload source
    print("Uploading src/ and scripts/ ...")
    subprocess.run(
        ["modal", "volume", "put", "crystal-prop-bench",
         "src/", "src/"],
        check=True,
    )
    subprocess.run(
        ["modal", "volume", "put", "crystal-prop-bench",
         "scripts/", "scripts/"],
        check=True,
    )
    subprocess.run(
        ["modal", "volume", "put", "crystal-prop-bench",
         "pyproject.toml", "pyproject.toml"],
        check=True,
    )

    print("Upload complete. Run training with: modal run scripts/run_tier3_modal.py::train")


@app.function(
    gpu="A10G",
    timeout=43200,  # 12 hours
    image=image,
    volumes={REMOTE_ROOT: volume},
)
def train_tier3(config_overrides: dict | None = None) -> str:
    """Run Tier 3 training on GPU.

    Volume is mounted at /vol. Training code uses relative paths,
    so we chdir to /vol where data/, configs/, src/, scripts/ live.
    """
    import os
    import subprocess
    import sys

    os.chdir(REMOTE_ROOT)

    # Install the package from the uploaded source
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", "."],
        check=True,
    )

    # Patch save_checkpoint to also commit the volume after each run
    import scripts.run_tier3 as tier3_mod
    _original_save_checkpoint = tier3_mod._save_checkpoint

    def _save_and_commit(completed: set) -> None:
        _original_save_checkpoint(completed)
        volume.commit()

    tier3_mod._save_checkpoint = _save_and_commit

    tier3_mod.main(config_overrides)

    volume.commit()
    return "Tier 3 training complete"


@app.local_entrypoint()
def train(
    epochs: int = 200,
    lr: float = 0.001,
    batch_size: int = 256,
) -> None:
    """Launch Tier 3 training on Modal A10G GPU.

    Run: modal run scripts/run_tier3_modal.py::train
    """
    overrides = {"epochs": epochs, "lr": lr, "batch_size": batch_size}
    print(f"Launching Tier 3 training (epochs={epochs}, lr={lr}, batch_size={batch_size})...")
    result = train_tier3.remote(overrides)
    print(result)


@app.local_entrypoint()
def download() -> None:
    """Download results from Modal volume back to local machine.

    Run: modal run scripts/run_tier3_modal.py::download
    """
    import subprocess

    print("Downloading results/ ...")
    subprocess.run(
        ["modal", "volume", "get", "crystal-prop-bench",
         "results/", "results/"],
        check=True,
    )
    print("Download complete. Results are in results/")
