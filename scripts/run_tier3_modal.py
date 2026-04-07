"""Modal entrypoint for Tier 3 ALIGNN training on A10G GPU.

Usage:
    modal run scripts/run_tier3_modal.py
    modal run scripts/run_tier3_modal.py --epochs 100 --lr 0.0005
"""

from __future__ import annotations

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

data_volume = modal.Volume.from_name("crystal-prop-bench-data", create_if_missing=True)
results_volume = modal.Volume.from_name("crystal-prop-bench-results", create_if_missing=True)


@app.function(
    gpu="A10G",
    timeout=7200,
    image=image,
    volumes={
        "/root/data": data_volume,
        "/root/results": results_volume,
    },
)
def train_tier3(config_overrides: dict | None = None) -> str:
    """Run Tier 3 training on GPU.

    Volumes are mounted at /root/data and /root/results, matching
    the relative paths used by the training code when cwd is /root.
    """
    import os
    import subprocess
    import sys

    os.chdir("/root")

    # Install the package
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."], check=True)

    from scripts.run_tier3 import main
    main(config_overrides)

    # Commit writes to persistent volumes
    data_volume.commit()
    results_volume.commit()

    return "Tier 3 training complete"


@app.local_entrypoint()
def run(
    epochs: int = 200,
    lr: float = 0.001,
    batch_size: int = 256,
) -> None:
    overrides = {"epochs": epochs, "lr": lr, "batch_size": batch_size}
    result = train_tier3.remote(overrides)
    print(result)
