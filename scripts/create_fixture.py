"""One-time script to create test fixtures from Materials Project.

Fetches 25 crystals per chemistry family, computes features, and saves
to tests/fixtures/. Requires MP_API_KEY environment variable.

Run once, commit the results. CI never needs API access.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import pandas as pd

from crystal_prop_bench.data.mp_adapter import MPAdapter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FIXTURES_DIR = Path("tests/fixtures")
SAMPLES_PER_FAMILY = 25
FAMILIES = ["oxide", "sulfide", "nitride", "halide"]


def main() -> None:
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    # Fetch full dataset
    adapter = MPAdapter(cache_dir=Path("data/mp"))
    df = adapter.load()

    # Load structures
    with open(adapter.cache_path() / "structures.pkl", "rb") as f:
        structures = pickle.load(f)

    # Sample 25 per family, deterministic
    sampled_rows = []
    for family in FAMILIES:
        family_df = df[df["chemistry_family"] == family]
        sample = family_df.sample(n=SAMPLES_PER_FAMILY, random_state=42)
        sampled_rows.append(sample)
        logger.info("Sampled %d %s crystals", len(sample), family)

    fixture_df = pd.concat(sampled_rows, ignore_index=True)
    logger.info("Total fixture size: %d", len(fixture_df))

    # Save tabular fixture
    fixture_df.to_parquet(FIXTURES_DIR / "fixture_crystals.parquet", index=False)

    # Save corresponding structures
    fixture_structures = {
        row["material_id"]: structures[row["material_id"]]
        for _, row in fixture_df.iterrows()
        if row["material_id"] in structures
    }
    with open(FIXTURES_DIR / "fixture_structures.pkl", "wb") as f:
        pickle.dump(fixture_structures, f)

    logger.info(
        "Saved %d fixture crystals and %d structures",
        len(fixture_df),
        len(fixture_structures),
    )

    # Pre-compute Magpie features
    from crystal_prop_bench.data.featurizers import compute_magpie_features

    magpie_df = compute_magpie_features(fixture_df)
    magpie_df.to_parquet(FIXTURES_DIR / "fixture_magpie_features.parquet", index=False)
    logger.info("Saved Magpie features: %d rows, %d cols", *magpie_df.shape)

    # Pre-compute Voronoi features (some will fail)
    from crystal_prop_bench.data.featurizers import compute_voronoi_features

    voronoi_df = compute_voronoi_features(fixture_df, fixture_structures)
    voronoi_df.to_parquet(
        FIXTURES_DIR / "fixture_voronoi_features.parquet", index=False
    )
    logger.info("Saved Voronoi features: %d rows, %d cols", *voronoi_df.shape)
    logger.info(
        "Voronoi survival rate: %d/%d (%.1f%%)",
        len(voronoi_df),
        len(fixture_df),
        100.0 * len(voronoi_df) / len(fixture_df),
    )

    # Save the expected survival count for test pinning
    survival_meta = {"expected_voronoi_count": len(voronoi_df)}
    with open(FIXTURES_DIR / "fixture_meta.json", "w") as f:
        json.dump(survival_meta, f)


if __name__ == "__main__":
    main()
