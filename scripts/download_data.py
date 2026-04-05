"""Download and cache Materials Project data."""

import logging
from pathlib import Path

from crystal_prop_bench.data.mp_adapter import MPAdapter

logging.basicConfig(level=logging.INFO)


def main() -> None:
    adapter = MPAdapter(cache_dir=Path("data/mp"))
    df = adapter.load()
    print(f"Loaded {len(df)} crystals")
    print(f"Families: {df['chemistry_family'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
