"""DatasetAdapter ABC — template-method pattern.

Two abstract methods (load_raw, cache_path). Concrete load() handles
chemistry classification, schema validation, filtering, and caching.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from crystal_prop_bench.data.chemistry import classify_chemistry_family

logger = logging.getLogger(__name__)


class DatasetAdapter(ABC):
    """Base adapter for materials property datasets.

    Subclasses implement load_raw() and cache_path().
    The concrete load() method handles the shared pipeline:
    load_raw -> classify chemistry -> validate -> filter -> cache.
    """

    @abstractmethod
    def load_raw(self) -> pd.DataFrame:
        """Fetch data from source. Return raw tabular DataFrame.

        Expected columns: material_id, formula_pretty,
        formation_energy_per_atom, band_gap, nsites, spacegroup_number.
        Structure objects stored separately.
        """
        ...

    @abstractmethod
    def cache_path(self) -> Path:
        """Directory for reading/writing cached data."""
        ...

    def load(self, force_refresh: bool = False) -> pd.DataFrame:
        """Load dataset with chemistry classification and caching.

        Returns DataFrame with 'chemistry_family' column added.
        Rows with chemistry_family=None are filtered out.
        """
        cache_dir = self.cache_path()
        parquet_path = cache_dir / "crystals.parquet"

        if parquet_path.exists() and not force_refresh:
            logger.info("Loading cached data from %s", parquet_path)
            return pd.read_parquet(parquet_path)

        logger.info("Fetching raw data...")
        df = self.load_raw()

        # Classify chemistry family
        from pymatgen.core import Composition

        df["chemistry_family"] = df["formula_pretty"].apply(
            lambda f: classify_chemistry_family(Composition(f))
        )

        # Log drop statistics
        total = len(df)
        dropped = df["chemistry_family"].isna().sum()
        logger.info(
            "Chemistry classification: %d/%d dropped (%.1f%%) — below purity threshold",
            dropped,
            total,
            100.0 * dropped / total if total > 0 else 0.0,
        )
        for family in ["oxide", "sulfide", "nitride", "halide"]:
            count = (df["chemistry_family"] == family).sum()
            logger.info("  %s: %d crystals", family, count)

        # Filter out unclassified compounds
        df = df[df["chemistry_family"].notna()].reset_index(drop=True)

        # Validate schema (imported here to avoid circular deps)
        from crystal_prop_bench.data.schemas import validate_crystal_df

        df = validate_crystal_df(df)

        # Cache
        cache_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(parquet_path, index=False)
        logger.info("Cached %d crystals to %s", len(df), parquet_path)

        return df
