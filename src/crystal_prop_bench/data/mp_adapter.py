"""Materials Project adapter using mp-api client."""

from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path

import pandas as pd
from mp_api.client import MPRester

from crystal_prop_bench.data.adapter import DatasetAdapter

logger = logging.getLogger(__name__)


class MPAdapter(DatasetAdapter):
    """Adapter for Materials Project data via mp-api."""

    def __init__(
        self,
        api_key: str | None = None,
        cache_dir: Path = Path("data/mp"),
    ) -> None:
        self._api_key = api_key or os.environ.get("MP_API_KEY", "")
        self._cache_dir = cache_dir

    def cache_path(self) -> Path:
        return self._cache_dir

    def load_raw(self) -> pd.DataFrame:
        """Fetch all crystals from Materials Project with target properties."""
        logger.info("Fetching data from Materials Project API...")

        with MPRester(self._api_key) as mpr:
            docs = mpr.materials.summary.search(
                fields=[
                    "material_id",
                    "formula_pretty",
                    "formation_energy_per_atom",
                    "band_gap",
                    "nsites",
                    "symmetry",
                    "structure",
                ],
            )

        logger.info("Fetched %d documents from Materials Project", len(docs))

        # Store structures separately (too complex for parquet)
        structures = {}
        rows = []
        for doc in docs:
            mid = str(doc.material_id)
            structures[mid] = doc.structure
            rows.append({
                "material_id": mid,
                "formula_pretty": doc.formula_pretty,
                "formation_energy_per_atom": doc.formation_energy_per_atom,
                "band_gap": doc.band_gap,
                "nsites": doc.nsites,
                "spacegroup_number": doc.symmetry.number,
            })

        df = pd.DataFrame(rows)

        # Cache structures as pickle
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        structures_path = self._cache_dir / "structures.pkl"
        with open(structures_path, "wb") as f:
            pickle.dump(structures, f)
        logger.info("Cached %d structures to %s", len(structures), structures_path)

        return df
