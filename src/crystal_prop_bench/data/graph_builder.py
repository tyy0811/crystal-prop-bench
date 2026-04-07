"""ALIGNN graph construction from pymatgen structures.

Converts pymatgen Structure -> JARVIS Atoms -> DGL (atom_graph, line_graph)
using JARVIS/ALIGNN graph utilities. Handles caching and failure logging.
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Structure

logger = logging.getLogger(__name__)


def pymatgen_to_jarvis(structure: Structure):  # type: ignore[no-untyped-def]
    """Convert pymatgen Structure to JARVIS Atoms."""
    from jarvis.core.atoms import Atoms as JarvisAtoms

    return JarvisAtoms(
        lattice_mat=structure.lattice.matrix.tolist(),
        coords=structure.frac_coords.tolist(),
        elements=[str(site.specie) for site in structure],
        cartesian=False,
    )


def build_alignn_graph(
    atoms: object,
    cutoff: float = 8.0,
    max_neighbors: int = 12,
) -> tuple:
    """Build (atom_graph, line_graph, lattice_mat) for one JARVIS Atoms object.

    Returns DGL graph pair + lattice matrix (3x3 numpy array)
    with node/edge features set by ALIGNN convention.
    """
    from alignn.graphs import Graph

    g, lg = Graph.atom_dgl_multigraph(
        atoms=atoms,
        cutoff=cutoff,
        max_neighbors=max_neighbors,
        atom_features="cgcnn",
        compute_line_graph=True,
        use_canonize=False,
    )
    lattice_mat = np.array(atoms.lattice_mat, dtype=np.float32)  # type: ignore[attr-defined]
    return g, lg, lattice_mat


def _cache_key(
    material_ids: list[str],
    cutoff: float,
    max_neighbors: int,
) -> str:
    """Deterministic hash of graph build parameters and requested IDs."""
    manifest = json.dumps({
        "ids": sorted(material_ids),
        "cutoff": cutoff,
        "max_neighbors": max_neighbors,
    }, sort_keys=True)
    return hashlib.sha256(manifest.encode()).hexdigest()[:16]


def _load_validated_cache(
    cache_path: Path,
    requested_ids: set[str],
    cutoff: float,
    max_neighbors: int,
) -> dict[str, tuple] | None:
    """Load cache only if it matches current parameters and covers requested IDs."""
    meta_path = cache_path.with_suffix(".meta.json")
    if not cache_path.exists() or not meta_path.exists():
        return None

    with open(meta_path) as f:
        meta = json.load(f)

    if meta.get("cutoff") != cutoff or meta.get("max_neighbors") != max_neighbors:
        logger.info(
            "Cache parameters mismatch (cutoff=%s/%s, max_neighbors=%s/%s), rebuilding",
            meta.get("cutoff"), cutoff, meta.get("max_neighbors"), max_neighbors,
        )
        return None

    with open(cache_path, "rb") as f:
        graphs = pickle.load(f)

    cached_ids = set(graphs.keys())
    missing = requested_ids - cached_ids
    if missing:
        logger.warning(
            "Cache missing %d/%d requested IDs, rebuilding",
            len(missing), len(requested_ids),
        )
        return None

    logger.info(
        "Loaded validated ALIGNN graph cache: %d graphs from %s",
        len(graphs), cache_path,
    )
    return graphs  # type: ignore[no-any-return]


def _save_cache(
    cache_path: Path,
    graphs: dict[str, tuple],
    cutoff: float,
    max_neighbors: int,
) -> None:
    """Save graphs and metadata for cache validation."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as fw:
        pickle.dump(graphs, fw)

    meta = {
        "cutoff": cutoff,
        "max_neighbors": max_neighbors,
        "n_graphs": len(graphs),
        "ids": sorted(graphs.keys()),
    }
    meta_path = cache_path.with_suffix(".meta.json")
    with open(meta_path, "w") as fm:
        json.dump(meta, fm)

    logger.info("Cached %d ALIGNN graphs to %s", len(graphs), cache_path)


def build_alignn_graphs(
    df: pd.DataFrame,
    structures: dict[str, Structure],
    cutoff: float = 8.0,
    max_neighbors: int = 12,
    cache_path: Path | None = None,
) -> dict[str, tuple]:
    """Build ALIGNN graph pairs for all structures with caching.

    Parameters
    ----------
    df : DataFrame with material_id column.
    structures : material_id -> pymatgen Structure mapping.
    cutoff : Bond distance cutoff in Angstroms.
    max_neighbors : Max neighbors per atom.
    cache_path : If provided, cache graphs as pickle with metadata sidecar.

    Returns
    -------
    Dict mapping material_id to (atom_graph, line_graph, lattice_mat).
    Structures that fail graph construction are dropped and logged.
    """
    material_ids = df["material_id"].values
    requested_ids = {mid for mid in material_ids if mid in structures}

    if cache_path:
        cached = _load_validated_cache(cache_path, requested_ids, cutoff, max_neighbors)
        if cached is not None:
            return cached

    from joblib import Parallel, delayed  # type: ignore[import-untyped]

    families = df["chemistry_family"].values if "chemistry_family" in df.columns else None

    work_items = []
    for i, mid in enumerate(material_ids):
        if mid in structures:
            family = families[i] if families is not None else "unknown"
            work_items.append((mid, structures[mid], family))

    def _build_one(mid: str, structure: Structure, family: str) -> tuple[str, tuple | None, str]:
        try:
            atoms = pymatgen_to_jarvis(structure)
            g, lg, lat = build_alignn_graph(atoms, cutoff=cutoff, max_neighbors=max_neighbors)
            return mid, (g, lg, lat), family
        except Exception as e:
            logger.debug("Graph construction failed for %s: %s", mid, e)
            return mid, None, family

    if not work_items:
        logger.info("No structures to build graphs for")
        return {}

    n_jobs = min(8, len(work_items))
    logger.info("Building ALIGNN graphs for %d structures (%d jobs)...", len(work_items), n_jobs)

    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_build_one)(mid, struct, family)
        for mid, struct, family in work_items
    )

    graphs: dict[str, tuple] = {}
    failed_by_family: dict[str, int] = {}
    for mid, graph_pair, family in results:
        if graph_pair is not None:
            graphs[mid] = graph_pair
        else:
            failed_by_family[family] = failed_by_family.get(family, 0) + 1

    n_missing = len(material_ids) - len(work_items)
    n_failed = sum(failed_by_family.values())
    logger.info(
        "ALIGNN graph construction: %d succeeded, %d failed, %d missing structure",
        len(graphs), n_failed, n_missing,
    )
    for family, count in sorted(failed_by_family.items()):
        logger.info("  Failed in %s: %d", family, count)

    if n_failed > 0:
        logger.warning(
            "ALIGNN graph construction dropped %d structures (%.1f%% failure rate)",
            n_failed, 100.0 * n_failed / len(work_items),
        )

    if cache_path:
        _save_cache(cache_path, graphs, cutoff, max_neighbors)

    return graphs
