"""ALIGNN graph construction from pymatgen structures.

Converts pymatgen Structure -> JARVIS Atoms -> DGL (atom_graph, line_graph)
using JARVIS/ALIGNN graph utilities. Handles caching and failure logging.
"""

from __future__ import annotations

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
    cache_path : If provided, cache graphs as pickle.

    Returns
    -------
    Dict mapping material_id to (atom_graph, line_graph, lattice_mat).
    Structures that fail graph construction are dropped and logged.
    """
    if cache_path and cache_path.exists():
        logger.info("Loading cached ALIGNN graphs from %s", cache_path)
        with open(cache_path, "rb") as f:
            return pickle.load(f)  # type: ignore[no-any-return]

    from joblib import Parallel, delayed  # type: ignore[import-untyped]

    material_ids = df["material_id"].values
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

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as fw:
            pickle.dump(graphs, fw)
        logger.info("Cached ALIGNN graphs to %s", cache_path)

    return graphs
