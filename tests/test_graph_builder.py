"""Tests for ALIGNN graph construction from pymatgen structures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
dgl = pytest.importorskip("dgl")
pytest.importorskip("alignn")

from crystal_prop_bench.data.graph_builder import (
    build_alignn_graph,
    build_alignn_graphs,
    pymatgen_to_jarvis,
)


class TestPymatgenToJarvis:
    def test_converts_structure(self, fixture_structures: dict) -> None:
        mid = next(iter(fixture_structures))
        structure = fixture_structures[mid]
        atoms = pymatgen_to_jarvis(structure)
        assert len(atoms.elements) == len(structure)
        assert atoms.lattice_mat is not None

    def test_preserves_element_count(self, fixture_structures: dict) -> None:
        mid = next(iter(fixture_structures))
        structure = fixture_structures[mid]
        atoms = pymatgen_to_jarvis(structure)
        assert len(atoms.elements) == structure.num_sites


class TestBuildAlignnGraph:
    def test_returns_graph_pair(self, fixture_structures: dict) -> None:
        mid = next(iter(fixture_structures))
        structure = fixture_structures[mid]
        atoms = pymatgen_to_jarvis(structure)
        g, lg = build_alignn_graph(atoms, cutoff=8.0)
        assert isinstance(g, dgl.DGLGraph)
        assert isinstance(lg, dgl.DGLGraph)

    def test_atom_graph_nodes_match_sites(self, fixture_structures: dict) -> None:
        mid = next(iter(fixture_structures))
        structure = fixture_structures[mid]
        atoms = pymatgen_to_jarvis(structure)
        g, _ = build_alignn_graph(atoms, cutoff=8.0)
        assert g.num_nodes() == len(structure)

    def test_atom_graph_has_node_features(self, fixture_structures: dict) -> None:
        mid = next(iter(fixture_structures))
        structure = fixture_structures[mid]
        atoms = pymatgen_to_jarvis(structure)
        g, _ = build_alignn_graph(atoms, cutoff=8.0)
        assert "atom_features" in g.ndata
        assert g.ndata["atom_features"].shape[0] == g.num_nodes()

    def test_atom_graph_has_edges(self, fixture_structures: dict) -> None:
        mid = next(iter(fixture_structures))
        structure = fixture_structures[mid]
        atoms = pymatgen_to_jarvis(structure)
        g, _ = build_alignn_graph(atoms, cutoff=8.0)
        assert g.num_edges() > 0

    def test_line_graph_nodes_equal_atom_edges(self, fixture_structures: dict) -> None:
        mid = next(iter(fixture_structures))
        structure = fixture_structures[mid]
        atoms = pymatgen_to_jarvis(structure)
        g, lg = build_alignn_graph(atoms, cutoff=8.0)
        assert lg.num_nodes() == g.num_edges()


class TestBuildAlignnGraphs:
    def test_builds_for_all_structures(
        self, fixture_crystals: pd.DataFrame, fixture_structures: dict,
    ) -> None:
        graphs = build_alignn_graphs(fixture_crystals, fixture_structures, cutoff=8.0)
        # Some may fail, but most should succeed
        assert len(graphs) > 0
        assert len(graphs) <= len(fixture_crystals)

    def test_returns_graph_pairs(
        self, fixture_crystals: pd.DataFrame, fixture_structures: dict,
    ) -> None:
        graphs = build_alignn_graphs(fixture_crystals, fixture_structures, cutoff=8.0)
        mid = next(iter(graphs))
        g, lg = graphs[mid]
        assert isinstance(g, dgl.DGLGraph)
        assert isinstance(lg, dgl.DGLGraph)

    def test_caching_roundtrip(
        self, fixture_crystals: pd.DataFrame, fixture_structures: dict, tmp_path: Path,
    ) -> None:
        cache = tmp_path / "graphs.pkl"
        g1 = build_alignn_graphs(
            fixture_crystals, fixture_structures, cutoff=8.0, cache_path=cache,
        )
        assert cache.exists()
        g2 = build_alignn_graphs(
            fixture_crystals, fixture_structures, cutoff=8.0, cache_path=cache,
        )
        assert set(g1.keys()) == set(g2.keys())

    def test_missing_structure_skipped(
        self, fixture_crystals: pd.DataFrame,
    ) -> None:
        # Empty structures dict — all should be skipped
        graphs = build_alignn_graphs(fixture_crystals, {}, cutoff=8.0)
        assert len(graphs) == 0
