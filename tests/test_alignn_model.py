"""Tests for ALIGNN model wrapper: build, train, predict."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
dgl = pytest.importorskip("dgl")
pytest.importorskip("alignn")

from crystal_prop_bench.data.graph_builder import build_alignn_graphs  # noqa: E402
from crystal_prop_bench.models.alignn_model import (  # noqa: E402
    ALIGNNDataset,
    build_alignn,
    collate_alignn,
    predict_alignn,
    train_alignn,
)


@pytest.fixture
def small_graphs(fixture_crystals: pd.DataFrame, fixture_structures: dict) -> dict:
    """Build graphs from fixture structures (small subset for speed)."""
    # Use first 20 structures for fast tests
    small_df = fixture_crystals.head(20).copy()
    graphs = build_alignn_graphs(small_df, fixture_structures, cutoff=8.0)
    return graphs


class TestBuildAlignn:
    def test_returns_model(self) -> None:
        model = build_alignn()
        assert model is not None

    def test_model_has_parameters(self) -> None:
        model = build_alignn()
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params > 0
        # ~400K params expected
        assert n_params < 2_000_000


class TestALIGNNDataset:
    def test_length(self, small_graphs: dict, fixture_crystals: pd.DataFrame) -> None:
        graph_ids = [mid for mid in fixture_crystals["material_id"].head(20) if mid in small_graphs]
        targets = np.random.randn(len(graph_ids))
        ds = ALIGNNDataset(small_graphs, graph_ids, targets)
        assert len(ds) == len(graph_ids)

    def test_getitem_returns_triple(self, small_graphs: dict, fixture_crystals: pd.DataFrame) -> None:
        graph_ids = [mid for mid in fixture_crystals["material_id"].head(20) if mid in small_graphs]
        targets = np.random.randn(len(graph_ids))
        ds = ALIGNNDataset(small_graphs, graph_ids, targets)
        g, lg, t = ds[0]
        assert isinstance(g, dgl.DGLGraph)
        assert isinstance(lg, dgl.DGLGraph)
        assert isinstance(t, (float, np.floating))


class TestCollateAlignn:
    def test_batches_graphs(self, small_graphs: dict, fixture_crystals: pd.DataFrame) -> None:
        graph_ids = [mid for mid in fixture_crystals["material_id"].head(20) if mid in small_graphs]
        targets = np.random.randn(len(graph_ids))
        ds = ALIGNNDataset(small_graphs, graph_ids, targets)
        batch = collate_alignn([ds[i] for i in range(min(4, len(ds)))])
        bg, blg, bt = batch
        assert isinstance(bg, dgl.DGLGraph)
        assert isinstance(blg, dgl.DGLGraph)
        assert bt.shape[0] == min(4, len(ds))


class TestForwardPass:
    def test_single_graph(self, small_graphs: dict) -> None:
        model = build_alignn()
        model.eval()
        mid = next(iter(small_graphs))
        g, lg = small_graphs[mid]
        with torch.no_grad():
            out = model(g, lg)
        assert out.shape == (1, 1) or out.shape == (1,)

    def test_batched_graphs(self, small_graphs: dict, fixture_crystals: pd.DataFrame) -> None:
        model = build_alignn()
        model.eval()
        graph_ids = [mid for mid in fixture_crystals["material_id"].head(20) if mid in small_graphs][:4]
        gs = [small_graphs[mid][0] for mid in graph_ids]
        lgs = [small_graphs[mid][1] for mid in graph_ids]
        bg = dgl.batch(gs)
        blg = dgl.batch(lgs)
        with torch.no_grad():
            out = model(bg, blg)
        assert out.shape[0] == len(graph_ids)


class TestPredictAlignn:
    def test_returns_correct_length(self, small_graphs: dict) -> None:
        model = build_alignn()
        model.eval()
        graph_ids = list(small_graphs.keys())
        preds = predict_alignn(model, small_graphs, graph_ids, device="cpu", batch_size=8)
        assert len(preds) == len(graph_ids)
        assert isinstance(preds, np.ndarray)


class TestTrainAlignn:
    def test_trains_and_returns_residuals(
        self, small_graphs: dict, fixture_crystals: pd.DataFrame,
    ) -> None:
        """Smoke test: train for 2 epochs on tiny data, verify contract."""
        graph_ids = [mid for mid in fixture_crystals["material_id"].head(20) if mid in small_graphs]
        if len(graph_ids) < 8:
            pytest.skip("Not enough graphs for train/val/cal split")

        rng = np.random.RandomState(42)
        targets = rng.randn(len(graph_ids))

        # Split: 50% train, 25% val, 25% cal
        n = len(graph_ids)
        n_train = n // 2
        n_val = n // 4
        train_ids = graph_ids[:n_train]
        val_ids = graph_ids[n_train:n_train + n_val]
        cal_ids = graph_ids[n_train + n_val:]
        train_targets = targets[:n_train]
        val_targets = targets[n_train:n_train + n_val]
        cal_targets = targets[n_train + n_val:]

        model = build_alignn()
        model, cal_residuals = train_alignn(
            model=model,
            graphs=small_graphs,
            train_ids=train_ids,
            train_targets=train_targets,
            val_ids=val_ids,
            val_targets=val_targets,
            cal_ids=cal_ids,
            cal_targets=cal_targets,
            seed=42,
            epochs=2,
            lr=1e-3,
            patience=5,
            batch_size=4,
            device="cpu",
        )
        assert cal_residuals is not None
        assert len(cal_residuals) == len(cal_ids)
        assert (cal_residuals >= 0).all()
