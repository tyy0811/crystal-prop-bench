"""ALIGNN model wrapper: build, train, predict.

Imports the ALIGNN architecture from the alignn package.
Owns the training loop, early stopping, and prediction export.
Mirrors the train_lgbm contract: returns (model, cal_residuals).
"""

from __future__ import annotations

import copy
import logging

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


class ALIGNNDataset(Dataset):  # type: ignore[type-arg]
    """Dataset wrapping (atom_graph, line_graph, lattice, target) tuples."""

    def __init__(
        self,
        graphs: dict[str, tuple],
        material_ids: list[str],
        targets: np.ndarray,
    ) -> None:
        self.graphs = graphs
        self.material_ids = material_ids
        self.targets = targets

    def __len__(self) -> int:
        return len(self.material_ids)

    def __getitem__(self, idx: int) -> tuple:
        mid = self.material_ids[idx]
        g, lg, lat = self.graphs[mid]
        return g, lg, torch.tensor(lat, dtype=torch.float32), float(self.targets[idx])


def collate_alignn(samples: list[tuple]) -> tuple:
    """Collate function for ALIGNN DataLoader.

    Returns (batched_graph, batched_line_graph, stacked_lattices, targets).
    """
    import dgl

    gs, lgs, lats, targets = zip(*samples)
    return (
        dgl.batch(list(gs)),
        dgl.batch(list(lgs)),
        torch.stack(list(lats)),
        torch.tensor(targets, dtype=torch.float32),
    )


def build_alignn(
    alignn_layers: int = 4,
    gcn_layers: int = 4,
    embedding_features: int = 64,
    hidden_features: int = 256,
    output_features: int = 1,
) -> torch.nn.Module:
    """Build ALIGNN model with specified architecture."""
    from alignn.models.alignn import ALIGNN, ALIGNNConfig

    config = ALIGNNConfig(
        name="alignn",
        alignn_layers=alignn_layers,
        gcn_layers=gcn_layers,
        atom_input_features=92,
        edge_input_features=80,
        triplet_input_features=40,
        embedding_features=embedding_features,
        hidden_features=hidden_features,
        output_features=output_features,
    )
    return ALIGNN(config)


def predict_alignn(
    model: torch.nn.Module,
    graphs: dict[str, tuple],
    material_ids: list[str],
    device: str = "cuda",
    batch_size: int = 256,
) -> np.ndarray:
    """Batch inference. Returns predictions array matching material_ids order."""
    dummy_targets = np.zeros(len(material_ids))
    ds = ALIGNNDataset(graphs, material_ids, dummy_targets)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_alignn)

    model.eval()
    all_preds: list[np.ndarray] = []
    with torch.no_grad():
        for bg, blg, blat, _ in loader:
            bg = bg.to(device)
            blg = blg.to(device)
            blat = blat.to(device)
            out = model([bg, blg, blat])
            pred = out["out"] if isinstance(out, dict) else out
            all_preds.append(pred.reshape(-1).cpu().numpy())

    return np.concatenate(all_preds)


def train_alignn(
    model: torch.nn.Module,
    graphs: dict[str, tuple],
    train_ids: list[str],
    train_targets: np.ndarray,
    val_ids: list[str],
    val_targets: np.ndarray,
    cal_ids: list[str],
    cal_targets: np.ndarray,
    seed: int = 42,
    epochs: int = 200,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    patience: int = 30,
    scheduler_patience: int = 10,
    batch_size: int = 256,
    device: str = "cuda",
) -> tuple[torch.nn.Module, np.ndarray, np.ndarray]:
    """Train ALIGNN and compute calibration residuals.

    Mirrors train_lgbm contract:
    - Val set for early stopping only.
    - Cal set held out, residuals computed post-training.
    - Returns (model, cal_residuals, cal_preds).
    """
    torch.manual_seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)

    model = model.to(device)

    train_ds = ALIGNNDataset(graphs, train_ids, train_targets)
    val_ds = ALIGNNDataset(graphs, val_ids, val_targets)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_alignn, drop_last=False, generator=g,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_alignn,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=scheduler_patience, factor=0.5,
    )

    best_val_mae = float("inf")
    patience_counter = 0
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):
        # Train
        model.train()
        train_losses: list[float] = []
        for bg, blg, blat, targets in train_loader:
            bg, blg, blat = bg.to(device), blg.to(device), blat.to(device)
            targets = targets.to(device)
            out = model([bg, blg, blat])
            preds = out["out"] if isinstance(out, dict) else out
            preds = preds.reshape(-1)
            loss = F.l1_loss(preds, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # Validate
        model.eval()
        val_preds_list: list[np.ndarray] = []
        val_true_list: list[np.ndarray] = []
        with torch.no_grad():
            for bg, blg, blat, targets in val_loader:
                bg, blg, blat = bg.to(device), blg.to(device), blat.to(device)
                out = model([bg, blg, blat])
                preds = out["out"] if isinstance(out, dict) else out
                preds = preds.reshape(-1)
                val_preds_list.append(preds.cpu().numpy())
                val_true_list.append(targets.numpy())

        val_preds_arr = np.concatenate(val_preds_list)
        val_true_arr = np.concatenate(val_true_list)
        val_mae = float(np.mean(np.abs(val_true_arr - val_preds_arr)))

        scheduler.step(val_mae)

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_counter = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0 or patience_counter >= patience:
            logger.info(
                "Epoch %d/%d: train_mae=%.4f, val_mae=%.4f, best=%.4f, patience=%d/%d",
                epoch + 1, epochs, np.mean(train_losses), val_mae,
                best_val_mae, patience_counter, patience,
            )

        if patience_counter >= patience:
            logger.info("Early stopping at epoch %d", epoch + 1)
            break

    model.load_state_dict(best_state)
    model.eval()

    # Compute calibration residuals (cal set never seen during training)
    cal_preds = predict_alignn(model, graphs, cal_ids, device=device, batch_size=batch_size)
    cal_residuals = np.abs(cal_targets - cal_preds)

    logger.info(
        "Trained ALIGNN (seed=%d): best_val_mae=%.4f, cal_MAE=%.4f",
        seed, best_val_mae, float(np.mean(cal_residuals)),
    )

    return model, cal_residuals, cal_preds
