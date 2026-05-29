"""Student memory-bank export and loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from mh_patchcore_tinyedge.checkpoint import LoadedCheckpoint
from mh_patchcore_tinyedge.model import flatten_features
from mh_patchcore_tinyedge.preprocess import load_image_tensor


BANK_SCHEMA = "mh_patchcore_tinyedge.memory_bank.v1"


@dataclass(frozen=True)
class MemoryBank:
    features: np.ndarray
    embedding_dim: int
    output_grid_hw: tuple[int, int]


@torch.no_grad()
def build_student_bank(
    *,
    checkpoint: LoadedCheckpoint,
    image_paths: Iterable[Path],
    device: torch.device,
    batch_size: int = 8,
    progress: object | None = None,
    task_id: object | None = None,
) -> MemoryBank:
    paths = list(image_paths)
    if not paths:
        raise ValueError("at least one image is required to build a memory bank")
    chunks: list[np.ndarray] = []
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start : start + batch_size]
        images = [
            load_image_tensor(str(path), checkpoint.preprocessing) for path in batch_paths
        ]
        batch = torch.stack(images, dim=0).to(device=device, dtype=torch.float32)
        features = flatten_features(checkpoint.model(batch))
        values = features.detach().cpu().numpy().astype(np.float32, copy=False)
        if values.ndim != 3 or values.shape[2] != checkpoint.embedding_dim:
            raise ValueError(
                "model output does not match checkpoint embedding dimension: "
                f"{values.shape}"
            )
        chunks.append(values.reshape(-1, checkpoint.embedding_dim))
        if progress is not None and task_id is not None:
            progress.advance(task_id, advance=len(batch_paths))
    bank = np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
    _validate_features(bank, checkpoint.embedding_dim)
    return MemoryBank(
        features=bank,
        embedding_dim=checkpoint.embedding_dim,
        output_grid_hw=checkpoint.output_grid_hw,
    )


def save_memory_bank(path: str | Path, bank: MemoryBank) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        schema=np.asarray(BANK_SCHEMA),
        features=bank.features.astype(np.float32, copy=False),
        embedding_dim=np.asarray(bank.embedding_dim, dtype=np.int64),
        output_grid_hw=np.asarray(bank.output_grid_hw, dtype=np.int64),
    )


def load_memory_bank(path: str | Path) -> MemoryBank:
    bank_path = Path(path)
    with np.load(bank_path) as data:
        schema = str(data["schema"].item())
        if schema != BANK_SCHEMA:
            raise ValueError(f"unsupported memory-bank schema: {schema}")
        features = data["features"].astype(np.float32, copy=False)
        embedding_dim = int(data["embedding_dim"].item())
        grid_raw = data["output_grid_hw"].astype(np.int64, copy=False)
    output_grid_hw = (int(grid_raw[0]), int(grid_raw[1]))
    _validate_features(features, embedding_dim)
    if output_grid_hw[0] <= 0 or output_grid_hw[1] <= 0:
        raise ValueError("memory-bank output_grid_hw values must be positive")
    return MemoryBank(
        features=features,
        embedding_dim=embedding_dim,
        output_grid_hw=output_grid_hw,
    )


def _validate_features(features: np.ndarray, embedding_dim: int) -> None:
    if features.ndim != 2 or features.shape[0] <= 0 or features.shape[1] != embedding_dim:
        raise ValueError(
            "memory-bank features must have shape [N, D] with D matching the "
            f"checkpoint: {features.shape}, D={embedding_dim}"
        )
    if not np.all(np.isfinite(features)):
        raise ValueError("memory-bank features must contain only finite values")
