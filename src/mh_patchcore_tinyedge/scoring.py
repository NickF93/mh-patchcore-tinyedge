"""Student-bank anomaly scoring."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from mh_patchcore_tinyedge.bank import MemoryBank
from mh_patchcore_tinyedge.checkpoint import LoadedCheckpoint
from mh_patchcore_tinyedge.model import flatten_features
from mh_patchcore_tinyedge.preprocess import load_image_tensor


@dataclass(frozen=True)
class ImageScore:
    path: Path
    score: float
    patch_scores: np.ndarray


@torch.no_grad()
def score_image(
    *,
    checkpoint: LoadedCheckpoint,
    bank: MemoryBank,
    image_path: str | Path,
    device: torch.device,
    prototype_chunk_size: int = 4096,
) -> ImageScore:
    _validate_bank_for_checkpoint(bank, checkpoint)
    path = Path(image_path)
    image = load_image_tensor(str(path), checkpoint.preprocessing)
    batch = image.unsqueeze(0).to(device=device, dtype=torch.float32)
    features = flatten_features(checkpoint.model(batch))[0]
    predicted = features.detach().cpu().numpy().astype(np.float32, copy=False)
    patch_scores = nearest_squared_l2(
        predicted,
        bank.features,
        prototype_chunk_size=prototype_chunk_size,
    )
    return ImageScore(
        path=path,
        score=float(np.max(patch_scores)),
        patch_scores=patch_scores.reshape(checkpoint.output_grid_hw),
    )


def score_images(
    *,
    checkpoint: LoadedCheckpoint,
    bank: MemoryBank,
    image_paths: Iterable[Path],
    device: torch.device,
    prototype_chunk_size: int = 4096,
    progress: object | None = None,
    task_id: object | None = None,
) -> list[ImageScore]:
    scores: list[ImageScore] = []
    for path in image_paths:
        scores.append(
            score_image(
                checkpoint=checkpoint,
                bank=bank,
                image_path=path,
                device=device,
                prototype_chunk_size=prototype_chunk_size,
            )
        )
        if progress is not None and task_id is not None:
            progress.advance(task_id)
    return scores


def nearest_squared_l2(
    query: np.ndarray,
    prototypes: np.ndarray,
    *,
    prototype_chunk_size: int = 4096,
) -> np.ndarray:
    if query.ndim != 2 or prototypes.ndim != 2 or query.shape[1] != prototypes.shape[1]:
        raise ValueError(
            "query and prototypes must have shapes [P, D] and [N, D] with matching D"
        )
    if prototype_chunk_size <= 0:
        raise ValueError("prototype_chunk_size must be positive")
    query32 = query.astype(np.float32, copy=False)
    prototypes32 = prototypes.astype(np.float32, copy=False)
    query_sq = np.sum(query32 * query32, axis=1, keepdims=True)
    best = np.full((query32.shape[0],), np.inf, dtype=np.float32)
    for start in range(0, prototypes32.shape[0], prototype_chunk_size):
        chunk = prototypes32[start : start + prototype_chunk_size]
        chunk_sq = np.sum(chunk * chunk, axis=1, keepdims=True).T
        distances = query_sq + chunk_sq - 2.0 * np.matmul(query32, chunk.T)
        np.maximum(distances, 0.0, out=distances)
        best = np.minimum(best, np.min(distances, axis=1))
    return best.astype(np.float32, copy=False)


def _validate_bank_for_checkpoint(bank: MemoryBank, checkpoint: LoadedCheckpoint) -> None:
    if bank.embedding_dim != checkpoint.embedding_dim:
        raise ValueError(
            "memory-bank embedding dimension does not match checkpoint: "
            f"{bank.embedding_dim} != {checkpoint.embedding_dim}"
        )
    if bank.output_grid_hw != checkpoint.output_grid_hw:
        raise ValueError(
            "memory-bank output grid does not match checkpoint: "
            f"{bank.output_grid_hw} != {checkpoint.output_grid_hw}"
        )
