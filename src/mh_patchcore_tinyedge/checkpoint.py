"""Checkpoint loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from mh_patchcore_tinyedge.model import DepthwiseStudent, parse_model_config
from mh_patchcore_tinyedge.preprocess import PreprocessingConfig, parse_preprocessing_config


@dataclass(frozen=True)
class LoadedCheckpoint:
    path: Path
    model: DepthwiseStudent
    preprocessing: PreprocessingConfig
    embedding_dim: int
    output_grid_hw: tuple[int, int]


def load_checkpoint(path: str | Path, *, device: torch.device) -> LoadedCheckpoint:
    checkpoint_path = Path(path)
    raw = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(raw, Mapping):
        raise ValueError(f"invalid checkpoint: {checkpoint_path}")
    if raw.get("schema") != "tinyedge.student_z.checkpoint.v1":
        raise ValueError("unsupported checkpoint schema")
    config = _mapping(raw.get("config"), "config")
    embedding_dim = _positive_int(raw.get("embedding_dim"), "embedding_dim")
    output_grid_hw = _grid(raw.get("output_grid_hw"))
    model_config = parse_model_config(_mapping(config.get("model"), "config.model"))
    preprocessing = parse_preprocessing_config(
        _mapping(config.get("preprocessing"), "config.preprocessing")
    )
    model = DepthwiseStudent(
        config=model_config,
        output_dim=embedding_dim,
        output_grid_hw=output_grid_hw,
    )
    state_dict = raw.get("model_state_dict")
    if not isinstance(state_dict, Mapping):
        raise ValueError("checkpoint is missing model_state_dict")
    model.load_state_dict(dict(state_dict))
    model.to(device)
    model.eval()
    return LoadedCheckpoint(
        path=checkpoint_path,
        model=model,
        preprocessing=preprocessing,
        embedding_dim=embedding_dim,
        output_grid_hw=output_grid_hw,
    )


def _mapping(value: object, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"checkpoint field must be a mapping: {field_name}")
    return value


def _positive_int(value: object, field_name: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"checkpoint field must be positive: {field_name}")
    return parsed


def _grid(value: object) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("checkpoint output_grid_hw must contain two integers")
    grid = (int(value[0]), int(value[1]))
    if grid[0] <= 0 or grid[1] <= 0:
        raise ValueError("checkpoint output_grid_hw values must be positive")
    return grid
