"""Image preprocessing used by exported checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch
from PIL import Image


@dataclass(frozen=True)
class PreprocessingConfig:
    resize_hw: tuple[int, int]
    crop_hw: tuple[int, int]
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    interpolation: str


def parse_preprocessing_config(raw: Mapping[str, Any]) -> PreprocessingConfig:
    return PreprocessingConfig(
        resize_hw=_pair(raw.get("resize_hw"), "resize_hw"),
        crop_hw=_pair(raw.get("crop_hw"), "crop_hw"),
        mean=_float_tuple(raw.get("mean"), "mean"),
        std=_float_tuple(raw.get("std"), "std"),
        interpolation=str(raw.get("interpolation", "bilinear")),
    )


def load_image_tensor(path: str, config: PreprocessingConfig) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    resample = Image.BILINEAR if config.interpolation == "bilinear" else Image.NEAREST
    resize_h, resize_w = config.resize_hw
    image = image.resize((resize_w, resize_h), resample=resample)
    image = _center_crop(image, config.crop_hw)
    array = np.asarray(image, dtype=np.float32) / 255.0
    mean = np.asarray(config.mean, dtype=np.float32)
    std = np.asarray(config.std, dtype=np.float32)
    if np.any(std == 0.0):
        raise ValueError("preprocessing std values must be non-zero")
    array = (array - mean) / std
    return torch.from_numpy(array.transpose(2, 0, 1)).contiguous()


def _center_crop(image: Image.Image, crop_hw: tuple[int, int]) -> Image.Image:
    crop_h, crop_w = crop_hw
    width, height = image.size
    if crop_h > height or crop_w > width:
        raise ValueError(
            f"crop size {crop_hw} exceeds resized image size {(height, width)}"
        )
    left = (width - crop_w) // 2
    top = (height - crop_h) // 2
    return image.crop((left, top, left + crop_w, top + crop_h))


def _pair(value: object, field_name: str) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"preprocessing.{field_name} must contain two integers")
    pair = (int(value[0]), int(value[1]))
    if pair[0] <= 0 or pair[1] <= 0:
        raise ValueError(f"preprocessing.{field_name} values must be positive")
    return pair


def _float_tuple(value: object, field_name: str) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"preprocessing.{field_name} must contain three numbers")
    return (float(value[0]), float(value[1]), float(value[2]))
