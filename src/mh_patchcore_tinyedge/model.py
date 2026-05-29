"""Student model definition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class ModelConfig:
    input_channels: int
    stem_channels: int
    stage_channels: tuple[int, ...]
    stage_blocks: tuple[int, ...]
    stage_strides: tuple[int, ...]


class DepthwiseStudent(nn.Module):
    """Small fully convolutional image-to-feature model."""

    def __init__(
        self,
        *,
        config: ModelConfig,
        output_dim: int,
        output_grid_hw: tuple[int, int],
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(
                config.input_channels,
                config.stem_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(config.stem_channels),
            nn.ReLU6(inplace=True),
        ]
        current_channels = config.stem_channels
        for channels, blocks, stride in zip(
            config.stage_channels,
            config.stage_blocks,
            config.stage_strides,
            strict=True,
        ):
            for block_index in range(blocks):
                layers.append(
                    DepthwiseSeparableBlock(
                        in_channels=current_channels,
                        out_channels=channels,
                        stride=stride if block_index == 0 else 1,
                    )
                )
                current_channels = channels
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(output_grid_hw)
        self.head = nn.Conv2d(current_channels, output_dim, kernel_size=1)
        self.output_dim = output_dim
        self.output_grid_hw = output_grid_hw

    def forward(self, images: Tensor) -> Tensor:
        return self.head(self.pool(self.features(images)))


class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, *, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),
        )

    def forward(self, value: Tensor) -> Tensor:
        return self.block(value)


def flatten_features(features: Tensor) -> Tensor:
    if features.ndim != 4:
        raise ValueError(f"model output must have shape [B, D, H, W]: {features.shape}")
    return features.permute(0, 2, 3, 1).reshape(
        features.shape[0],
        features.shape[2] * features.shape[3],
        features.shape[1],
    )


def parse_model_config(raw: Mapping[str, Any]) -> ModelConfig:
    if str(raw.get("architecture", "depthwise_cnn")) != "depthwise_cnn":
        raise ValueError("checkpoint model architecture must be depthwise_cnn")
    return ModelConfig(
        input_channels=_positive_int(raw.get("input_channels", 3), "input_channels"),
        stem_channels=_positive_int(raw.get("stem_channels", 16), "stem_channels"),
        stage_channels=_int_tuple(raw.get("stage_channels", (32, 64)), "stage_channels"),
        stage_blocks=_int_tuple(raw.get("stage_blocks", (1, 1)), "stage_blocks"),
        stage_strides=_int_tuple(raw.get("stage_strides", (2, 2)), "stage_strides"),
    )


def _int_tuple(value: object, field_name: str) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"model.{field_name} must be a non-empty list")
    parsed = tuple(int(item) for item in value)
    if any(item <= 0 for item in parsed):
        raise ValueError(f"model.{field_name} values must be positive")
    return parsed


def _positive_int(value: object, field_name: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"model.{field_name} must be positive")
    return parsed
