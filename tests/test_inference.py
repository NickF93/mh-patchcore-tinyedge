from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from mh_patchcore_tinyedge.bank import (
    MemoryBank,
    build_student_bank,
    load_memory_bank,
    save_memory_bank,
)
from mh_patchcore_tinyedge.checkpoint import load_checkpoint
from mh_patchcore_tinyedge.cli import main
from mh_patchcore_tinyedge.model import DepthwiseStudent, ModelConfig
from mh_patchcore_tinyedge.scoring import nearest_squared_l2, score_image


def test_nearest_squared_l2_matches_expected() -> None:
    query = np.asarray([[0.0, 0.0], [2.0, 0.0]], dtype=np.float32)
    prototypes = np.asarray([[1.0, 0.0], [4.0, 0.0]], dtype=np.float32)

    distances = nearest_squared_l2(query, prototypes, prototype_chunk_size=1)

    np.testing.assert_allclose(distances, np.asarray([1.0, 1.0], dtype=np.float32))


def test_build_save_load_and_score_memory_bank(tmp_path: Path) -> None:
    checkpoint_path = _write_checkpoint(tmp_path)
    image_dir = tmp_path / "images"
    _write_image(image_dir / "normal_0.png", value=20)
    _write_image(image_dir / "normal_1.png", value=80)
    _write_image(image_dir / "test.png", value=180)
    device = torch.device("cpu")
    checkpoint = load_checkpoint(checkpoint_path, device=device)

    bank = build_student_bank(
        checkpoint=checkpoint,
        image_paths=sorted(image_dir.glob("normal_*.png")),
        device=device,
        batch_size=1,
    )
    bank_path = tmp_path / "bank.npz"
    save_memory_bank(bank_path, bank)
    loaded_bank = load_memory_bank(bank_path)
    result = score_image(
        checkpoint=checkpoint,
        bank=loaded_bank,
        image_path=image_dir / "test.png",
        device=device,
    )

    assert loaded_bank.features.shape == (8, 3)
    assert loaded_bank.output_grid_hw == (2, 2)
    assert result.patch_scores.shape == (2, 2)
    assert result.score >= 0.0


def test_cli_build_score_and_inspect(tmp_path: Path) -> None:
    checkpoint_path = _write_checkpoint(tmp_path)
    image_dir = tmp_path / "images"
    _write_image(image_dir / "normal" / "0.png", value=40)
    _write_image(image_dir / "normal" / "1.png", value=60)
    _write_image(image_dir / "test" / "0.png", value=120)
    bank_path = tmp_path / "runtime" / "bank.npz"
    scores_path = tmp_path / "runtime" / "scores.csv"
    heatmap_dir = tmp_path / "runtime" / "heatmaps"

    assert (
        main(
            [
                "bank",
                "build",
                "--checkpoint",
                str(checkpoint_path),
                "--images",
                str(image_dir / "normal"),
                "--output",
                str(bank_path),
                "--device",
                "cpu",
                "--batch-size",
                "1",
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "score",
                "--checkpoint",
                str(checkpoint_path),
                "--bank",
                str(bank_path),
                "--input",
                str(image_dir / "test"),
                "--output",
                str(scores_path),
                "--heatmap-dir",
                str(heatmap_dir),
                "--device",
                "cpu",
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "inspect",
                "--checkpoint",
                str(checkpoint_path),
                "--bank",
                str(bank_path),
                "--device",
                "cpu",
            ]
        )
        == 0
    )

    rows = list(csv.DictReader(scores_path.open("r", encoding="utf-8")))
    assert len(rows) == 1
    assert float(rows[0]["score"]) >= 0.0
    assert Path(rows[0]["heatmap"]).is_file()


def test_score_rejects_incompatible_bank(tmp_path: Path) -> None:
    checkpoint_path = _write_checkpoint(tmp_path)
    image_path = tmp_path / "image.png"
    _write_image(image_path, value=100)
    checkpoint = load_checkpoint(checkpoint_path, device=torch.device("cpu"))
    bank = MemoryBank(
        features=np.ones((2, 4), dtype=np.float32),
        embedding_dim=4,
        output_grid_hw=(2, 2),
    )

    with pytest.raises(ValueError, match="embedding dimension"):
        score_image(
            checkpoint=checkpoint,
            bank=bank,
            image_path=image_path,
            device=torch.device("cpu"),
        )


def _write_checkpoint(tmp_path: Path) -> Path:
    model_config = ModelConfig(
        input_channels=3,
        stem_channels=4,
        stage_channels=(4,),
        stage_blocks=(1,),
        stage_strides=(1,),
    )
    model = DepthwiseStudent(
        config=model_config,
        output_dim=3,
        output_grid_hw=(2, 2),
    )
    checkpoint_path = tmp_path / "last.pt"
    torch.save(
        {
            "schema": "tinyedge.student_z.checkpoint.v1",
            "model_state_dict": model.state_dict(),
            "config": {
                "preprocessing": {
                    "resize_hw": (16, 16),
                    "crop_hw": (16, 16),
                    "mean": (0.5, 0.5, 0.5),
                    "std": (0.25, 0.25, 0.25),
                    "interpolation": "bilinear",
                },
                "model": {
                    "architecture": "depthwise_cnn",
                    "input_channels": 3,
                    "stem_channels": 4,
                    "stage_channels": (4,),
                    "stage_blocks": (1,),
                    "stage_strides": (1,),
                },
            },
            "embedding_dim": 3,
            "output_grid_hw": (2, 2),
        },
        checkpoint_path,
    )
    return checkpoint_path


def _write_image(path: Path, *, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.full((16, 16, 3), value, dtype=np.uint8)
    Image.fromarray(array, mode="RGB").save(path)
