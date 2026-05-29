"""Image path collection."""

from __future__ import annotations

from pathlib import Path


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def collect_image_paths(path: str | Path) -> list[Path]:
    root = Path(path)
    if root.is_file():
        if root.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"unsupported image extension: {root}")
        return [root]
    if not root.is_dir():
        raise ValueError(f"image path does not exist: {root}")
    paths = [
        candidate
        for candidate in root.rglob("*")
        if candidate.is_file() and candidate.suffix.lower() in IMAGE_SUFFIXES
    ]
    if not paths:
        raise ValueError(f"no supported images found under: {root}")
    return sorted(paths)
