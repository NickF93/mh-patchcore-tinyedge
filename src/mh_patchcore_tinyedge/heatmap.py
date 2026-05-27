"""Heatmap rendering."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def save_heatmap(path: str | Path, values: np.ndarray) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if values.ndim != 2:
        raise ValueError(f"heatmap values must have shape [H, W]: {values.shape}")
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("heatmap values must contain finite values")
    low = float(np.min(finite))
    high = float(np.max(finite))
    if high <= low:
        scaled = np.zeros(values.shape, dtype=np.uint8)
    else:
        scaled = np.clip((values - low) / (high - low), 0.0, 1.0)
        scaled = (scaled * 255.0).astype(np.uint8)
    Image.fromarray(scaled, mode="L").save(output)
