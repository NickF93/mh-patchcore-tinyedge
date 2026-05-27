"""Runtime device selection."""

from __future__ import annotations

import re

import torch


def resolve_device(requested: str) -> torch.device:
    value = requested.strip().lower()
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if value == "cpu":
        return torch.device("cpu")
    if value == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is not available")
        return torch.device("cuda:0")
    match = re.fullmatch(r"cuda:(\d+)", value)
    if match is None:
        raise ValueError("device must be one of: auto, cpu, cuda, cuda:<index>")
    if not torch.cuda.is_available():
        raise ValueError(f"{value} was requested but CUDA is not available")
    index = int(match.group(1))
    device_count = torch.cuda.device_count()
    if index >= device_count:
        raise ValueError(f"{value} is unavailable; CUDA device count is {device_count}")
    return torch.device(value)
