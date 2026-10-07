"""Prepare and cache Euclidean distance fields of original input images."""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor

from .geometry import geometry_dtype


def distance_fields(images: Tensor, threshold: float = 0.0) -> Tensor:
    """Return [B,1,H,W] distance to foreground, in normalized coordinate units.

    White is +1 and black is -1. Preparation is deliberately detached: only
    sampling coordinates, not the distance-transform algorithm, need gradients.
    Compute once per original image and reuse across steps and stages.
    """
    from scipy.ndimage import distance_transform_edt

    if images.ndim != 4 or images.shape[0] < 1 or min(images.shape[-2:]) < 2:
        raise ValueError("images must have shape [B,C,H>=2,W>=2] with a nonempty batch")
    if images.shape[1] < 1 or not images.is_floating_point():
        raise ValueError("images must have floating-point channels normalized to [-1, 1]")
    if not math.isfinite(threshold) or not -1 <= threshold < 1:
        raise ValueError("threshold must be in [-1, 1)")
    values = images.detach().to(device="cpu", dtype=torch.float32).mean(dim=1).numpy()
    if not np.isfinite(values).all() or values.min() < -1 or values.max() > 1:
        raise ValueError("images must contain finite values in [-1, 1]")
    height, width = images.shape[-2:]
    maps = []
    for index, image in enumerate(values):
        foreground = image <= threshold
        if not foreground.any():
            raise ValueError(f"image {index} has no foreground at threshold {threshold}")
        maps.append(distance_transform_edt(
            ~foreground, sampling=(1 / (height - 1), 1 / (width - 1))))
    return torch.from_numpy(np.stack(maps)[:, None]).to(
        device=images.device, dtype=geometry_dtype(images))
