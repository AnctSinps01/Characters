"""Euclidean distance field of a target glyph, in normalized coordinate units.

Mirrors twist's preprocessing: the field is computed once per target, detached,
and reused.  Sampling it at generated ink positions gives a Chamfer-style term
that tolerates small stroke shifts where pixel MSE does not.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor


def distance_fields(images: Tensor, threshold: float = 0.0,
                    *, allow_empty: bool = False) -> Tensor:
    """Return [B,1,H,W] distance to foreground; white +1, black -1.

    ``allow_empty`` keeps blank targets usable (their field is all zeros)
    instead of raising, which matters for synthetic tests and padded slots.
    """
    from scipy.ndimage import distance_transform_edt

    if images.ndim != 4 or images.shape[0] < 1 or min(images.shape[-2:]) < 2:
        raise ValueError("images must have shape [B,C,H>=2,W>=2] with a nonempty batch")
    if images.shape[1] < 1 or not images.is_floating_point():
        raise ValueError("images must be floating-point channels normalized to [-1, 1]")
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
            if not allow_empty:
                raise ValueError(f"image {index} has no foreground at threshold {threshold}")
            maps.append(np.zeros((height, width), dtype=np.float32))
            continue
        maps.append(distance_transform_edt(
            ~foreground, sampling=(1 / (height - 1), 1 / (width - 1))))
    dtype = torch.float64 if images.dtype == torch.float64 else torch.float32
    return torch.from_numpy(np.stack(maps)[:, None]).to(device=images.device, dtype=dtype)


def chamfer_to_target(endpoint: Tensor, target: Tensor) -> Tensor:
    """Distance-weighted error of generated ink against the target's ink.

    ``endpoint`` and ``target`` are [B,1,H,W] in [-1,1].  The target field is
    computed here and detached; gradients flow only through the generated ink
    weight, exactly like twist's ``source_distance_loss`` without a warp.
    """
    field = distance_fields(target, allow_empty=True)
    ink = ((1 - endpoint) / 2).clamp(0, 1)
    mass = ink.sum(dim=(1, 2, 3)).clamp_min(1e-8)
    error = (ink * field.to(ink.dtype).square()).sum(dim=(1, 2, 3)) / mass
    # Blank targets contribute nothing rather than an arbitrary constant.
    has_ink = (target <= 0).flatten(1).any(dim=1).to(error.dtype)
    return (error * has_ink).sum() / has_ink.sum().clamp_min(1)
