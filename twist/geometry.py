"""Fixed boundary projection, Lipschitz bounds, and backward sampling maps.

Coordinates use [0, 1]^2 in (x, y) order. A grid maps an output location
to the original source image, so composition appends small maps on the right.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor
import torch.nn.functional as F


def geometry_dtype(tensor: Tensor) -> torch.dtype:
    return torch.float64 if tensor.dtype == torch.float64 else torch.float32


def identity_grid(batch: int, size: tuple[int, int], *, device: torch.device,
                  dtype: torch.dtype = torch.float32) -> Tensor:
    height, width = size
    if batch < 1 or min(height, width) < 2:
        raise ValueError("a sampling grid needs a nonempty batch and dimensions >= 2")
    rows, columns = torch.meshgrid(
        torch.linspace(0, 1, height, device=device, dtype=dtype),
        torch.linspace(0, 1, width, device=device, dtype=dtype), indexing="ij")
    return torch.stack((columns, rows), dim=-1).unsqueeze(0).expand(batch, -1, -1, -1)


def lipschitz_bound(displacement: Tensor) -> Tensor:
    """Per-image conservative bound for the entire bilinear control field.

    Each Jacobian column is a convex combination of neighboring edge slopes.
    The bound uses the Frobenius norm to bound the operator norm, including
    all cells rather than only Jacobians sampled at a few pixel locations.
    """
    if displacement.ndim != 4 or displacement.shape[1] != 2:
        raise ValueError("displacement must have shape [B, 2, Hc, Wc]")
    height, width = displacement.shape[-2:]
    if min(height, width) < 2:
        raise ValueError("a control field needs at least two nodes per dimension")
    field = displacement.to(dtype=geometry_dtype(displacement))
    horizontal = (field[..., 1:] - field[..., :-1]) * (width - 1)
    vertical = (field[..., 1:, :] - field[..., :-1, :]) * (height - 1)
    a = torch.linalg.vector_norm(horizontal, dim=1).amax(dim=(-2, -1))
    b = torch.linalg.vector_norm(vertical, dim=1).amax(dim=(-2, -1))
    # vector_norm has a finite zero subgradient, unlike sqrt(a*a + b*b).
    return torch.linalg.vector_norm(torch.stack((a, b), dim=-1), dim=-1)


def constrain_displacement(raw: Tensor, limit: float, *, epsilon: float = 1e-6) -> Tensor:
    """Fix the boundary first, then rescale to a strict sub-unit bound."""
    if not math.isfinite(limit) or not 0 < limit < 1:
        raise ValueError("limit must be strictly between zero and one")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    if raw.ndim != 4 or raw.shape[1] != 2 or min(raw.shape[-2:]) < 3:
        raise ValueError("raw control field must have shape [B, 2, Hc>=3, Wc>=3]")
    field = raw.to(dtype=geometry_dtype(raw))
    # Rebuild the zero ring so boundary predictions cannot influence the bound.
    field = F.pad(field[:, :, 1:-1, 1:-1], (1, 1, 1, 1))
    bound = lipschitz_bound(field)
    scale = (limit / (bound + epsilon)).clamp(max=1.0)
    return field * scale[:, None, None, None]


def sample_displacement(displacement: Tensor, coordinates: Tensor) -> Tensor:
    """Evaluate a continuous bilinear field, with zero displacement outside."""
    if coordinates.ndim != 4 or coordinates.shape[-1] != 2:
        raise ValueError("coordinates must have shape [B, H, W, 2]")
    if displacement.shape[0] != coordinates.shape[0]:
        raise ValueError("field and coordinates must have the same batch size")
    field = displacement.to(dtype=coordinates.dtype)
    sampled = F.grid_sample(field, coordinates * 2 - 1, mode="bilinear",
                            padding_mode="zeros", align_corners=True)
    return sampled.permute(0, 2, 3, 1)


def sample_image(images: Tensor, coordinates: Tensor) -> Tensor:
    """Resample the original image; retain its border under roundoff error."""
    if images.ndim != 4 or coordinates.ndim != 4 or coordinates.shape[-1] != 2:
        raise ValueError("expected images [B,C,H,W] and coordinates [B,Hout,Wout,2]")
    if images.shape[0] != coordinates.shape[0] or images.device != coordinates.device:
        raise ValueError("images and coordinates must share their batch size and device")
    return F.grid_sample(images.to(dtype=coordinates.dtype), coordinates * 2 - 1,
                         mode="bilinear", padding_mode="border", align_corners=True)


def _time_scale(time: float | Tensor, coordinates: Tensor) -> Tensor:
    batch = coordinates.shape[0]
    if isinstance(time, (int, float)):
        if not math.isfinite(time) or not 0 <= time <= 1:
            raise ValueError("time must be finite and in [0, 1]")
        times = coordinates.new_tensor(float(time))
    else:
        times = time.to(device=coordinates.device, dtype=coordinates.dtype)
        if not bool(torch.isfinite(times).all()) or bool(((times < 0) | (times > 1)).any()):
            raise ValueError("time must be finite and in [0, 1]")
    if times.ndim == 0:
        return times.reshape(1, 1, 1, 1)
    if times.shape != (batch,):
        raise ValueError("time must be a scalar or a tensor with shape [B]")
    return times[:, None, None, None]


def compose_displacements(displacements: Sequence[Tensor], coordinates: Tensor,
                          time: float | Tensor = 1.0) -> Tensor:
    """Evaluate g1(g2(...gK(x))) using the original control fields.

    Evaluating each field at the updated coordinates preserves the actual
    continuous composition. Re-interpolating a cached coarse cumulative map
    would instead introduce a new approximation with no inherited bound.
    """
    coordinates = coordinates.to(dtype=geometry_dtype(coordinates))
    strength = _time_scale(time, coordinates)
    for field in reversed(displacements):
        coordinates = coordinates + strength * sample_displacement(field, coordinates)
    return coordinates


def composed_grid(displacements: Sequence[Tensor], source: Tensor,
                  time: float | Tensor = 1.0) -> Tensor:
    coordinates = identity_grid(source.shape[0], source.shape[-2:], device=source.device,
                                dtype=geometry_dtype(source))
    return compose_displacements(displacements, coordinates, time)
