"""Stationary velocity exponentials and backward sampling on [0,1]^2."""
from __future__ import annotations

import math

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


def zero_boundary(velocity: Tensor) -> Tensor:
    """Fix only the boundary; preserve every interior velocity without clipping."""
    if velocity.ndim != 4 or velocity.shape[1] != 2 or min(velocity.shape[-2:]) < 2:
        raise ValueError("velocity must have shape [B,2,H>=2,W>=2]")
    field = velocity.to(dtype=geometry_dtype(velocity))
    return F.pad(field[:, :, 1:-1, 1:-1], (1, 1, 1, 1))


def sample_displacement(displacement: Tensor, coordinates: Tensor) -> Tensor:
    """Evaluate a continuous bilinear field, with zero displacement outside."""
    if coordinates.ndim != 4 or coordinates.shape[-1] != 2:
        raise ValueError("coordinates must have shape [B, H, W, 2]")
    if displacement.ndim != 4 or displacement.shape[1] != 2:
        raise ValueError("displacement must have shape [B,2,H,W]")
    if displacement.shape[0] != coordinates.shape[0] or displacement.device != coordinates.device:
        raise ValueError("field and coordinates must share batch size and device")
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


def time_scale(time: float | Tensor, reference: Tensor, *,
               minimum: float = -1.0, maximum: float = 1.0) -> Tensor:
    if isinstance(time, (int, float)):
        if not math.isfinite(time) or not minimum <= time <= maximum:
            raise ValueError(f"time must be finite and in [{minimum}, {maximum}]")
        times = reference.new_tensor(float(time))
    elif isinstance(time, Tensor):
        times = time.to(device=reference.device, dtype=reference.dtype)
        if not bool(torch.isfinite(times).all()) or bool(((times < minimum) | (times > maximum)).any()):
            raise ValueError(f"time must be finite and in [{minimum}, {maximum}]")
    else:
        raise ValueError("time must be a scalar or a tensor with shape [B]")
    if times.ndim == 0:
        return times.reshape(1, 1, 1, 1)
    if times.shape != (reference.shape[0],):
        raise ValueError("time must be a scalar or a tensor with shape [B]")
    return times[:, None, None, None]


def scaling_and_squaring(velocity: Tensor, *, time: float | Tensor = 1.0,
                         squaring_steps: int = 7) -> Tensor:
    """Approximate exp(time*v)-id as a dense displacement [B,2,H,W].

    Start with d=time*v/2**n, then square id+d via d <- d+d(id+d).
    Time scales the velocity before integration, never the final displacement.
    Negative time integrates the same stationary field in reverse.
    """
    if type(squaring_steps) is not int or squaring_steps < 0:
        raise ValueError("squaring_steps must be a nonnegative integer")
    field = zero_boundary(velocity)
    coordinates = identity_grid(field.shape[0], field.shape[-2:],
                                device=field.device, dtype=field.dtype)
    strength = time_scale(time, coordinates)
    displacement = field * strength * (2.0 ** -squaring_steps)
    for _ in range(squaring_steps):
        shifted = coordinates + displacement.permute(0, 2, 3, 1)
        displacement = displacement + sample_displacement(displacement, shifted).permute(0, 3, 1, 2)
    return displacement


def velocity_grid(velocity: Tensor, reference: Tensor, time: float | Tensor = 1.0,
                  *, squaring_steps: int = 7) -> Tensor:
    """Backward map exp(time*v); velocity must already match image resolution."""
    if velocity.ndim != 4 or velocity.shape[1] != 2 or reference.ndim != 4:
        raise ValueError("expected velocity [B,2,H,W] and reference images [B,C,H,W]")
    if velocity.shape[-2:] != reference.shape[-2:]:
        raise ValueError("velocity must have the same spatial resolution as the reference image")
    if velocity.shape[0] != reference.shape[0] or velocity.device != reference.device:
        raise ValueError("velocity and reference must share batch size and device")
    displacement = scaling_and_squaring(velocity, time=time,
                                         squaring_steps=squaring_steps)
    coordinates = identity_grid(reference.shape[0], reference.shape[-2:], device=reference.device,
                                dtype=displacement.dtype)
    return coordinates + displacement.permute(0, 2, 3, 1)


def compose_grids(outer: Tensor, inner: Tensor) -> Tensor:
    """Return outer(inner(x)); interpolate displacement, not absolute coordinates."""
    if outer.shape != inner.shape or outer.ndim != 4 or outer.shape[-1] != 2:
        raise ValueError("grids must have equal shape [B,H,W,2]")
    coordinates = identity_grid(outer.shape[0], outer.shape[1:3],
                                device=outer.device, dtype=outer.dtype)
    return inner + sample_displacement((outer - coordinates).permute(0, 3, 1, 2), inner)


def jacobian_determinant(grid: Tensor) -> Tensor:
    """Finite-difference determinant at cell centers, in normalized units."""
    if grid.ndim != 4 or grid.shape[-1] != 2 or min(grid.shape[1:3]) < 2:
        raise ValueError("grid must have shape [B,H>=2,W>=2,2]")
    field = grid.to(dtype=geometry_dtype(grid))
    height, width = field.shape[1:3]
    dx = ((field[:, :-1, 1:] - field[:, :-1, :-1])
          + (field[:, 1:, 1:] - field[:, 1:, :-1])) * ((width - 1) / 2)
    dy = ((field[:, 1:, :-1] - field[:, :-1, :-1])
          + (field[:, 1:, 1:] - field[:, :-1, 1:])) * ((height - 1) / 2)
    return dx[..., 0] * dy[..., 1] - dx[..., 1] * dy[..., 0]
