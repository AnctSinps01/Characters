"""Three losses: soft-balanced image MSE, source distance, and deformation."""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import torch
from torch import Tensor

from .config import LossConfig
from .geometry import sample_image
from .model import RegistrationResult


def foreground_weights(images: Tensor) -> Tensor:
    """Soft ink occupancy, without a learned mask or foreground tuning knob."""
    return ((1 - images.float().mean(dim=1, keepdim=True)) / 2).clamp(0, 1)


def soft_weighted_mse(prediction: Tensor, target: Tensor, epsilon: float = 1e-8) -> Tensor:
    """Balance foreground and background means separately within each image.

    Both use the target's soft occupancy rather than a hard sign threshold.
    Nonempty foreground/background groups receive equal automatic weight.
    Empty groups are excluded instead of halving an all-white/all-black loss.
    """
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("prediction and target must have equal shape [B,C,H,W]")
    squared = (prediction.float() - target.float()).square().mean(dim=1, keepdim=True)
    foreground = foreground_weights(target)
    weights = torch.stack((foreground, 1 - foreground), dim=1)
    mass = weights.sum(dim=(2, 3, 4))
    error = (weights * squared[:, None]).sum(dim=(2, 3, 4))
    present = (mass > epsilon).to(dtype=error.dtype)
    group_means = error / mass.clamp_min(epsilon)
    return ((group_means * present).sum(dim=1) / present.sum(dim=1).clamp_min(1)).mean()


def source_distance_loss(grid: Tensor, target: Tensor, source_distance: Tensor,
                         epsilon: float = 1e-8) -> Tensor:
    """B foreground q should sample original A close to A's foreground."""
    expected = (target.shape[0], 1, *target.shape[-2:])
    if source_distance.shape != expected or grid.shape != (*expected[:1], *expected[2:], 2):
        raise ValueError("distance must be [B,1,H,W] and grid [B,H,W,2] matching target")
    if target.device != grid.device or source_distance.device != grid.device:
        raise ValueError("target, source distance and grid must share a device")
    sampled = sample_image(source_distance.detach(), grid).float()
    foreground = foreground_weights(target)
    mass = foreground.sum(dim=(1, 2, 3))
    error = (foreground * sampled.square()).sum(dim=(1, 2, 3))
    return (error / mass.clamp_min(epsilon)).mean()


def bending_energy(displacement: Tensor) -> Tensor:
    """Discrete control-grid Hessian energy, with normalized grid spacing."""
    if displacement.ndim != 4 or displacement.shape[1] != 2:
        raise ValueError("displacement must have shape [B,2,Hc,Wc]")
    height, width = displacement.shape[-2:]
    if min(height, width) < 3:
        raise ValueError("bending requires at least three control nodes per dimension")
    field = displacement.float()
    center = field[:, :, 1:-1, 1:-1]
    dxx = (field[:, :, 1:-1, 2:] - 2 * center + field[:, :, 1:-1, :-2]) * (width - 1)**2
    dyy = (field[:, :, 2:, 1:-1] - 2 * center + field[:, :, :-2, 1:-1]) * (height - 1)**2
    dxy = (field[:, :, 2:, 2:] - field[:, :, 2:, :-2]
           - field[:, :, :-2, 2:] + field[:, :, :-2, :-2]) * ((width - 1) * (height - 1) / 4)
    return (dxx.square() + 2 * dxy.square() + dyy.square()).sum(dim=1).mean()


def deformation_components(displacements: Sequence[Tensor]) -> tuple[Tensor, Tensor]:
    if not displacements:
        raise ValueError("deformation regularization needs at least one displacement")
    bending = torch.stack([bending_energy(field) for field in displacements]).mean()
    movement = torch.stack([field.float().square().sum(dim=1).mean()
                            for field in displacements]).mean()
    return bending, movement


@dataclass(frozen=True)
class LossBreakdown:
    total: Tensor
    mse: Tensor
    distance: Tensor
    deformation: Tensor
    bending: Tensor
    displacement: Tensor


def registration_loss(result: RegistrationResult, target: Tensor,
                      source_distance: Tensor, config: LossConfig | None = None) -> LossBreakdown:
    """Train the fitted endpoint, not an independently predicted intermediate frame."""
    config = config or LossConfig()
    mse = soft_weighted_mse(result.image, target, config.epsilon)
    distance = source_distance_loss(result.grid, target, source_distance, config.epsilon)
    bending, movement = deformation_components(result.displacements)
    deformation = bending + config.displacement_ratio * movement
    total = (config.mse_weight * mse + config.distance_weight * distance
             + config.deformation_weight * deformation)
    return LossBreakdown(total, mse, distance, deformation, bending, movement)
