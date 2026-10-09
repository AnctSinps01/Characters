"""Exactly three objectives: bidirectional similarity, distance, and velocity smoothness."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import Tensor

from .config import LossConfig
from .geometry import sample_image
from .model import RegistrationResult


def foreground_weights(images: Tensor) -> Tensor:
    """Detached soft foreground: black=1, white=0, gray retains ink coverage."""
    return ((1 - images.detach().float().mean(dim=1, keepdim=True)) / 2).clamp(0, 1)


def gradient_weighted_mse(prediction: Tensor, target: Tensor,
                          uniform_ratio: float = 0.1, epsilon: float = 1e-8) -> Tensor:
    """Emphasize the reference image's edges and retain uniform image error.

    Sobel gradients use replicate padding so the canvas boundary does not
    create an artificial edge. Normalize gradient energy independently for
    each image; constant targets fall back to ordinary MSE. Weights are
    detached target statistics, rather than gradients of the predicted image.
    """
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("prediction and target must have equal shape [B,C,H,W]")
    if not 0 <= uniform_ratio <= 1 or epsilon <= 0:
        raise ValueError("uniform_ratio must be in [0,1] and epsilon must be positive")
    squared = (prediction.float() - target.float()).square().mean(dim=1, keepdim=True)
    # 在混合精度上下文内也用 FP32 计算弱边缘的权重。
    with torch.autocast(device_type=target.device.type, enabled=False):
        gray = target.detach().float().mean(dim=1, keepdim=True)
        kernels = gray.new_tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                   [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]])[:, None] / 8
        gradients = F.conv2d(F.pad(gray, (1, 1, 1, 1), mode="replicate"), kernels)
        weights = gradients.square().sum(dim=1, keepdim=True)
    mass = weights.sum(dim=(1, 2, 3))
    plain = squared.mean(dim=(1, 2, 3))
    edge = (weights * squared).sum(dim=(1, 2, 3)) / mass.clamp_min(epsilon)
    edge = torch.where(mass > epsilon, edge, plain)
    return (uniform_ratio * plain + (1 - uniform_ratio) * edge).mean()


def source_distance_loss(grid: Tensor, target: Tensor, source_distance: Tensor,
                         epsilon: float = 1e-8) -> Tensor:
    """B foreground q should sample original A close to A's foreground."""
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    expected = (target.shape[0], 1, *target.shape[-2:])
    if source_distance.shape != expected or grid.shape != (*expected[:1], *expected[2:], 2):
        raise ValueError("distance must be [B,1,H,W] and grid [B,H,W,2] matching target")
    if target.device != grid.device or source_distance.device != grid.device:
        raise ValueError("target, source distance and grid must share a device")
    sampled = sample_image(source_distance.detach(), grid).float()
    foreground = foreground_weights(target)
    mass = foreground.sum(dim=(1, 2, 3))
    error = (foreground * sampled.square()).sum(dim=(1, 2, 3))
    return (error / (mass + epsilon)).mean()


def velocity_smoothness(velocity: Tensor) -> Tensor:
    """Viscous H1 energy: ||dv/dx||^2 + ||dv/dy||^2, with normalized spacing.

    Penalize first derivatives mildly; no hard incompressibility, bending,
    or velocity-magnitude penalty is imposed.
    """
    if velocity.ndim != 4 or velocity.shape[1] != 2 or min(velocity.shape[-2:]) < 2:
        raise ValueError("velocity must have shape [B,2,H>=2,W>=2]")
    field = velocity.float()
    height, width = field.shape[-2:]
    dx = (field[..., 1:] - field[..., :-1]) * (width - 1)
    dy = (field[..., 1:, :] - field[..., :-1, :]) * (height - 1)
    return dx.square().sum(dim=1).mean() + dy.square().sum(dim=1).mean()


@dataclass(frozen=True)
class LossBreakdown:
    total: Tensor
    similarity: Tensor
    distance: Tensor
    smoothness: Tensor
    forward_similarity: Tensor
    reverse_similarity: Tensor
    forward_distance: Tensor
    reverse_distance: Tensor


def registration_loss(result: RegistrationResult, source_distance: Tensor,
                      target_distance: Tensor, config: LossConfig | None = None) -> LossBreakdown:
    """Sum the two directions for similarity and Chamfer; smooth unscaled v.

    Both original-image distance fields must be precomputed. Foreground
    weights and distance fields are detached; gradients pass through sampling
    maps to the velocity network.
    """
    config = config or LossConfig()
    forward_similarity = gradient_weighted_mse(
        result.image, result.target, config.mse_uniform_ratio, config.epsilon)
    reverse_similarity = gradient_weighted_mse(
        result.reverse_image, result.source, config.mse_uniform_ratio, config.epsilon)
    similarity = forward_similarity + reverse_similarity
    forward_distance = source_distance_loss(
        result.grid, result.target, source_distance, config.epsilon)
    reverse_distance = source_distance_loss(
        result.reverse_grid, result.source, target_distance, config.epsilon)
    distance = forward_distance + reverse_distance
    smoothness = velocity_smoothness(result.velocity)
    total = (config.similarity_weight * similarity + config.distance_weight * distance
             + config.smoothness_weight * smoothness)
    return LossBreakdown(total, similarity, distance, smoothness, forward_similarity,
                         reverse_similarity, forward_distance, reverse_distance)
