"""Architecture and loss settings for the first twist implementation."""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True, slots=True)
class ModelConfig:
    image_size: int = 64
    patch_size: int = 8
    in_channels: int = 1
    token_dim: int = 256
    num_heads: int = 8
    blocks_per_stage: int = 2       # Dual-stream attention blocks within one stage.
    num_stages: int = 4             # Sequential registration stages.
    mlp_ratio: float = 4.0
    lipschitz_limit: float = 0.8
    share_stage_weights: bool = False
    head_init_std: float = 1e-3     # Small initial movement; gradients reach both streams.

    def __post_init__(self) -> None:
        for name in ("image_size", "patch_size", "in_channels", "token_dim",
                     "num_heads", "blocks_per_stage", "num_stages"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.image_size < 2 or self.image_size % self.patch_size:
            raise ValueError("image_size must be at least 2 and divisible by patch_size")
        if self.token_dim % 4 or self.token_dim % self.num_heads:
            raise ValueError("token_dim must be divisible by 4 and num_heads")
        if not math.isfinite(self.mlp_ratio) or int(self.token_dim * self.mlp_ratio) < 1:
            raise ValueError("mlp_ratio must produce a positive FFN width")
        if not math.isfinite(self.lipschitz_limit) or not 0 < self.lipschitz_limit < 1:
            raise ValueError("lipschitz_limit must be strictly between 0 and 1")
        if type(self.share_stage_weights) is not bool:
            raise ValueError("share_stage_weights must be a bool")
        if not math.isfinite(self.head_init_std) or self.head_init_std < 0:
            raise ValueError("head_init_std must be finite and nonnegative")

    @property
    def patch_grid_size(self) -> int:
        return self.image_size // self.patch_size

    @property
    def control_grid_size(self) -> int:
        # One fixed zero ring around all predicted patch displacements.
        return self.patch_grid_size + 2


@dataclass(frozen=True, slots=True)
class LossConfig:
    # The three top-level loss proportions. Defaults need experimental tuning.
    mse_weight: float = 1.0
    distance_weight: float = 1.0
    deformation_weight: float = 1e-4
    # Inside deformation = bending + displacement_ratio * displacement.
    displacement_ratio: float = 0.01
    distance_threshold: float = 0.0  # Input convention: white=+1, black=-1.
    epsilon: float = 1e-8

    def __post_init__(self) -> None:
        for name in ("mse_weight", "distance_weight", "deformation_weight",
                     "displacement_ratio"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.mse_weight + self.distance_weight + self.deformation_weight == 0:
            raise ValueError("at least one top-level loss weight must be positive")
        if not math.isfinite(self.distance_threshold) or not -1 <= self.distance_threshold < 1:
            raise ValueError("distance_threshold must be in [-1, 1)")
        if not math.isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("epsilon must be finite and positive")
