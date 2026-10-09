"""Predict one stationary velocity and render a bidirectional morph."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .config import ModelConfig
from .geometry import identity_grid, sample_image, time_scale, velocity_grid
from .stage import RegistrationStage, validate_pair


@dataclass(frozen=True)
class WarpResult:
    image: Tensor
    grid: Tensor                 # A backward map exp(t*v).
    reverse_grid: Tensor         # B backward map exp(-(1-t)*v).
    source_image: Tensor
    target_image: Tensor


@dataclass(frozen=True)
class RegistrationResult:
    source: Tensor
    target: Tensor
    velocity: Tensor             # [B,2,H,W], same resolution as source; normalized coordinate units.
    image: Tensor                # Unblended A(exp(v)), for endpoint supervision.
    grid: Tensor
    reverse_image: Tensor        # Unblended B(exp(-v)).
    reverse_grid: Tensor
    squaring_steps: int

    def render(self, time: float | Tensor) -> WarpResult:
        """(1-t)*A(exp(t*v)) + t*B(exp(-(1-t)*v)), without a network rerun."""
        strength = time_scale(time, self.grid, minimum=0.0, maximum=1.0)
        # Scalar endpoint requests reuse already integrated maps and samples.
        # Tensor times keep the differentiable path, including d(image)/dt.
        if isinstance(time, (int, float)) and time in (0, 1):
            identity = identity_grid(self.source.shape[0], self.source.shape[-2:],
                                     device=self.grid.device, dtype=self.grid.dtype)
            if time == 0:
                original = self.source.to(self.grid.dtype)
                return WarpResult(original, identity, self.reverse_grid, original, self.reverse_image)
            original = self.target.to(self.grid.dtype)
            return WarpResult(original, self.grid, identity, self.image, original)
        # Preserve Python scalar times so internal validation cannot synchronize
        # GPU scalars on every preview frame.
        scaled_time = time if isinstance(time, Tensor) else float(time)
        grid = velocity_grid(self.velocity, self.source, scaled_time,
                             squaring_steps=self.squaring_steps)
        reverse_grid = velocity_grid(self.velocity, self.target, scaled_time - 1,
                                     squaring_steps=self.squaring_steps)
        source_image = sample_image(self.source, grid)
        target_image = sample_image(self.target, reverse_grid)
        # Preserve original pixels exactly at the trajectory endpoints.
        source_image = torch.where(strength == 0, self.source.to(source_image.dtype), source_image)
        target_image = torch.where(strength == 1, self.target.to(target_image.dtype), target_image)
        image = (1 - strength) * source_image + strength * target_image
        return WarpResult(image, grid, reverse_grid, source_image, target_image)


class Twist(nn.Module):
    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        self.network = RegistrationStage(self.config)

    def forward(self, source: Tensor, target: Tensor) -> RegistrationResult:
        """Encode A/B once and decode v once; integrate exp(v) and exp(-v)."""
        validate_pair(source, target, self.config)
        velocity = self.network(source, target)
        grid = velocity_grid(velocity, source, squaring_steps=self.config.squaring_steps)
        reverse_grid = velocity_grid(velocity, target, -1.0,
                                     squaring_steps=self.config.squaring_steps)
        return RegistrationResult(source, target, velocity, sample_image(source, grid), grid,
                                  sample_image(target, reverse_grid), reverse_grid,
                                  self.config.squaring_steps)
