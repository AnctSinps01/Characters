"""Reference-to-latent model: joint attention over reference and standard latents."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    latent_channels: int = 4
    latent_size: int = 8
    references: int = 4
    dim: int = 384
    heads: int = 6
    layers: int = 6
    mlp_ratio: float = 4.0

    def __post_init__(self):
        for name in ("latent_channels", "latent_size", "references", "dim", "heads", "layers"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.dim % self.heads:
            raise ValueError("dim must be divisible by heads")
        if self.mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be positive")

    @property
    def latent_dim(self) -> int:
        """Numbers in one latent; one latent is one token."""
        return self.latent_channels * self.latent_size * self.latent_size
