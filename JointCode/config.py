"""Decode a concatenated (content code, author code) pair back to an image."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DecoderConfig:
    latent_channels: int = 4      # channels per code; two codes are concatenated
    latent_size: int = 8
    base_channels: int = 64

    def __post_init__(self):
        for name in ("latent_channels", "latent_size", "base_channels"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.base_channels % 8:
            raise ValueError("base_channels must be divisible by 8 for GroupNorm")

    @property
    def in_channels(self) -> int:
        return self.latent_channels * 2
