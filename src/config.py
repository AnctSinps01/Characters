from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Architecture choices that are still experimental in the design document."""

    image_size: int = 64
    patch_size: int = 8
    in_channels: int = 1
    hidden_size: int = 128
    num_heads: int = 8
    mlp_ratio: float = 4.0                  # FNN hidden dim
    local_depth: int = 4
    set_pool_depth: int = 2                 # NZ to M {CA -> SA -> FFN}
    generator_depth: int = 8
    description_tokens: int = 32
    style_tokens: int = 32
    time_frequency_size: int = hidden_size  # Time embed origin dim
    time_scale: float = 1_000.0             # 在生成正弦/余弦特征前，把 t∈[0,1] 放大
    sigma: float = 0.1

    def __post_init__(self) -> None:
        positive_ints = {
            "image_size": self.image_size,
            "patch_size": self.patch_size,
            "in_channels": self.in_channels,
            "hidden_size": self.hidden_size,
            "num_heads": self.num_heads,
            "set_pool_depth": self.set_pool_depth,
            "description_tokens": self.description_tokens,
            "style_tokens": self.style_tokens,
            "time_frequency_size": self.time_frequency_size,
        }
        for name, value in positive_ints.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.local_depth < 0 or self.generator_depth < 0:
            raise ValueError("local_depth and generator_depth cannot be negative")
        if self.image_size % self.patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.hidden_size % 4:
            raise ValueError("hidden_size must be divisible by 4 for 2D sin/cos positions")
        if self.time_frequency_size % 2:
            raise ValueError("time_frequency_size must be even")
        if self.mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be positive")
        if self.sigma < 0:
            raise ValueError("sigma cannot be negative")

    @property
    def grid_size(self) -> int:
        return self.image_size // self.patch_size

    @property
    def patch_count(self) -> int:
        return self.grid_size**2
