from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Architecture choices that are still experimental in the design document."""

    image_size: int = 64
    patch_size: int = 8
    in_channels: int = 1
    hidden_size: int = 256
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


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Edit these defaults for a new run; resume restores saved settings."""

    steps: int = 100_000
    batch_size: int = 64
    accumulation_steps: int = 1
    min_references: int = 1               # Inclusive lower bound
    max_references: int = 16              # Sample reference count uniformly up to this
    learning_rate: float = 1e-4
    min_lr_ratio: float = 0.1
    warmup_steps: int = 1_000
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    ema_decay: float = 0.999
    author_style_loss_weight: float = 0.05  # Same-author/orthogonal-author token loss
    precision: str = "bf16"
    preload_to_gpu: bool = False          # Keep prepared uint8 glyphs in VRAM
    workers: int = 8
    val_batches: int = 16
    validate_every: int = 1_000
    save_every: int = 500
    log_every: int = 50
    sample_steps: int = 32                # t 从 0 到 1 采样次数
    seed: int = 42

    def __post_init__(self):
        if type(self.preload_to_gpu) is not bool:
            raise ValueError("preload_to_gpu must be a bool")
        for name in ("steps", "batch_size", "accumulation_steps", "min_references",
                     "max_references", "val_batches", "validate_every", "save_every",
                     "log_every", "sample_steps"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not 0 <= self.warmup_steps < self.steps or self.workers < 0:
            raise ValueError("expected 0 <= warmup_steps < steps and workers >= 0")
        if self.min_references > self.max_references:
            raise ValueError("min_references cannot exceed max_references")
        for name in ("learning_rate", "gradient_clip"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if (not math.isfinite(self.author_style_loss_weight)
                or self.author_style_loss_weight < 0):
            raise ValueError("author_style_loss_weight must be finite and nonnegative")
        if not 0 <= self.min_lr_ratio <= 1 or not 0 <= self.ema_decay < 1:
            raise ValueError("invalid min_lr_ratio or ema_decay")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")
        if self.precision not in ("fp32", "fp16", "bf16"):
            raise ValueError("precision must be fp32, fp16 or bf16")
