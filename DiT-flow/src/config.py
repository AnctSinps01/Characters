from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Pure-noise flow with independent content and reference-style encoders."""

    image_size: int = 64
    patch_size: int = 8
    in_channels: int = 1
    hidden_size: int = 256
    num_heads: int = 8
    mlp_ratio: float = 2.0                  # FNN hidden dim
    local_depth: int = 8                    # 单参考手写字 SA 层数
                                            # 当前设置只有固定一层 CA 风格浓缩。
    content_depth: int = 8                  # 目标标准字 SA 层数，独立调整
    generator_depth: int = 8                # 主干层数
    description_tokens: int = 16            # 单字描述 token 数
    time_frequency_size: int = hidden_size  # Time embed origin dim
    time_scale: float = 1_000.0             # 在生成正弦/余弦特征前，把 t∈[0,1] 放大

    def __post_init__(self) -> None:
        positive_ints = {
            "image_size": self.image_size,
            "patch_size": self.patch_size,
            "in_channels": self.in_channels,
            "hidden_size": self.hidden_size,
            "num_heads": self.num_heads,
            "description_tokens": self.description_tokens,
            "time_frequency_size": self.time_frequency_size,
        }
        for name, value in positive_ints.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if min(self.local_depth, self.content_depth, self.generator_depth) < 0:
            raise ValueError("local_depth, content_depth and generator_depth cannot be negative")
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

    @property
    def grid_size(self) -> int:
        return self.image_size // self.patch_size

    @property
    def patch_count(self) -> int:
        return self.grid_size**2


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Edit these defaults for a new run; resume restores saved settings."""

    steps: int = 1_000_000
    batch_size: int = 64
    min_references: int = 4              # 参考字数下限
    max_references: int = 4              # 参考字数上限
    learning_rate: float = 1e-4
    min_lr_ratio: float = 0.1
    warmup_steps: int = 1_000
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    precision: str = "bf16"
    preload_to_gpu: bool = False            # 放进显存 | 放进内存
    workers: int = 4
    val_batches: int = 16
    validate_every: int = 1_000
    save_every: int = 500
    log_every: int = 50
    sample_steps: int = 32                # t 从 0 到 1 采样次数
    seed: int = 42

    def __post_init__(self):
        if type(self.preload_to_gpu) is not bool:
            raise ValueError("preload_to_gpu must be a bool")
        for name in ("steps", "batch_size", "min_references",
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
        if not 0 <= self.min_lr_ratio <= 1:
            raise ValueError("invalid min_lr_ratio")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")
        if self.precision not in ("fp32", "fp16", "bf16"):
            raise ValueError("precision must be fp32, fp16 or bf16")
