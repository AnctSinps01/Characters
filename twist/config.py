"""Stationary-velocity registration and loss configuration."""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True, slots=True)
class ModelConfig:
    image_size: int = 64           # 正方形输入图的边长，必须能被 patch_size 整除。
    patch_size: int = 8            # 每个 patch 的边长；每张图有 (image_size / patch_size)² 个 token。
    in_channels: int = 1           # 输入通道数，灰度线条图为 1。
    token_dim: int = 256           # token 特征维度，必须同时能被 4 和 num_heads 整除。
    num_heads: int = 8             # 自注意力与交叉注意力的头数。
    num_blocks: int = 8            # 单个配准网络内堆叠的 SA / CA / FFN 双流块数。
    mlp_ratio: float = 4.0         # FFN 隐藏层宽度为 int(token_dim * mlp_ratio)。
    squaring_steps: int = 7        # 先除以 2**n，再自复合 n 次，等效 2**n 个小步。
    head_init_std: float = 1e-3    # 速度输出头权重的初始标准差。

    def __post_init__(self) -> None:
        for name in ("image_size", "patch_size", "in_channels", "token_dim",
                     "num_heads", "num_blocks"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.image_size < 2 or self.image_size % self.patch_size:
            raise ValueError("image_size must be at least 2 and divisible by patch_size")
        if self.token_dim % 4 or self.token_dim % self.num_heads:
            raise ValueError("token_dim must be divisible by 4 and num_heads")
        if not math.isfinite(self.mlp_ratio) or int(self.token_dim * self.mlp_ratio) < 1:
            raise ValueError("mlp_ratio must produce a positive FFN width")
        if type(self.squaring_steps) is not int or self.squaring_steps < 0:
            raise ValueError("squaring_steps must be a nonnegative integer")
        if not math.isfinite(self.head_init_std) or self.head_init_std < 0:
            raise ValueError("head_init_std must be finite and nonnegative")

    @property
    def patch_grid_size(self) -> int:
        """每个方向上的 patch 数量。"""
        return self.image_size // self.patch_size


@dataclass(frozen=True, slots=True)
class LossConfig:
    # 有且只有三项：双向图像相似度、双向距离场、未缩放速度的一阶平滑。
    similarity_weight: float = 1.0  # 两个方向的梯度加权 MSE 相加。
    mse_uniform_ratio: float = 0.2   # 各方向 MSE 的全图误差占比，其余为参考图梯度加权误差。
    distance_weight: float = 1.0     # 双向距离场 Chamfer，相加而非取平均。
    smoothness_weight: float = 1e-3  # 一阶速度梯度平方，按原图归一化像素间距计算。
    distance_threshold: float = 0.0  # 距离场将 <= 此值视为前景；输入白色 +1、黑色 -1。
    epsilon: float = 1e-8            # 归一化稳定常数；距离场的前景质量分母加 epsilon。

    def __post_init__(self) -> None:
        for name in ("similarity_weight", "distance_weight", "smoothness_weight"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.similarity_weight + self.distance_weight + self.smoothness_weight == 0:
            raise ValueError("at least one top-level loss weight must be positive")
        if not math.isfinite(self.mse_uniform_ratio) or not 0 <= self.mse_uniform_ratio <= 1:
            raise ValueError("mse_uniform_ratio must be in [0, 1]")
        if not math.isfinite(self.distance_threshold) or not -1 <= self.distance_threshold < 1:
            raise ValueError("distance_threshold must be in [-1, 1)")
        if not math.isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("epsilon must be finite and positive")
