"""twist 首版的模型结构与损失配置。"""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True, slots=True)
class ModelConfig:
    image_size: int = 64           # 正方形输入图的边长，必须能被 patch_size 整除。
    patch_size: int = 8            # 每个 patch 的边长；每张图有 (image_size / patch_size)² 个 token。
    in_channels: int = 1           # 输入通道数，灰度线条图为 1。
    token_dim: int = 128           # token 特征维度，必须同时能被 4 和 num_heads 整除。
    num_heads: int = 8             # 自注意力与交叉注意力的头数。
    blocks_per_stage: int = 4      # 单个配准阶段内堆叠的双流注意力块数。
    num_stages: int = 8            # 串联配准阶段数；每阶段读取前一阶段变形后的 A。
    mlp_ratio: float = 2.0         # FFN 隐藏层宽度为 int(token_dim * mlp_ratio)。
    lipschitz_limit: float = 0.8   # 每阶段位移场的 Lipschitz 上界，须在 (0,1) 内以保证不折叠。
    share_stage_weights: bool = False  # True：各阶段共用参数；False：各阶段参数独立。
    head_init_std: float = 1e-3    # 位移输出头权重的初始标准差，从小变形开始。

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
        """每个方向上的 patch 数量。"""
        return self.image_size // self.patch_size

    @property
    def control_grid_size(self) -> int:
        # 预测的 patch 位移外增加一圈固定零节点，保持图像边界不动。
        return self.patch_grid_size + 2


@dataclass(frozen=True, slots=True)
class LossConfig:
    # 总损失为以下三项的加权和；默认比例是实验起点，可在此调整。
    mse_weight: float = 1.0        # 梯度平方加权 MSE 的系数。
    mse_uniform_ratio: float = 0.2  # MSE 内普通全图误差的占比，其余为 B 的梯度平方加权误差。
    distance_weight: float = 0.0   # 距离场匹配损失的系数，距离使用归一化坐标单位。
    deformation_weight: float = 1e-4  # 形变正则的系数。
    # 形变正则 = 弯曲损失 + displacement_ratio * 位移平方损失。
    displacement_ratio: float = 0.01  # 形变正则内部的小位移惩罚比例。
    distance_threshold: float = 0.0  # 距离场将 <= 此值视为前景；输入白色 +1、黑色 -1。
    epsilon: float = 1e-8          # 损失归一化分母的下限，避免除零。

    def __post_init__(self) -> None:
        for name in ("mse_weight", "distance_weight", "deformation_weight",
                     "displacement_ratio"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.mse_weight + self.distance_weight + self.deformation_weight == 0:
            raise ValueError("at least one top-level loss weight must be positive")
        if not math.isfinite(self.mse_uniform_ratio) or not 0 <= self.mse_uniform_ratio <= 1:
            raise ValueError("mse_uniform_ratio must be in [0, 1]")
        if not math.isfinite(self.distance_threshold) or not -1 <= self.distance_threshold < 1:
            raise ValueError("distance_threshold must be in [-1, 1)")
        if not math.isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("epsilon must be finite and positive")
