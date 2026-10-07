"""单字实验设置；模型结构和三类 loss 比例仍在 twist/config.py。"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class TrainingConfig:
    character: str = "中"
    mode: str = "handwriting"      # handwriting：全部同字手写样本；synthetic：标准字的可控形变。
    batch_size: int = 16           # 样本数不足时自动减小；--writer 单对测试时为 1。
    steps: int = 1_000             # 优化器更新次数；续训时也是累计目标步数。
    learning_rate: float = 3e-4
    gradient_clip: float = 1.0
    log_every: int = 10
    preview_every: int = 100       # 同时保存检查点和轨迹 GIF。
    seed: int = 42
    precision: str = "fp32"       # 首次验证用 fp32；CUDA 支持时可以改 bf16。
    synthetic_displacement: float = 0.06  # 合成位移模长上限，归一化坐标单位。
    synthetic_lipschitz: float = 0.5
    animation_frames: int = 21    # 保存 GIF 的总帧数，包含 t=0 和 t=1。

    def __post_init__(self):
        if len(self.character) != 1:
            raise ValueError("character must be exactly one character")
        if self.mode not in ("synthetic", "handwriting"):
            raise ValueError("mode must be synthetic or handwriting")
        for name in ("steps", "batch_size", "log_every", "preview_every", "animation_frames"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.animation_frames < 2:
            raise ValueError("animation_frames must be at least two")
        for name in ("learning_rate", "gradient_clip", "synthetic_displacement"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0 < self.synthetic_lipschitz < 1:
            raise ValueError("synthetic_lipschitz must be between zero and one")
        if self.precision not in ("fp32", "bf16"):
            raise ValueError("precision must be fp32 or bf16")
