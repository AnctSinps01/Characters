"""全数据集同字手写配对训练。模型和 loss 配置位于 twist/config.py。"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class TrainingConfig:
    steps: int = 100_000          # 累计优化器更新次数，--stop-after 可先跑短实验。
    batch_size: int = 32
    learning_rate: float = 1e-4
    gradient_clip: float = 1.0
    log_every: int = 50
    validate_every: int = 1_000
    save_every: int = 1_000
    val_batches: int = 16         # 固定验证对数=val_batches*初始batch_size；0 为全部可配对验证源图。
    preview_samples: int = 4      # 固定预览的验证配对，尽量选不同字符。
    animation_frames: int = 21    # GIF 总帧数，包含 t=0 和 t=1。
    precision: str = "auto"      # CUDA 支持时用 bf16，否则 fp32；几何采样和 loss 保持 float32。
    preload_to_gpu: bool = False # 默认所有 uint8 字图驻内存；True 则所有字图驻训练显卡。
    distance_cache_size: int = 8192  # 源图距离场的内存缓存张数；64² 时上限约 128 MiB。
    seed: int = 42

    def __post_init__(self):
        for name in ("steps", "batch_size", "log_every", "validate_every", "save_every",
                     "preview_samples", "animation_frames", "distance_cache_size"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.animation_frames < 2:
            raise ValueError("animation_frames must be at least two")
        if type(self.val_batches) is not int or self.val_batches < 0:
            raise ValueError("val_batches must be an integer >= 0")
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise ValueError("seed must be an integer in [0, 2**63)")
        for name in ("learning_rate", "gradient_clip"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.precision not in ("auto", "fp32", "bf16"):
            raise ValueError("precision must be auto, fp32 or bf16")
        if type(self.preload_to_gpu) is not bool:
            raise ValueError("preload_to_gpu must be a bool")
