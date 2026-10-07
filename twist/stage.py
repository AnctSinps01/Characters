"""A single registration stage: current A and fixed B -> constrained u."""
from __future__ import annotations

from torch import Tensor, nn
import torch.nn.functional as F

from .config import ModelConfig
from .geometry import constrain_displacement
from .layers import DualStreamBlock, PatchEmbedding


def validate_pair(source: Tensor, target: Tensor, config: ModelConfig) -> None:
    if source.ndim != 4 or target.shape != source.shape:
        raise ValueError("A and B must have equal shape [batch, channels, height, width]")
    if source.shape[0] < 1 or source.shape[1] != config.in_channels:
        raise ValueError("batch must be nonempty and channels must match config.in_channels")
    if source.shape[-2:] != (config.image_size, config.image_size):
        raise ValueError(f"expected {config.image_size}x{config.image_size} images")
    if source.device != target.device or source.dtype != target.dtype:
        raise ValueError("A and B must have the same device and dtype")
    if not source.is_floating_point():
        raise ValueError("A and B must be floating-point images normalized to [-1, 1]")


class RegistrationStage(nn.Module):
    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        self.patch_embedding = PatchEmbedding(self.config)  # Shared by A and B.
        self.blocks = nn.ModuleList(
            DualStreamBlock(self.config) for _ in range(self.config.blocks_per_stage))
        self.output_norm = nn.LayerNorm(self.config.token_dim)
        self.displacement_head = nn.Linear(self.config.token_dim, 2)
        nn.init.normal_(self.displacement_head.weight, std=self.config.head_init_std)
        nn.init.zeros_(self.displacement_head.bias)

    def forward(self, current_source: Tensor, target: Tensor) -> Tensor:
        validate_pair(current_source, target, self.config)
        source_tokens = self.patch_embedding(current_source)
        target_tokens = self.patch_embedding(target)
        for block in self.blocks:
            source_tokens, target_tokens = block(source_tokens, target_tokens)
        # B's updated query tokens decode the output-frame sampling displacement.
        vectors = self.displacement_head(self.output_norm(target_tokens))
        size = self.config.patch_grid_size
        interior = vectors.transpose(1, 2).reshape(vectors.shape[0], 2, size, size)
        controls = F.pad(interior, (1, 1, 1, 1))
        return constrain_displacement(controls, self.config.lipschitz_limit)
