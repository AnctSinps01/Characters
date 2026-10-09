"""A single velocity network decoding one 2D vector per original image pixel."""
from __future__ import annotations

from torch import Tensor, nn

from .config import ModelConfig
from .geometry import zero_boundary
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
            DualStreamBlock(self.config) for _ in range(self.config.num_blocks))
        self.output_norm = nn.LayerNorm(self.config.token_dim)
        self.velocity_head = nn.Linear(self.config.token_dim, 2 * self.config.patch_size**2)
        nn.init.normal_(self.velocity_head.weight, std=self.config.head_init_std)
        nn.init.zeros_(self.velocity_head.bias)

    def forward(self, source: Tensor, target: Tensor) -> Tensor:
        validate_pair(source, target, self.config)
        source_tokens = self.patch_embedding(source)
        target_tokens = self.patch_embedding(target)
        for block in self.blocks:
            source_tokens, target_tokens = block(source_tokens, target_tokens)
        # Each token decodes its patch_size**2 independent (vx, vy) vectors.
        vectors = self.velocity_head(self.output_norm(target_tokens))
        batch, grid, patch = vectors.shape[0], self.config.patch_grid_size, self.config.patch_size
        patches = vectors.reshape(batch, grid, grid, patch, patch, 2)
        # [B,patch_y,patch_x,pixel_y,pixel_x,xy] -> [B,xy,image_y,image_x].
        velocity = patches.permute(0, 5, 1, 3, 2, 4).reshape(
            batch, 2, self.config.image_size, self.config.image_size)
        return zero_boundary(velocity)
