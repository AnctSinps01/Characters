from __future__ import annotations

import torch
from torch import Tensor, nn

from .config import ModelConfig
from .layers import (
    AdaLNZeroDiTBlock,
    ContinuousTimeEmbedding,
    FinalVelocityLayer,
    PatchEmbed2D,
    unpatchify,
)


class ConditionalDiT(nn.Module):
    """Predict a pixel-space velocity conditioned on the target glyph and style."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.patch_embed = PatchEmbed2D(
            config.image_size,
            config.patch_size,
            2 * config.in_channels,
            config.hidden_size,
        )
        self.time_embed = ContinuousTimeEmbedding(
            config.hidden_size,
            config.time_frequency_size,
            config.time_scale,
        )
        self.blocks = nn.ModuleList(
            AdaLNZeroDiTBlock(config.hidden_size, config.num_heads, config.mlp_ratio)
            for _ in range(config.generator_depth)
        )
        self.final = FinalVelocityLayer(
            config.hidden_size, config.patch_size, config.in_channels
        )

    def forward(
        self,
        state: Tensor,
        times: Tensor,
        target_standard: Tensor,
        style: Tensor,
    ) -> Tensor:
        if state.shape != target_standard.shape or state.ndim != 4:
            raise ValueError("state and target_standard must share shape [B, C, H, W]")
        if state.shape[0] != times.shape[0] or state.shape[0] != style.shape[0]:
            raise ValueError("state, times, and style must have the same batch size")
        tokens = self.patch_embed(torch.cat((state, target_standard), dim=1))
        time_embedding = self.time_embed(times).to(dtype=tokens.dtype)
        for block in self.blocks:
            tokens = block(tokens, style, time_embedding)
        velocity_patches = self.final(tokens, time_embedding)
        return unpatchify(
            velocity_patches,
            self.config.grid_size,
            self.config.patch_size,
            self.config.in_channels,
        )
