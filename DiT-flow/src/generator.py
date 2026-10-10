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
    """Predict velocity using SA -> Content CA -> Style CA -> FFN blocks."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.patch_embed = PatchEmbed2D(
            config.image_size,
            config.patch_size,
            config.in_channels,
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
        content: Tensor,
        style: Tensor,
        style_mask: Tensor | None = None,
    ) -> Tensor:
        if state.ndim != 4 or state.shape[1] != self.config.in_channels:
            raise ValueError(
                f"state must have shape [B, {self.config.in_channels}, H, W]"
            )
        batch = state.shape[0]
        if times.shape != (batch,):
            raise ValueError("times must have shape [B]")
        if content.shape != (batch, self.config.patch_count, self.config.hidden_size):
            raise ValueError("content must have shape [B, L, D]")
        if (style.ndim != 3 or style.shape[0] != batch
                or style.shape[1] < 1 or style.shape[2] != self.config.hidden_size):
            raise ValueError("style must have shape [B, N*Z, D]")
        padding_mask = None
        if style_mask is not None:
            if style_mask.shape != style.shape[:2]:
                raise ValueError("style_mask must have shape [B, N*Z]")
            style_mask = style_mask.to(device=style.device, dtype=torch.bool)
            if (~style_mask.any(dim=1)).any():
                raise ValueError("each sample needs at least one valid style token")
            padding_mask = ~style_mask
        tokens = self.patch_embed(state)
        time_embedding = self.time_embed(times).to(dtype=tokens.dtype)
        for block in self.blocks:
            tokens = block(tokens, content, style, time_embedding, padding_mask)
        velocity_patches = self.final(tokens, time_embedding)
        return unpatchify(
            velocity_patches,
            self.config.grid_size,
            self.config.patch_size,
            self.config.in_channels,
        )
