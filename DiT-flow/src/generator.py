from __future__ import annotations

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
    """Predict pixel-space velocity from the current state, time, and style."""

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
        style: Tensor,
    ) -> Tensor:
        if state.ndim != 4 or state.shape[1] != self.config.in_channels:
            raise ValueError(
                f"state must have shape [B, {self.config.in_channels}, H, W]"
            )
        if state.shape[0] != times.shape[0] or state.shape[0] != style.shape[0]:
            raise ValueError("state, times, and style must have the same batch size")
        tokens = self.patch_embed(state)
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
