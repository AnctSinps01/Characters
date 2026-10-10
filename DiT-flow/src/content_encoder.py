from __future__ import annotations

from torch import Tensor, nn

from .config import ModelConfig
from .layers import PatchEmbed2D, SelfAttentionBlock


class ContentEncoder(nn.Module):
    """The reference encoder's patch/SA path, without the final CA compression."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.patch_embed = PatchEmbed2D(
            config.image_size, config.patch_size,
            config.in_channels, config.hidden_size,
        )
        self.local_blocks = nn.ModuleList(
            SelfAttentionBlock(config.hidden_size, config.num_heads, config.mlp_ratio)
            for _ in range(config.content_depth)
        )

    def forward(self, standard: Tensor) -> Tensor:
        if standard.ndim != 4 or standard.shape[1] != self.config.in_channels:
            raise ValueError(
                f"standard must have shape [B, {self.config.in_channels}, H, W]"
            )
        tokens = self.patch_embed(standard)
        for block in self.local_blocks:
            tokens = block(tokens)
        return tokens
