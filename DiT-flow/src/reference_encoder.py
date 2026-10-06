from __future__ import annotations

import torch
from torch import Tensor, nn

from .config import ModelConfig
from .layers import (
    CrossAttentionBlock,
    PatchEmbed2D,
    SelfAttentionBlock,
    SetAggregationBlock,
)


class ReferenceEncoder(nn.Module):
    """Encode a variable, unordered set of paired handwriting/standard glyphs."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.patch_embed = PatchEmbed2D(
            config.image_size,
            config.patch_size,
            2 * config.in_channels,
            config.hidden_size,
        )
        self.local_blocks = nn.ModuleList(
            SelfAttentionBlock(config.hidden_size, config.num_heads, config.mlp_ratio)
            for _ in range(config.local_depth)
        )
        self.description_queries = nn.Parameter(
            torch.empty(config.description_tokens, config.hidden_size)
        )
        self.local_pool = CrossAttentionBlock(
            config.hidden_size, config.num_heads, config.mlp_ratio
        )
        self.style_queries = nn.Parameter(
            torch.empty(config.style_tokens, config.hidden_size)
        )
        self.set_pool_blocks = nn.ModuleList(
            SetAggregationBlock(
                config.hidden_size, config.num_heads, config.mlp_ratio
            )
            for _ in range(config.set_pool_depth)
        )
        nn.init.normal_(self.description_queries, std=0.02)
        nn.init.normal_(self.style_queries, std=0.02)

    def forward(
        self,
        handwriting: Tensor,
        standard: Tensor,
        reference_mask: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            handwriting: ``[B, N, C, H, W]`` reference handwriting images.
            standard: ``[B, N, C, H, W]`` matched standard glyph images.
            reference_mask: optional ``[B, N]`` boolean mask; ``True`` means valid.
        Returns:
            Style tokens with shape ``[B, M, D]``.
        """

        if handwriting.shape != standard.shape or handwriting.ndim != 5:
            raise ValueError(
                "handwriting and standard must share shape [B, N, C, H, W]"
            )
        batch, reference_count, channels, height, width = handwriting.shape
        expected_image_shape = (
            self.config.in_channels,
            self.config.image_size,
            self.config.image_size,
        )
        if (channels, height, width) != expected_image_shape:
            raise ValueError(
                f"expected reference image shape {expected_image_shape}, "
                f"got {(channels, height, width)}"
            )
        if reference_count < 1:
            raise ValueError("each batch item needs at least one reference slot")

        if reference_mask is None:
            reference_mask = torch.ones(
                batch, reference_count, dtype=torch.bool, device=handwriting.device
            )
        if reference_mask.shape != (batch, reference_count):
            raise ValueError(
                f"reference_mask must have shape {(batch, reference_count)}, "
                f"got {reference_mask.shape}"
            )
        reference_mask = reference_mask.to(device=handwriting.device, dtype=torch.bool)
        if (~reference_mask.any(dim=1)).any():
            raise ValueError("each batch item must contain at least one valid reference")

        pairs = torch.cat((handwriting, standard), dim=2).reshape(
            batch * reference_count,
            2 * channels,
            height,
            width,
        )
        local_tokens = self.patch_embed(pairs)
        for block in self.local_blocks:
            local_tokens = block(local_tokens)

        description_queries = self.description_queries.unsqueeze(0).expand(
            batch * reference_count, -1, -1
        )
        descriptions = self.local_pool(description_queries, local_tokens)
        descriptions = descriptions.reshape(
            batch,
            reference_count * self.config.description_tokens,
            self.config.hidden_size,
        )

        valid_descriptions = reference_mask.unsqueeze(-1).expand(
            -1, -1, self.config.description_tokens
        )
        memory_padding_mask = ~valid_descriptions.reshape(batch, -1)
        style = self.style_queries.unsqueeze(0).expand(batch, -1, -1)
        for block in self.set_pool_blocks:
            style = block(
                style,
                descriptions,
                memory_padding_mask=memory_padding_mask,
            )
        return style
