from __future__ import annotations

import torch
from torch import Tensor, nn

from .config import ModelConfig
from .layers import (
    CrossAttentionBlock,
    PatchEmbed2D,
    SelfAttentionBlock,
)


class ReferenceEncoder(nn.Module):
    """Encode each reference handwriting into Z tokens; concatenate, do not pool."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.patch_embed = PatchEmbed2D(
            config.image_size,
            config.patch_size,
            config.in_channels,
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
        nn.init.normal_(self.description_queries, std=0.02)

    def forward(
        self,
        handwriting: Tensor,
        reference_mask: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            handwriting: ``[B, N, C, H, W]`` reference handwriting images.
            reference_mask: optional ``[B, N]`` boolean mask; ``True`` means valid.
        Returns:
            Concatenated descriptions with shape ``[B, N*Z, D]``.
            No position embedding is added to the learned queries or output.
        """

        if handwriting.ndim != 5:
            raise ValueError("handwriting must have shape [B, N, C, H, W]")
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

        images = handwriting.reshape(
            batch * reference_count,
            channels,
            height,
            width,
        )
        local_tokens = self.patch_embed(images)
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

        return descriptions
