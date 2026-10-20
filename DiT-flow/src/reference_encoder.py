from __future__ import annotations

import torch
from torch import Tensor, nn

from .config import ModelConfig
from .layers import CrossAttentionBlock, JointAttentionBlock, PatchEmbed2D


class ReferenceEncoder(nn.Module):
    """Jointly encode paired references; pool their positioned difference to Z tokens."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.handwriting_patch_embed = PatchEmbed2D(
            config.image_size, config.patch_size,
            config.in_channels, config.hidden_size,
        )
        self.standard_patch_embed = PatchEmbed2D(
            config.image_size, config.patch_size,
            config.in_channels, config.hidden_size,
        )
        self.type_embeddings = nn.Parameter(torch.empty(2, 1, config.hidden_size))
        self.local_blocks = nn.ModuleList(
            JointAttentionBlock(config.hidden_size, config.num_heads, config.mlp_ratio)
            for _ in range(config.local_depth)
        )
        self.description_queries = nn.Parameter(
            torch.empty(config.description_tokens, config.hidden_size)
        )
        self.local_pool = CrossAttentionBlock(
            config.hidden_size, config.num_heads, config.mlp_ratio
        )
        nn.init.normal_(self.type_embeddings, std=0.02)
        nn.init.normal_(self.description_queries, std=0.02)

    def forward(
        self,
        handwriting: Tensor,
        standard: Tensor,
        reference_mask: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            handwriting: [B, N, C, H, W] reference handwriting images.
            standard: matching standard glyphs, with exactly the same shape/order.
            reference_mask: optional [B, N] boolean mask; True means valid.
        Returns:
            Concatenated descriptions with shape [B, N*Z, D].
            Each pair interacts independently. After joint attention, subtract the
            standard stream from the handwriting stream and restore 2D positions.
            One CA pools each difference; queries/output get no position embedding.
        """
        if handwriting.ndim != 5:
            raise ValueError("handwriting must have shape [B, N, C, H, W]")
        if standard.shape != handwriting.shape:
            raise ValueError("reference standard glyphs must match handwriting shape and order")
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

        image_shape = (batch * reference_count, channels, height, width)
        hand_tokens = self.handwriting_patch_embed(handwriting.reshape(image_shape))
        standard_tokens = self.standard_patch_embed(standard.reshape(image_shape))
        hand_tokens = hand_tokens + self.type_embeddings[0].to(dtype=hand_tokens.dtype)
        standard_tokens = standard_tokens + self.type_embeddings[1].to(dtype=standard_tokens.dtype)
        for block in self.local_blocks:
            hand_tokens, standard_tokens = block(hand_tokens, standard_tokens)

        differences = hand_tokens - standard_tokens
        differences = differences + self.handwriting_patch_embed.positions.to(dtype=differences.dtype)
        description_queries = self.description_queries.unsqueeze(0).expand(
            batch * reference_count, -1, -1
        )
        descriptions = self.local_pool(description_queries, differences)
        return descriptions.reshape(
            batch,
            reference_count * self.config.description_tokens,
            self.config.hidden_size,
        )
