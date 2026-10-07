"""Two self-attention streams and one B-query cross-attention per block."""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .config import ModelConfig


def position_embedding(grid_size: int, width: int) -> Tensor:
    """Fixed 2D sinusoidal patch positions; this is not a time embedding."""
    rows, columns = torch.meshgrid(
        torch.arange(grid_size, dtype=torch.float32),
        torch.arange(grid_size, dtype=torch.float32), indexing="ij")
    frequencies = torch.exp(
        -math.log(10_000.0) * torch.arange(width // 4) / (width // 4))
    row_angles = rows.reshape(-1, 1) * frequencies
    column_angles = columns.reshape(-1, 1) * frequencies
    return torch.cat((row_angles.sin(), row_angles.cos(),
                      column_angles.sin(), column_angles.cos()), dim=-1).unsqueeze(0)


class PatchEmbedding(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.projection = nn.Conv2d(
            config.in_channels, config.token_dim,
            kernel_size=config.patch_size, stride=config.patch_size)
        self.register_buffer(
            "positions", position_embedding(config.patch_grid_size, config.token_dim),
            persistent=False)

    def forward(self, images: Tensor) -> Tensor:
        tokens = self.projection(images).flatten(2).transpose(1, 2)
        return tokens + self.positions.to(dtype=tokens.dtype)


class FeedForward(nn.Module):
    def __init__(self, width: int, ratio: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(width, int(width * ratio)), nn.GELU(),
            nn.Linear(int(width * ratio), width))

    def forward(self, tokens: Tensor) -> Tensor:
        return self.layers(tokens)


class DualStreamBlock(nn.Module):
    """A: SA -> FFN; B: SA -> CA(read A) -> FFN, all with residuals.

    A's FFN precedes the cross-attention so even the last A FFN contributes
    to the decoded B tokens. Cross-attention writes only the B query stream.
    """
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        width, heads = config.token_dim, config.num_heads
        self.source_self_norm = nn.LayerNorm(width)
        self.source_self_attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.source_ffn_norm = nn.LayerNorm(width)
        self.source_ffn = FeedForward(width, config.mlp_ratio)
        self.target_self_norm = nn.LayerNorm(width)
        self.target_self_attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.cross_query_norm = nn.LayerNorm(width)
        self.cross_memory_norm = nn.LayerNorm(width)
        self.cross_attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.target_ffn_norm = nn.LayerNorm(width)
        self.target_ffn = FeedForward(width, config.mlp_ratio)

    def forward(self, source: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
        normalized_source = self.source_self_norm(source)
        source = source + self.source_self_attention(
            normalized_source, normalized_source, normalized_source,
            need_weights=False)[0]
        source = source + self.source_ffn(self.source_ffn_norm(source))

        normalized_target = self.target_self_norm(target)
        target = target + self.target_self_attention(
            normalized_target, normalized_target, normalized_target,
            need_weights=False)[0]
        memory = self.cross_memory_norm(source)
        target = target + self.cross_attention(
            self.cross_query_norm(target), memory, memory, need_weights=False)[0]
        target = target + self.target_ffn(self.target_ffn_norm(target))
        return source, target
