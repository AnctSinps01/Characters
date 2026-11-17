"""Vision Transformer OCR model for 64x64 grayscale Chinese glyphs."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True, slots=True)
class ViTConfig:
    image_size: int = 64
    patch_size: int = 8
    in_channels: int = 1
    num_classes: int = 3755
    hidden_size: int = 384
    depth: int = 12
    num_heads: int = 6
    mlp_ratio: float = 4.0
    dropout: float = 0.0

    def __post_init__(self) -> None:
        for name in ("image_size", "patch_size", "in_channels", "num_classes",
                     "hidden_size", "num_heads"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.depth < 1:
            raise ValueError("depth must be positive")
        if self.image_size % self.patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")

    @property
    def patch_count(self) -> int:
        return (self.image_size // self.patch_size) ** 2

    @property
    def token_count(self) -> int:
        return self.patch_count + 1


class PatchEmbedding(nn.Module):
    """Non-overlapping image patches projected to transformer width."""

    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        self.image_size = config.image_size
        self.in_channels = config.in_channels
        self.projection = nn.Conv2d(
            config.in_channels,
            config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )

    def forward(self, images: Tensor) -> Tensor:
        expected = (self.image_size, self.image_size)
        if images.ndim != 4 or images.shape[1] != self.in_channels:
            raise ValueError(f"images must have shape [batch, {self.in_channels}, height, width]")
        if images.shape[-2:] != expected:
            raise ValueError(f"expected {expected} images, got {tuple(images.shape[-2:])}")
        return self.projection(images).flatten(2).transpose(1, 2)


class TransformerBlock(nn.Module):
    """Pre-LN ViT block: residual self-attention followed by residual MLP."""

    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        width = config.hidden_size
        self.norm1 = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(
            width, config.num_heads, dropout=config.dropout, batch_first=True
        )
        self.attention_dropout = nn.Dropout(config.dropout)
        self.norm2 = nn.LayerNorm(width)
        inner = int(width * config.mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(width, inner),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(inner, width),
            nn.Dropout(config.dropout),
        )

    def forward(self, tokens: Tensor) -> Tensor:
        normalized = self.norm1(tokens)
        attended = self.attention(normalized, normalized, normalized, need_weights=False)[0]
        tokens = tokens + self.attention_dropout(attended)
        return tokens + self.mlp(self.norm2(tokens))


class VisionTransformerOCR(nn.Module):
    """CLS-token ViT classifier; returns logits for the prepared character indices."""

    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        self.config = config
        width = config.hidden_size
        self.patch_embed = PatchEmbedding(config)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, width))
        self.position_embedding = nn.Parameter(torch.zeros(1, config.token_count, width))
        self.position_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(config.depth))
        self.norm = nn.LayerNorm(width)
        self.head = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(width, config.num_classes),
        )
        self.apply(self._init_weights)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            if module.in_proj_weight is not None:
                nn.init.xavier_uniform_(module.in_proj_weight)
            if module.in_proj_bias is not None:
                nn.init.zeros_(module.in_proj_bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def tokenize(self, images: Tensor) -> Tensor:
        patches = self.patch_embed(images)
        cls = self.cls_token.expand(images.shape[0], -1, -1)
        tokens = torch.cat((cls, patches), dim=1)
        return self.position_dropout(tokens + self.position_embedding)

    def forward_features(self, images: Tensor) -> Tensor:
        tokens = self.tokenize(images)
        for block in self.blocks:
            tokens = block(tokens)
        return self.norm(tokens)[:, 0]

    def forward(self, images: Tensor) -> Tensor:
        return self.head(self.forward_features(images))

    def checkpoint_config(self) -> dict:
        return asdict(self.config)
