from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


_LAYER_SCALE_INIT = 1e-4


def _sincos_1d(positions: Tensor, width: int) -> Tensor:
    if width % 2:
        raise ValueError("1D sin/cos width must be even")
    frequencies = torch.arange(width // 2, dtype=torch.float32)
    frequencies = torch.exp(-math.log(10_000.0) * frequencies / (width // 2))
    angles = positions.float().reshape(-1, 1) * frequencies.reshape(1, -1)
    return torch.cat((angles.sin(), angles.cos()), dim=1)


def sincos_2d_position_embedding(grid_size: int, width: int) -> Tensor:
    """Return fixed row/column positions with shape ``[1, grid_size**2, width]``."""

    if width % 4:
        raise ValueError("2D sin/cos width must be divisible by 4")
    rows, columns = torch.meshgrid(
        torch.arange(grid_size), torch.arange(grid_size), indexing="ij"
    )
    row_embedding = _sincos_1d(rows.reshape(-1), width // 2)
    column_embedding = _sincos_1d(columns.reshape(-1), width // 2)
    return torch.cat((row_embedding, column_embedding), dim=1).unsqueeze(0)


class PatchEmbed2D(nn.Module):
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        in_channels: int,
        hidden_size: int,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.projection = nn.Conv2d(
            in_channels,
            hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
        )
        positions = sincos_2d_position_embedding(image_size // patch_size, hidden_size)
        self.register_buffer("positions", positions, persistent=False)

    def forward(self, images: Tensor) -> Tensor:
        if images.ndim != 4:
            raise ValueError(f"images must have shape [B, C, H, W], got {images.shape}")
        if images.shape[-2:] != (self.image_size, self.image_size):
            raise ValueError(
                f"expected {self.image_size}x{self.image_size} images, got {images.shape[-2:]}"
            )
        tokens = self.projection(images).flatten(2).transpose(1, 2)
        return tokens + self.positions.to(dtype=tokens.dtype)


class Mlp(nn.Module):
    def __init__(self, hidden_size: int, mlp_ratio: float) -> None:
        super().__init__()
        inner_size = int(hidden_size * mlp_ratio)
        self.layers = nn.Sequential(
            nn.Linear(hidden_size, inner_size),
            nn.GELU(),
            nn.Linear(inner_size, hidden_size),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.layers(inputs)


class SelfAttentionBlock(nn.Module):
    """Pre-norm self-attention and MLP with static LayerScale residuals."""

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(hidden_size)
        self.attention = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=0.0, batch_first=True
        )
        self.attention_scale = nn.Parameter(
            torch.full((hidden_size,), _LAYER_SCALE_INIT)
        )
        self.mlp_norm = nn.LayerNorm(hidden_size)
        self.mlp = Mlp(hidden_size, mlp_ratio)
        self.mlp_scale = nn.Parameter(torch.full((hidden_size,), _LAYER_SCALE_INIT))

    def forward(self, inputs: Tensor) -> Tensor:
        normalized = self.attention_norm(inputs)
        attended = self.attention(normalized, normalized, normalized, need_weights=False)[0]
        outputs = inputs + self.attention_scale * attended
        return outputs + self.mlp_scale * self.mlp(self.mlp_norm(outputs))


class _JointAttentionStream(nn.Module):
    """One stream's independent projections and LayerScale residual updates."""

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.attention_norm = nn.LayerNorm(hidden_size)
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size)
        self.projection = nn.Linear(hidden_size, hidden_size)
        self.attention_scale = nn.Parameter(
            torch.full((hidden_size,), _LAYER_SCALE_INIT)
        )
        self.mlp_norm = nn.LayerNorm(hidden_size)
        self.mlp = Mlp(hidden_size, mlp_ratio)
        self.mlp_scale = nn.Parameter(torch.full((hidden_size,), _LAYER_SCALE_INIT))
        nn.init.xavier_uniform_(self.qkv.weight)
        nn.init.zeros_(self.qkv.bias)

    def project(self, tokens: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        batch, count, width = tokens.shape
        qkv = self.qkv(self.attention_norm(tokens)).reshape(
            batch, count, 3, self.num_heads, width // self.num_heads
        ).permute(2, 0, 3, 1, 4)
        return qkv.unbind(0)

    def update(self, tokens: Tensor, attended: Tensor) -> Tensor:
        outputs = tokens + self.attention_scale * self.projection(attended)
        return outputs + self.mlp_scale * self.mlp(self.mlp_norm(outputs))


class JointAttentionBlock(nn.Module):
    """Two independent streams attending jointly to both reference images."""

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float) -> None:
        super().__init__()
        self.handwriting = _JointAttentionStream(hidden_size, num_heads, mlp_ratio)
        self.standard = _JointAttentionStream(hidden_size, num_heads, mlp_ratio)

    def forward(self, handwriting: Tensor, standard: Tensor) -> tuple[Tensor, Tensor]:
        if handwriting.ndim != 3 or standard.shape != handwriting.shape:
            raise ValueError("joint reference streams must have matching [B, L, D] shapes")
        hand_q, hand_k, hand_v = self.handwriting.project(handwriting)
        standard_q, standard_k, standard_v = self.standard.project(standard)
        # Sequence concatenation gives every query access to BOTH images.
        queries = torch.cat((hand_q, standard_q), dim=2)
        keys = torch.cat((hand_k, standard_k), dim=2)
        values = torch.cat((hand_v, standard_v), dim=2)
        attended = F.scaled_dot_product_attention(
            queries, keys, values, dropout_p=0.0
        )
        batch, _, count, _ = attended.shape
        attended = attended.transpose(1, 2).reshape(batch, count, handwriting.shape[-1])
        hand_attended, standard_attended = attended.chunk(2, dim=1)
        return (
            self.handwriting.update(handwriting, hand_attended),
            self.standard.update(standard, standard_attended),
        )


class CrossAttentionBlock(nn.Module):
    """Update queries with static LayerScale attention and MLP residuals."""

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(hidden_size)
        self.memory_norm = nn.LayerNorm(hidden_size)
        self.attention = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=0.0, batch_first=True
        )
        self.attention_scale = nn.Parameter(
            torch.full((hidden_size,), _LAYER_SCALE_INIT)
        )
        self.mlp_norm = nn.LayerNorm(hidden_size)
        self.mlp = Mlp(hidden_size, mlp_ratio)
        self.mlp_scale = nn.Parameter(torch.full((hidden_size,), _LAYER_SCALE_INIT))

    def forward(
        self,
        queries: Tensor,
        memory: Tensor,
        *,
        memory_padding_mask: Tensor | None = None,
    ) -> Tensor:
        normalized_queries = self.query_norm(queries)
        normalized_memory = self.memory_norm(memory)
        attended = self.attention(
            normalized_queries,
            normalized_memory,
            normalized_memory,
            key_padding_mask=memory_padding_mask,
            need_weights=False,
        )[0]
        outputs = queries + self.attention_scale * attended
        return outputs + self.mlp_scale * self.mlp(self.mlp_norm(outputs))


def modulate(inputs: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return inputs * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class ContinuousTimeEmbedding(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        frequency_size: int,
        time_scale: float,
        max_period: float = 10_000.0,
    ) -> None:
        super().__init__()
        frequencies = torch.arange(frequency_size // 2, dtype=torch.float32)
        frequencies = torch.exp(
            -math.log(max_period) * frequencies / (frequency_size // 2)
        )
        self.register_buffer("frequencies", frequencies, persistent=False)
        self.time_scale = time_scale
        self.mlp = nn.Sequential(
            nn.Linear(frequency_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, times: Tensor) -> Tensor:
        if times.ndim != 1:
            raise ValueError(f"times must have shape [B], got {times.shape}")
        angles = (
            times.float().unsqueeze(1)
            * self.time_scale
            * self.frequencies.unsqueeze(0)
        )
        embedding = torch.cat((angles.cos(), angles.sin()), dim=1)
        return self.mlp(embedding)


class AdaLNZeroDiTBlock(nn.Module):
    """Time-modulated SA/FFN with independent content CA then style CA."""

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float) -> None:
        super().__init__()
        self.self_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.self_attention = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=0.0, batch_first=True
        )
        self.content_query_norm = nn.LayerNorm(hidden_size)
        self.content_memory_norm = nn.LayerNorm(hidden_size)
        self.content_attention = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=0.0, batch_first=True
        )
        self.style_query_norm = nn.LayerNorm(hidden_size)
        self.style_memory_norm = nn.LayerNorm(hidden_size)
        self.style_attention = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=0.0, batch_first=True
        )
        self.content_scale = nn.Parameter(
            torch.full((hidden_size,), _LAYER_SCALE_INIT)
        )
        self.style_scale = nn.Parameter(torch.full((hidden_size,), _LAYER_SCALE_INIT))
        self.mlp_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(hidden_size, mlp_ratio)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size))
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(
        self, inputs: Tensor, content: Tensor, style: Tensor,
        time_embedding: Tensor, style_padding_mask: Tensor | None = None,
    ) -> Tensor:
        scale_sa, shift_sa, gate_sa, scale_mlp, shift_mlp, gate_mlp = (
            self.modulation(time_embedding).chunk(6, dim=1)
        )
        sa_inputs = modulate(self.self_norm(inputs), shift_sa, scale_sa)
        sa_outputs = self.self_attention(sa_inputs, sa_inputs, sa_inputs, need_weights=False)[0]
        outputs = inputs + gate_sa.unsqueeze(1) * sa_outputs

        normalized_content = self.content_memory_norm(content)
        content_outputs = self.content_attention(
            self.content_query_norm(outputs), normalized_content, normalized_content,
            need_weights=False,
        )[0]
        outputs = outputs + self.content_scale * content_outputs

        normalized_style = self.style_memory_norm(style)
        style_outputs = self.style_attention(
            self.style_query_norm(outputs), normalized_style, normalized_style,
            key_padding_mask=style_padding_mask, need_weights=False,
        )[0]
        outputs = outputs + self.style_scale * style_outputs

        mlp_inputs = modulate(self.mlp_norm(outputs), shift_mlp, scale_mlp)
        return outputs + gate_mlp.unsqueeze(1) * self.mlp(mlp_inputs)


class FinalVelocityLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))
        self.projection = nn.Linear(hidden_size, patch_size**2 * out_channels)
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, tokens: Tensor, time_embedding: Tensor) -> Tensor:
        scale, shift = self.modulation(time_embedding).chunk(2, dim=1)
        return self.projection(modulate(self.norm(tokens), shift, scale))


def unpatchify(patches: Tensor, grid_size: int, patch_size: int, channels: int) -> Tensor:
    batch, patch_count, patch_width = patches.shape
    expected_count = grid_size**2
    expected_width = patch_size**2 * channels
    if patch_count != expected_count or patch_width != expected_width:
        raise ValueError(
            f"expected patches [B, {expected_count}, {expected_width}], got {patches.shape}"
        )
    images = patches.reshape(
        batch, grid_size, grid_size, patch_size, patch_size, channels
    )
    images = images.permute(0, 5, 1, 3, 2, 4).contiguous()
    return images.reshape(
        batch, channels, grid_size * patch_size, grid_size * patch_size
    )
