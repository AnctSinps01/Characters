"""Joint residual attention over reference and standard-glyph latents."""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .config import ModelConfig


class JointAttention(nn.Module):
    """One attention layer with separate reference and standard QKV projections.

    Both token groups attend to the concatenation of both groups; only the
    projections differ, so the model can treat references and the target glyph
    differently while still sharing one attention map.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.heads = config.heads
        self.scale = (config.dim // config.heads) ** -0.5
        self.reference_qkv = nn.Linear(config.dim, config.dim * 3)
        self.standard_qkv = nn.Linear(config.dim, config.dim * 3)
        self.out = nn.Linear(config.dim, config.dim)

    def _split(self, linear, tokens):
        batch, length, _ = tokens.shape
        qkv = linear(tokens).reshape(batch, length, 3, self.heads,
                                     tokens.shape[-1] // self.heads)
        return qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, L, Dh]

    def forward(self, reference: Tensor, standard: Tensor) -> tuple[Tensor, Tensor]:
        q_ref, k_ref, v_ref = self._split(self.reference_qkv, reference)
        q_std, k_std, v_std = self._split(self.standard_qkv, standard)
        key = torch.cat([k_ref, k_std], dim=2)
        value = torch.cat([v_ref, v_std], dim=2)
        dropout = 0.0
        out_ref = torch.nn.functional.scaled_dot_product_attention(
            q_ref, key, value, dropout_p=dropout)
        out_std = torch.nn.functional.scaled_dot_product_attention(
            q_std, key, value, dropout_p=dropout)
        merge = lambda x: x.transpose(1, 2).reshape(x.shape[0], x.shape[2], -1)
        return self.out(merge(out_ref)), self.out(merge(out_std))


class Block(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.norm_reference = nn.LayerNorm(config.dim)
        self.norm_standard = nn.LayerNorm(config.dim)
        self.attention = JointAttention(config)
        self.norm_ffn = nn.LayerNorm(config.dim)
        hidden = int(config.dim * config.mlp_ratio)
        self.ffn = nn.Sequential(nn.Linear(config.dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, config.dim))

    def forward(self, reference: Tensor, standard: Tensor) -> tuple[Tensor, Tensor]:
        delta_reference, delta_standard = self.attention(self.norm_reference(reference),
                                                         self.norm_standard(standard))
        reference = reference + delta_reference
        standard = standard + delta_standard
        merged = self.norm_ffn(torch.cat([reference, standard], dim=1))
        delta = self.ffn(merged)
        reference = reference + delta[:, :reference.shape[1]]
        standard = standard + delta[:, reference.shape[1]:]
        return reference, standard


class Ref2Latent(nn.Module):
    """Predict the posterior-mean latent of a target handwriting from references."""

    def __init__(self, config: ModelConfig = ModelConfig()) -> None:
        super().__init__()
        self.config = config
        positions = config.positions
        self.reference_input = nn.Linear(config.latent_channels, config.dim)
        self.standard_input = nn.Linear(config.latent_channels, config.dim)
        self.position = nn.Parameter(torch.zeros(1, positions, config.dim))
        self.reference_index = nn.Parameter(torch.zeros(1, config.references, 1, config.dim))
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.layers))
        self.norm = nn.LayerNorm(config.dim)
        self.output = nn.Linear(config.dim, config.latent_channels)
        nn.init.normal_(self.position, std=0.02)
        nn.init.normal_(self.reference_index, std=0.02)

    def _tokens(self, latent: Tensor, projection: nn.Linear) -> Tensor:
        batch, channels, side, _ = latent.shape
        flat = latent.reshape(batch, channels, side * side).transpose(1, 2)  # [B, P, C]
        return projection(flat) + self.position

    def forward(self, reference: Tensor, standard: Tensor) -> Tensor:
        cfg = self.config
        if reference.ndim != 5:
            raise ValueError("reference must have shape [B, N, C, H, W]")
        if reference.shape[1] != cfg.references:
            raise ValueError(f"reference must hold {cfg.references} samples")
        expected = (cfg.latent_channels, cfg.latent_size, cfg.latent_size)
        if tuple(reference.shape[2:]) != expected or tuple(standard.shape[1:]) != expected:
            raise ValueError(f"latent shape must be [B, N, {expected}] and [B, {expected}]")
        batch = standard.shape[0]
        tokens = self._tokens(reference.reshape(batch * cfg.references, *expected),
                              self.reference_input)
        tokens = tokens.reshape(batch, cfg.references, cfg.positions, cfg.dim)
        tokens = tokens + self.reference_index
        reference_tokens = tokens.reshape(batch, cfg.references * cfg.positions, cfg.dim)
        standard_tokens = self._tokens(standard, self.standard_input)
        for block in self.blocks:
            reference_tokens, standard_tokens = block(reference_tokens, standard_tokens)
        standard_tokens = self.norm(standard_tokens)
        flat = self.output(standard_tokens)                      # [B, P, C]
        side = cfg.latent_size
        return flat.transpose(1, 2).reshape(batch, cfg.latent_channels, side, side)
