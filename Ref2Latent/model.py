"""Joint residual attention over reference latents and one standard-glyph latent.

One latent is one token: a 4x8x8 latent is flattened to 256 numbers, projected to
the model width, and attends to every other latent in the batch item.
"""
from __future__ import annotations

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
        out_ref = torch.nn.functional.scaled_dot_product_attention(q_ref, key, value)
        out_std = torch.nn.functional.scaled_dot_product_attention(q_std, key, value)
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
        self.reference_input = nn.Linear(config.latent_dim, config.dim)
        self.standard_input = nn.Linear(config.latent_dim, config.dim)
        self.reference_index = nn.Parameter(torch.zeros(1, config.references, config.dim))
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.layers))
        self.norm = nn.LayerNorm(config.dim)
        self.output = nn.Linear(config.dim, config.latent_dim)
        nn.init.normal_(self.reference_index, std=0.02)

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
        reference_tokens = self.reference_input(reference.reshape(batch, cfg.references, -1))
        reference_tokens = reference_tokens + self.reference_index
        standard_tokens = self.standard_input(standard.reshape(batch, -1)).unsqueeze(1)
        for block in self.blocks:
            reference_tokens, standard_tokens = block(reference_tokens, standard_tokens)
        flat = self.output(self.norm(standard_tokens).squeeze(1))
        return flat.reshape(batch, cfg.latent_channels, cfg.latent_size, cfg.latent_size)
