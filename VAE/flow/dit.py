"""DiT-style flow model over blocked-VAE codes.

Sequence layout: n reference tokens (one per reference character) followed by
64 spatial tokens (one per latent position).  The two kinds use separate QKV
projections, which removes the need for an explicit type embedding.  Time
enters every block through adaLN-Zero, and only the spatial tokens are
projected back to the four velocity channels.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DiTConfig:
    latent_size: int = 8
    content_channels: int = 2
    author_channels: int = 1
    predict_free: bool = True
    d_model: int = 128
    depth: int = 24
    heads: int = 8
    ffn: int = 512
    ref_dim: int = 64
    time_dim: int = 256

    @property
    def target_channels(self) -> int:
        return self.content_channels + self.author_channels + (1 if self.predict_free else 0)

    @property
    def spatial_tokens(self) -> int:
        return self.latent_size * self.latent_size

    @property
    def input_channels(self) -> int:
        return 4 + self.target_channels


def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10_000.0):
    half = dim // 2
    frequencies = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    arguments = t.float()[:, None] * frequencies[None]
    embedding = torch.cat([torch.cos(arguments), torch.sin(arguments)], dim=1)
    if dim % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class JointBlock(nn.Module):
    """Self-attention over reference + spatial tokens, then an FFN, both adaLN-Zero."""

    def __init__(self, config: DiTConfig):
        super().__init__()
        d = config.d_model
        self.heads = config.heads
        self.norm_spatial = nn.LayerNorm(d, elementwise_affine=False)
        self.norm_ref = nn.LayerNorm(d, elementwise_affine=False)
        self.spatial_qkv = nn.Linear(d, d * 3)
        self.ref_qkv = nn.Linear(d, d * 3)
        self.spatial_out = nn.Linear(d, d)
        self.ref_out = nn.Linear(d, d)
        self.norm_ffn = nn.LayerNorm(d, elementwise_affine=False)
        self.ffn = nn.Sequential(nn.Linear(d, config.ffn), nn.GELU(), nn.Linear(config.ffn, d))
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(d, d * 6))
        nn.init.zeros_(self.modulation[1].weight)
        nn.init.zeros_(self.modulation[1].bias)

    def forward(self, spatial, refs, embedding, ref_mask=None):
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = \
            self.modulation(embedding)[:, None].chunk(6, dim=-1)
        d = self.heads
        batch, n_spatial, _ = spatial.shape
        n_ref = refs.shape[1]

        q_s, k_s, v_s = self.spatial_qkv(modulate(self.norm_spatial(spatial), shift_a, scale_a)).chunk(3, dim=-1)
        q_r, k_r, v_r = self.ref_qkv(modulate(self.norm_ref(refs), shift_a, scale_a)).chunk(3, dim=-1)

        def heads(x):
            return x.reshape(batch, -1, d, x.shape[-1] // d).transpose(1, 2)

        q = heads(torch.cat([q_r, q_s], dim=1))
        k = heads(torch.cat([k_r, k_s], dim=1))
        v = heads(torch.cat([v_r, v_s], dim=1))
        mask = None
        if ref_mask is not None:
            valid = torch.cat([ref_mask, torch.ones(batch, n_spatial, dtype=torch.bool, device=ref_mask.device)], 1)
            mask = valid[:, None, None, :]
        attended = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        attended = attended.transpose(1, 2).reshape(batch, n_ref + n_spatial, -1)

        spatial = spatial + gate_a * self.spatial_out(attended[:, n_ref:])
        refs = refs + gate_a * self.ref_out(attended[:, :n_ref])

        hidden = modulate(self.norm_ffn(spatial), shift_f, scale_f)
        spatial = spatial + gate_f * self.ffn(hidden)
        return spatial, refs


class DiTFlow(nn.Module):
    def __init__(self, config: DiTConfig):
        super().__init__()
        self.config = config
        d = config.d_model
        self.spatial_embed = nn.Linear(config.input_channels, d)
        self.ref_embed = nn.Linear(config.ref_dim, d)
        self.pos = nn.Parameter(torch.zeros(1, config.spatial_tokens, d))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.time_mlp = nn.Sequential(nn.Linear(config.time_dim, d), nn.SiLU(), nn.Linear(d, d))
        self.blocks = nn.ModuleList([JointBlock(config) for _ in range(config.depth)])
        self.norm_out = nn.LayerNorm(d, elementwise_affine=False)
        self.head = nn.Linear(d, config.target_channels)
        self.modulation_out = nn.Sequential(nn.SiLU(), nn.Linear(d, d * 2))
        nn.init.zeros_(self.modulation_out[1].weight)
        nn.init.zeros_(self.modulation_out[1].bias)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, refs, standard, x_t, t, counts=None):
        batch, n_ref = refs.shape[0], refs.shape[1]
        spatial = torch.cat([standard, x_t], dim=1)
        tokens = self.spatial_embed(spatial.flatten(2).transpose(1, 2)) + self.pos
        ref_tokens = self.ref_embed(refs.reshape(batch, n_ref, -1))
        embedding = self.time_mlp(timestep_embedding(t, self.config.time_dim))

        ref_mask = None
        if counts is not None:
            ref_mask = torch.arange(n_ref, device=refs.device)[None] < counts[:, None]
        for block in self.blocks:
            tokens, ref_tokens = block(tokens, ref_tokens, embedding, ref_mask)

        shift, scale = self.modulation_out(embedding)[:, None].chunk(2, dim=-1)
        out = self.head(modulate(self.norm_out(tokens), shift, scale))
        return out.transpose(1, 2).reshape(batch, self.config.target_channels,
                                           self.config.latent_size, self.config.latent_size)


def flow_loss(model, refs, standard, target, counts=None, generator=None):
    batch = target.shape[0]
    noise = torch.randn(target.shape, device=target.device, generator=generator)
    t = torch.rand(batch, device=target.device, generator=generator)
    mixed = (1 - t)[:, None, None, None] * noise + t[:, None, None, None] * target
    predicted = model(refs, standard, mixed, t, counts)
    return F.mse_loss(predicted, target - noise), predicted


@torch.no_grad()
def sample(model, refs, standard, steps: int = 16, counts=None, generator=None):
    batch = standard.shape[0]
    shape = (batch, model.config.target_channels, model.config.latent_size, model.config.latent_size)
    x = torch.randn(shape, device=standard.device, generator=generator)
    for index in range(steps):
        t0 = index / steps
        t1 = (index + 1) / steps
        current = torch.full((batch,), t0, device=standard.device)
        x = x + (t1 - t0) * model(refs, standard, x, current, counts)
    return x
