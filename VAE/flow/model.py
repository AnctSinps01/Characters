"""Rectified-flow model over blocked-VAE codes.

The trunk only ever sees six channels: one pooled channel summarising the
reference author codes, the standard glyph's two content channels and the three
channels of the current noisy sample.  Time enters every block through
adaLN-Zero, so a freshly initialised block is the identity and the pretrained
VAE codes are not thrown away at step zero.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class FlowConfig:
    latent_size: int = 8
    content_channels: int = 2
    author_channels: int = 1
    hidden: int = 96
    blocks: int = 6
    time_dim: int = 128
    ref_hidden: int = 16

    @property
    def target_channels(self) -> int:
        return self.content_channels + self.author_channels

    @property
    def input_channels(self) -> int:
        return 1 + self.content_channels + self.target_channels


def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10_000.0):
    half = dim // 2
    frequencies = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    arguments = t.float()[:, None] * frequencies[None]
    embedding = torch.cat([torch.cos(arguments), torch.sin(arguments)], dim=1)
    if dim % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding


class ReferencePool(nn.Module):
    """Shared per-reference convolution followed by mean pooling over references."""

    def __init__(self, hidden: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, hidden, 3, padding=1), nn.GroupNorm(8, hidden), nn.SiLU(),
            nn.Conv2d(hidden, 1, 3, padding=1),
        )

    def forward(self, refs, counts=None):
        batch, count, _, height, width = refs.shape
        if count == 0:
            return refs.new_zeros(batch, 1, height, width)
        pooled = self.layers(refs.reshape(batch * count, 1, height, width))
        pooled = pooled.reshape(batch, count, 1, height, width)
        if counts is None:
            return pooled.mean(dim=1)
        mask = (torch.arange(count, device=refs.device)[None] < counts[:, None]).to(pooled.dtype)
        total = (pooled * mask[:, :, None, None, None]).sum(dim=1)
        return total / counts.to(pooled.dtype).view(batch, 1, 1, 1)


class AdaZeroBlock(nn.Module):
    """adaln-Zero residual block: zero-initialised gate keeps the block as identity."""

    def __init__(self, hidden: int, time_dim: int):
        super().__init__()
        self.norm = nn.GroupNorm(32, hidden)
        self.conv = nn.Conv2d(hidden, hidden, 3, padding=1)
        self.modulation = nn.Linear(time_dim, hidden * 3)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x, embedding):
        scale, shift, gate = self.modulation(embedding)[:, :, None, None].chunk(3, dim=1)
        return x + gate * (self.norm(self.conv(self.norm(x))) * (1 + scale) + shift)


class FlowModel(nn.Module):
    def __init__(self, config: FlowConfig):
        super().__init__()
        self.config = config
        hidden = config.hidden
        self.pool = ReferencePool(config.ref_hidden)
        self.time_mlp = nn.Sequential(
            nn.Linear(config.time_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.stem = nn.Conv2d(config.input_channels, hidden, 3, padding=1)
        self.stem_norm = nn.GroupNorm(32, hidden)
        self.blocks = nn.ModuleList([AdaZeroBlock(hidden, hidden) for _ in range(config.blocks)])
        self.head_norm = nn.GroupNorm(32, hidden)
        self.head = nn.Conv2d(hidden, config.target_channels, 3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, refs, standard, x_t, t, counts=None):
        embedding = self.time_mlp(timestep_embedding(t, self.config.time_dim))
        features = self.stem(torch.cat([self.pool(refs, counts), standard, x_t], dim=1))
        features = F.silu(self.stem_norm(features))
        for block in self.blocks:
            features = block(features, embedding)
        return self.head(F.silu(self.head_norm(features)))


def flow_loss(model, refs, standard, target, counts=None, generator=None):
    """Rectified flow: x_t = (1-t) noise + t target, regress the velocity target - noise."""
    batch = target.shape[0]
    noise = torch.randn(target.shape, device=target.device, generator=generator)
    t = torch.rand(batch, device=target.device, generator=generator)
    mixed = (1 - t)[:, None, None, None] * noise + t[:, None, None, None] * target
    predicted = model(refs, standard, mixed, t, counts)
    velocity = target - noise
    return F.mse_loss(predicted, velocity), predicted


@torch.no_grad()
def sample(model, refs, standard, steps: int = 16, counts=None, generator=None):
    """Euler integration from noise at t=0 to data at t=1."""
    batch = standard.shape[0]
    shape = (batch, model.config.target_channels,
             model.config.latent_size, model.config.latent_size)
    x = torch.randn(shape, device=standard.device, generator=generator)
    for index in range(steps):
        t0 = index / steps
        t1 = (index + 1) / steps
        current = torch.full((batch,), t0, device=standard.device)
        x = x + (t1 - t0) * model(refs, standard, x, current, counts)
    return x
