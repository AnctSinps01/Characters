"""Rectified-flow model over blocked-VAE codes.

Two variants share this file.  The plain one keeps the six concatenated channels
untouched through the trunk.  The dual-stream one routes the pooled reference
codes through their own dilated branch (1 -> style_hidden -> 1), multiplies that
style map into the trunk pointwise, and runs the trunk at a wider channel count
(6 -> trunk_channels -> 3).  Time enters every block through adaLN-Zero, so a
freshly initialised block is the identity.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def group_count(channels: int) -> int:
    for groups in (32, 16, 8, 4, 3, 2):
        if channels % groups == 0:
            return groups
    return 1


@dataclass
class FlowConfig:
    latent_size: int = 8
    content_channels: int = 2
    author_channels: int = 1
    blocks: int = 6
    time_dim: int = 128
    ref_hidden: int = 16
    trunk_channels: int = 6
    style_hidden: int = 0
    style_dilation: int = 2

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

    def __init__(self, hidden: int = 16, dilation: int = 1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, hidden, 3, padding=1), nn.GroupNorm(group_count(hidden), hidden), nn.SiLU(),
            nn.Conv2d(hidden, 1, 3, padding=dilation, dilation=dilation),
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


class StyleBranch(nn.Module):
    """Dilated 1 -> style_hidden -> 1 branch producing a pointwise style map."""

    def __init__(self, style_hidden: int, dilation: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, style_hidden, 3, padding=dilation, dilation=dilation),
            nn.GroupNorm(group_count(style_hidden), style_hidden), nn.SiLU(),
            nn.Conv2d(style_hidden, 1, 3, padding=1),
        )

    def forward(self, pooled):
        return self.layers(pooled)


class AdaZeroBlock(nn.Module):
    """adaLN-Zero residual block that keeps the channel count unchanged."""

    def __init__(self, channels: int, time_dim: int):
        super().__init__()
        self.norm = nn.GroupNorm(group_count(channels), channels)
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
        self.modulation = nn.Linear(time_dim, channels * 3)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x, embedding):
        scale, shift, gate = self.modulation(embedding)[:, :, None, None].chunk(3, dim=1)
        return x + gate * (self.norm(self.conv(self.norm(x))) * (1 + scale) + shift)


class FlowModel(nn.Module):
    def __init__(self, config: FlowConfig):
        super().__init__()
        self.config = config
        channels = config.trunk_channels
        self.pool = ReferencePool(config.ref_hidden, config.style_dilation if config.style_hidden else 1)
        self.style = StyleBranch(config.style_hidden, config.style_dilation) if config.style_hidden else None
        if channels == config.input_channels:
            self.stem = nn.Identity()
        else:
            self.stem = nn.Sequential(
                nn.Conv2d(config.input_channels, channels, 3, padding=1),
                nn.GroupNorm(group_count(channels), channels), nn.SiLU())
        self.time_mlp = nn.Sequential(
            nn.Linear(config.time_dim, config.time_dim), nn.SiLU(),
            nn.Linear(config.time_dim, config.time_dim))
        self.blocks = nn.ModuleList([AdaZeroBlock(channels, config.time_dim)
                                     for _ in range(config.blocks)])
        self.head_norm = nn.GroupNorm(group_count(channels), channels)
        self.head = nn.Conv2d(channels, config.target_channels, 3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, refs, standard, x_t, t, counts=None):
        embedding = self.time_mlp(timestep_embedding(t, self.config.time_dim))
        pooled = self.pool(refs, counts)
        features = self.stem(torch.cat([pooled, standard, x_t], dim=1))
        if self.style is not None:
            features = features * (1 + self.style(pooled))
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
