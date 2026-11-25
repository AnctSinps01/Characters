"""Decoder for a two-code latent: content code followed by author code."""
from __future__ import annotations

import torch
from torch import Tensor, nn

from .config import DecoderConfig


def norm(channels):
    return nn.GroupNorm(min(8, channels), channels, eps=1e-6)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.layers = nn.Sequential(
            norm(channels), nn.SiLU(), nn.Conv2d(channels, channels, 3, padding=1),
            norm(channels), nn.SiLU(), nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x):
        return x + self.layers(x)


class JointDecoder(nn.Module):
    """The VAE decoder, widened to accept two concatenated latent codes."""

    def __init__(self, config: DecoderConfig = DecoderConfig()):
        super().__init__()
        self.config = config
        c = config.base_channels
        self.levels = 3
        channels = [c * 4, c * 4, c * 2, c]
        decoder = [nn.Conv2d(config.in_channels, channels[0], 3, padding=1),
                   ResidualBlock(channels[0])]
        for source, target in zip(channels[:-1], channels[1:]):
            decoder += [nn.Upsample(scale_factor=2, mode="nearest"),
                        nn.Conv2d(source, target, 3, padding=1), ResidualBlock(target)]
        decoder += [norm(channels[-1]), nn.SiLU(),
                    nn.Conv2d(channels[-1], 1, 3, padding=1), nn.Tanh()]
        self.decoder = nn.Sequential(*decoder)

    def forward(self, content: Tensor, author: Tensor) -> Tensor:
        expected = (self.config.latent_channels, self.config.latent_size,
                    self.config.latent_size)
        if tuple(content.shape[1:]) != expected or tuple(author.shape[1:]) != expected:
            raise ValueError(f"codes must have shape [B, {expected}]")
        return self.decoder(torch.cat([content, author], dim=1))
