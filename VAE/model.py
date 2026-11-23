"""Spatial Gaussian VAE for single-channel glyph reconstruction."""
from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class VAEConfig:
    image_size: int = 64
    base_channels: int = 64
    latent_channels: int = 4

    def __post_init__(self):
        for name in ("image_size", "base_channels", "latent_channels"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.image_size % 8:
            raise ValueError("image_size must be divisible by 8")
        if self.base_channels % 8:
            raise ValueError("base_channels must be divisible by 8 for GroupNorm")

    @property
    def latent_shape(self):
        return (self.latent_channels, self.image_size // 8, self.image_size // 8)


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


class GlyphVAE(nn.Module):
    """Encode q(z|x) as diagonal Gaussians; decode a sample or its mean.

    No image/encoder skip connections reach the decoder. The only information
    path from x to its reconstruction is the spatial latent z.
    """
    def __init__(self, config: VAEConfig = VAEConfig()):
        super().__init__()
        self.config = config
        c = config.base_channels
        self.encoder = nn.Sequential(
            nn.Conv2d(1, c, 3, padding=1), ResidualBlock(c),
            nn.Conv2d(c, c * 2, 4, stride=2, padding=1), ResidualBlock(c * 2),
            nn.Conv2d(c * 2, c * 4, 4, stride=2, padding=1), ResidualBlock(c * 4),
            nn.Conv2d(c * 4, c * 4, 4, stride=2, padding=1), ResidualBlock(c * 4),
            norm(c * 4), nn.SiLU(),
            nn.Conv2d(c * 4, config.latent_channels * 2, 3, padding=1),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(config.latent_channels, c * 4, 3, padding=1), ResidualBlock(c * 4),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(c * 4, c * 4, 3, padding=1), ResidualBlock(c * 4),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(c * 4, c * 2, 3, padding=1), ResidualBlock(c * 2),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(c * 2, c, 3, padding=1), ResidualBlock(c),
            norm(c), nn.SiLU(), nn.Conv2d(c, 1, 3, padding=1), nn.Tanh(),
        )

    def encode(self, images: Tensor):
        expected = (1, self.config.image_size, self.config.image_size)
        if images.ndim != 4 or tuple(images.shape[1:]) != expected:
            raise ValueError(f"images must have shape [B, {expected}]")
        mean, logvar = self.encoder(images).chunk(2, dim=1)
        # FP32 distribution arithmetic, including under bf16/fp16 autocast.
        return mean.float(), logvar.float().clamp(-20.0, 10.0)

    @staticmethod
    def reparameterize(mean: Tensor, logvar: Tensor, noise: Tensor | None = None):
        if mean.shape != logvar.shape:
            raise ValueError("posterior mean/logvar shapes differ")
        if noise is None:
            noise = torch.randn_like(mean)
        elif noise.shape != mean.shape:
            raise ValueError("noise must match posterior shape")
        return mean + (0.5 * logvar).exp() * noise

    def decode(self, latent: Tensor):
        if latent.ndim != 4 or tuple(latent.shape[1:]) != self.config.latent_shape:
            raise ValueError(f"latent must have shape [B, {self.config.latent_shape}]")
        return self.decoder(latent)

    def forward(self, images: Tensor, *, sample: bool = True, noise: Tensor | None = None):
        mean, logvar = self.encode(images)
        z = self.reparameterize(mean, logvar, noise) if sample else mean
        return self.decode(z), mean, logvar


def vae_loss(reconstruction: Tensor, target: Tensor, mean: Tensor,
             logvar: Tensor, beta: float):
    """Pixel-mean MSE + beta * mean KL per scalar latent dimension.

    Explicit reductions make beta comparable across image/latent sizes. The
    unweighted per-image KL sum is returned as a diagnostic too.
    """
    if not math.isfinite(beta) or beta < 0:
        raise ValueError("beta must be finite and nonnegative")
    if reconstruction.shape != target.shape or mean.shape != logvar.shape:
        raise ValueError("loss input shapes differ")
    with torch.autocast(device_type=mean.device.type, enabled=False):
        mse = F.mse_loss(reconstruction.float(), target.float())
        kl_elements = 0.5 * (mean.float().square() + logvar.float().exp() - 1 - logvar.float())
        kl = kl_elements.mean()
        kl_per_image = kl_elements.flatten(1).sum(1).mean()
        loss = mse + beta * kl
    return {"loss": loss, "mse": mse, "kl": kl, "kl_per_image": kl_per_image}
