"""Distribution terms on the reference encoder's style tokens.

The probes showed three problems with the style representation:

* the null style (standard glyphs as references) is a constant vector roughly
  90x the energy of the writer cloud, so "no writer residual" is far from the
  origin;
* writer styles are dominated by one shared direction (77% of the energy);
* the writer cloud collapses to an effectively one-dimensional fan once the
  first two are constrained, which per-dimension variance does not prevent.

``style_terms`` therefore returns four dimensionless quantities measured on the
mean-pooled tokens ``v`` of a batch (one vector per episode):

    null_norm    ||n||^2                n = pooled null style, want it at 0
    mean_norm    ||E[v]||^2             shared direction, want it at 0
    rank_loss    1 - PR/(B-1)           PR = (sum l)^2 / sum l^2, want it at 0
    energy_loss  (E||v||^2 - target)^2  pins the scale so the ratios above
                                        cannot be gamed by inflating v or by
                                        collapsing everything to zero

Eigenvalues are taken on the B x B Gram matrix of the centred tokens, which is
cheap and exact (the batch has fewer samples than the 384 token dimensions).
Everything is computed in float32 with autocast disabled: ``eigvalsh`` has no
bfloat16 kernel.
"""
from __future__ import annotations

import torch
from torch import Tensor


def pooled_tokens(style: Tensor) -> Tensor:
    """Mean over the token axis: [B, N*Z, D] -> [B, D] in float32."""
    if style.ndim != 3:
        raise ValueError("style must have shape [B, N*Z, D]")
    return style.float().mean(dim=1)


def participation_ratio(values: Tensor) -> Tensor:
    """Effective number of directions carrying variance, in [1, B-1]."""
    centred = values - values.mean(dim=0, keepdim=True)
    gram = centred @ centred.T
    eigenvalues = torch.linalg.eigvalsh(gram).clamp_min(0)
    total = eigenvalues.sum()
    squared = eigenvalues.square().sum()
    return total.square() / squared.clamp_min(1e-12)


def style_terms(style: Tensor, null_style: Tensor,
                energy_target: float = 1.0) -> dict[str, Tensor]:
    """Return the four style-distribution terms plus their raw diagnostics."""
    if null_style.ndim != 3 or null_style.shape[:2] != style.shape[:2]:
        raise ValueError("null_style must match style on batch and token axes")
    if energy_target <= 0:
        raise ValueError("energy_target must be positive")

    device_type = style.device.type
    with torch.autocast(device_type=device_type, enabled=False):
        values = pooled_tokens(style)
        nulls = pooled_tokens(null_style)
        batch = values.shape[0]

        energy = values.square().sum(dim=-1).mean()
        mean_vector = values.mean(dim=0)
        mean_norm = mean_vector.square().sum()
        null_norm = nulls.square().sum(dim=-1).mean()

        ratio = participation_ratio(values)
        rank_loss = 1.0 - ratio / max(batch - 1, 1)
        energy_loss = (energy - values.new_tensor(energy_target)).square()

    return {"null_norm": null_norm, "mean_norm": mean_norm,
            "rank_loss": rank_loss, "energy_loss": energy_loss,
            "participation_ratio": ratio.detach(), "energy": energy.detach()}
