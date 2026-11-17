"""Distribution losses on the exact style tokens consumed by the DiT trunk.

Each generator block applies its own ``style_memory_norm`` before cross
attention. The caller passes those per-block tensors here, not the raw reference
encoder output. The losses are averaged across blocks and operate on
``[batch, reference_token, hidden]`` tensors:

* null_norm: mean squared normalized-null token norm per hidden dimension;
* mean_norm: mean squared per-position batch-mean norm per hidden dimension;
* rank_loss: 1 - participation-ratio/(batch-1), using the batch Gram matrix;
* token_norm_loss: squared error between mean token L2 norm and its target.

All reductions and Gram calculations use float32 outside autocast.
"""
from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor


def participation_ratio(values: Tensor) -> Tensor:
    """Effective rank of batch samples after centering, bounded by batch-1."""
    if values.ndim != 2:
        raise ValueError("values must have shape [batch, features]")
    batch = values.shape[0]
    if batch < 2:
        return values.new_zeros(())
    centred = values - values.mean(dim=0, keepdim=True)
    gram = centred @ centred.T
    trace = torch.diagonal(gram).sum()
    trace_square = gram.square().sum()
    return trace.square() / trace_square.clamp_min(1e-12)


def _one_layer_terms(style: Tensor, null_style: Tensor, mask: Tensor,
                     token_norm_target: float) -> dict[str, Tensor]:
    if style.ndim != 3 or null_style.shape != style.shape:
        raise ValueError("style and null_style must have matching [B, T, D] shapes")
    if mask.shape != style.shape[:2]:
        raise ValueError("mask must have shape [B, T]")
    valid = mask.to(device=style.device, dtype=torch.bool)
    weights = valid.unsqueeze(-1).float()
    count_per_position = weights.sum(dim=0).clamp_min(1.0)
    count = weights.sum().clamp_min(1.0)
    values = style.float()
    nulls = null_style.float()

    width = style.shape[-1]
    null_norm = ((nulls.square().sum(dim=-1, keepdim=True) * weights).sum()
                 / count / width)
    position_means = (values * weights).sum(dim=0) / count_per_position
    active_positions = valid.any(dim=0).float()
    mean_norm = ((position_means.square().sum(dim=-1) * active_positions).sum()
                 / active_positions.sum().clamp_min(1.0) / width)

    flattened = (values * weights).flatten(1)
    rank = participation_ratio(flattened)
    rank_loss = 1.0 - rank / max(style.shape[0] - 1, 1)

    token_norms = values.square().sum(dim=-1).clamp_min(1e-12).sqrt()
    mean_token_norm = (token_norms * valid.float()).sum() / count.squeeze(-1)
    token_norm_loss = (mean_token_norm - token_norm_target).square()

    return {"null_norm": null_norm, "mean_norm": mean_norm,
            "rank_loss": rank_loss, "token_norm_loss": token_norm_loss,
            "participation_ratio": rank.detach(),
            "mean_token_norm": mean_token_norm.detach()}


def style_terms(styles: Sequence[Tensor], null_styles: Sequence[Tensor],
                mask: Tensor, token_norm_target: float = 1.0) -> dict[str, Tensor]:
    """Compute terms on per-block post-LayerNorm style memories and average them."""
    if not styles or len(styles) != len(null_styles):
        raise ValueError("styles and null_styles must be nonempty sequences of equal length")
    if token_norm_target <= 0:
        raise ValueError("token_norm_target must be positive")

    device_type = styles[0].device.type
    with torch.autocast(device_type=device_type, enabled=False):
        per_layer = [_one_layer_terms(style, null_style, mask, token_norm_target)
                     for style, null_style in zip(styles, null_styles)]
        keys = per_layer[0].keys()
        return {key: torch.stack([item[key] for item in per_layer]).mean()
                for key in keys}
