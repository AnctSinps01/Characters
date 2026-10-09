from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from .model import Glyph2Hand


@dataclass(frozen=True, slots=True)
class FlowMatchingBatch:
    state: Tensor
    times: Tensor
    target_velocity: Tensor
    initial_state: Tensor


def build_flow_matching_batch(
    target_standard: Tensor,
    target_handwriting: Tensor,
    sigma: float,
    *,
    noise: Tensor | None = None,
    times: Tensor | None = None,
) -> FlowMatchingBatch:
    """Build ``x_0``, ``x_t`` and the constant target velocity from the document."""

    if target_standard.shape != target_handwriting.shape:
        raise ValueError("target_standard and target_handwriting must have equal shapes")
    if target_standard.ndim != 4:
        raise ValueError("target images must have shape [B, C, H, W]")
    if sigma < 0:
        raise ValueError("sigma cannot be negative")
    if noise is None:
        noise = torch.randn_like(target_standard)
    if noise.shape != target_standard.shape:
        raise ValueError("noise must match the target image shape")
    if times is None:
        times = torch.rand(
            target_standard.shape[0],
            device=target_standard.device,
            dtype=target_standard.dtype,
        )
    if times.shape != (target_standard.shape[0],):
        raise ValueError("times must have shape [B]")

    initial_state = target_standard + sigma * noise
    broadcast_times = times.reshape(-1, 1, 1, 1)
    state = (1 - broadcast_times) * initial_state + broadcast_times * target_handwriting
    target_velocity = target_handwriting - initial_state
    return FlowMatchingBatch(state, times, target_velocity, initial_state)


def flow_matching_loss(predicted_velocity: Tensor, target_velocity: Tensor) -> Tensor:
    if predicted_velocity.shape != target_velocity.shape:
        raise ValueError("predicted and target velocities must have equal shapes")
    return (predicted_velocity - target_velocity).square().flatten(1).mean(dim=1).mean()


def author_style_loss(
    style_tokens: Tensor,
    author_ids: Tensor,
    *,
    same_author_weight: float = 1.0,
    different_author_weight: float = 1.0,
) -> Tensor:
    """Match style-token slots within an author and separate different authors.

    ``style_tokens`` has shape ``[B, M, D]``.  Each fixed token slot is
    compared with a normalized dot product, so the loss is invariant to the
    overall scale of a style representation while retaining the learned slot
    correspondence.  Same-author pairs minimize ``1 - cosine``; different
    authors minimize ``cosine**2``, which makes their corresponding token
    directions orthogonal instead of treating the opposite direction as a
    valid solution.

    A batch may contain only one example of an author.  Missing pair types
    contribute zero without producing NaNs, and the zero remains connected to
    the graph so the returned loss is always differentiable.
    """

    if style_tokens.ndim != 3:
        raise ValueError("style_tokens must have shape [B, M, D]")
    author_ids = torch.as_tensor(author_ids, device=style_tokens.device)
    if author_ids.shape != (style_tokens.shape[0],):
        raise ValueError("author_ids must have shape [B]")
    for name, value in (("same_author_weight", same_author_weight),
                        ("different_author_weight", different_author_weight)):
        if not torch.isfinite(torch.as_tensor(value)) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")

    # Keep the comparison in float32 under bf16/fp16 autocast.  Normalization
    # is per fixed token slot, not over the whole matrix, so token positions
    # retain their meaning in the generator.
    normalized = F.normalize(style_tokens.float(), dim=-1, eps=1e-8)
    similarities = torch.einsum("bmd,cmd->bcm", normalized, normalized)
    pair_mask = ~torch.eye(
        style_tokens.shape[0], dtype=torch.bool, device=style_tokens.device
    )
    same_mask = pair_mask & author_ids[:, None].eq(author_ids[None, :])
    different_mask = pair_mask & ~author_ids[:, None].eq(author_ids[None, :])

    zero = style_tokens.float().sum() * 0.0
    same = (1.0 - similarities)[same_mask].mean() if same_mask.any() else zero
    different = similarities.square()[different_mask].mean() if different_mask.any() else zero
    return same_author_weight * same + different_author_weight * different


@torch.no_grad()
def sample_euler(
    model: Glyph2Hand,
    target_standard: Tensor,
    reference_handwriting: Tensor,
    reference_standard: Tensor,
    *,
    steps: int,
    sigma: float | None = None,
    noise: Tensor | None = None,
    reference_mask: Tensor | None = None,
) -> Tensor:
    """Integrate from target_standard + noise; only state/time/style reach the model."""

    if steps <= 0:
        raise ValueError("steps must be positive")
    sigma = model.config.sigma if sigma is None else sigma
    if sigma < 0:
        raise ValueError("sigma cannot be negative")
    if noise is None:
        noise = torch.randn_like(target_standard)
    if noise.shape != target_standard.shape:
        raise ValueError("noise must match target_standard")

    style = model.encode_style(
        reference_handwriting, reference_standard, reference_mask
    )
    state = target_standard + sigma * noise
    step_size = 1.0 / steps
    batch = target_standard.shape[0]
    for index in range(steps):
        times = torch.full(
            (batch,),
            index * step_size,
            device=state.device,
            dtype=state.dtype,
        )
        velocity = model.predict_velocity(state, times, style)
        state = state + step_size * velocity
    return state.clamp(-1, 1)
