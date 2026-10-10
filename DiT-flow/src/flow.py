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
    target_handwriting: Tensor,
    *,
    noise: Tensor | None = None,
    times: Tensor | None = None,
) -> FlowMatchingBatch:
    """Build a straight flow from standard Gaussian noise to handwriting."""

    if target_handwriting.ndim != 4:
        raise ValueError("target images must have shape [B, C, H, W]")
    if noise is None:
        noise = torch.randn_like(target_handwriting)
    if noise.shape != target_handwriting.shape:
        raise ValueError("noise must match the target image shape")
    if times is None:
        times = torch.rand(
            target_handwriting.shape[0],
            device=target_handwriting.device,
            dtype=target_handwriting.dtype,
        )
    if times.shape != (target_handwriting.shape[0],):
        raise ValueError("times must have shape [B]")

    initial_state = noise
    broadcast_times = times.reshape(-1, 1, 1, 1)
    state = (1 - broadcast_times) * initial_state + broadcast_times * target_handwriting
    target_velocity = target_handwriting - initial_state
    return FlowMatchingBatch(state, times, target_velocity, initial_state)


def flow_matching_loss(predicted_velocity: Tensor, target_velocity: Tensor) -> Tensor:
    if predicted_velocity.shape != target_velocity.shape:
        raise ValueError("predicted and target velocities must have equal shapes")
    return (predicted_velocity - target_velocity).square().flatten(1).mean(dim=1).mean()


def author_style_loss(
    descriptions: Tensor,
    author_ids: Tensor,
    *,
    reference_mask: Tensor | None = None,
    same_author_weight: float = 1.0,
    different_author_weight: float = 1.0,
) -> Tensor:
    """Compare corresponding Z slots, averaged over each episode's valid references.

    Input is [B, N, Z, D].  The masked reference mean is used only for this
    loss; the generator still reads all N*Z descriptions.  Pairing is inside
    the current batch, excludes self-pairs, and includes both directions.
    Same authors minimize 1-cosine; different authors minimize cosine**2.
    Empty pair types contribute a differentiable zero. No cross-batch memory.
    """

    if descriptions.ndim != 4 or min(descriptions.shape[1:]) < 1:
        raise ValueError("descriptions must have shape [B, N, Z, D]")
    batch, references, _, _ = descriptions.shape
    author_ids = torch.as_tensor(author_ids, device=descriptions.device)
    if author_ids.shape != (batch,):
        raise ValueError("author_ids must have shape [B]")
    for name, value in (("same_author_weight", same_author_weight),
                        ("different_author_weight", different_author_weight)):
        if not torch.isfinite(torch.as_tensor(value)) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")

    if reference_mask is None:
        reference_mask = torch.ones(
            batch, references, dtype=torch.bool, device=descriptions.device
        )
    if reference_mask.shape != (batch, references):
        raise ValueError("reference_mask must have shape [B, N]")
    reference_mask = reference_mask.to(device=descriptions.device, dtype=torch.bool)
    if (~reference_mask.any(dim=1)).any():
        raise ValueError("each sample needs at least one valid reference")
    # Keep both the reduction and cosine comparison in FP32 under autocast.
    with torch.autocast(device_type=descriptions.device.type, enabled=False):
        weights = reference_mask[:, :, None, None].float()
        summary = (descriptions.float() * weights).sum(dim=1) / weights.sum(dim=1)
        normalized = F.normalize(summary, dim=-1, eps=1e-8)
        similarities = torch.einsum("bzd,czd->bcz", normalized, normalized)
    pair_mask = ~torch.eye(
        batch, dtype=torch.bool, device=descriptions.device
    )
    same_mask = pair_mask & author_ids[:, None].eq(author_ids[None, :])
    different_mask = pair_mask & ~author_ids[:, None].eq(author_ids[None, :])

    zero = descriptions.float().sum() * 0.0
    same = (1.0 - similarities)[same_mask].mean() if same_mask.any() else zero
    different = similarities.square()[different_mask].mean() if different_mask.any() else zero
    return same_author_weight * same + different_author_weight * different


@torch.no_grad()
def sample_euler(
    model: Glyph2Hand,
    target_standard: Tensor,
    reference_handwriting: Tensor,
    *,
    steps: int,
    noise: Tensor | None = None,
    reference_mask: Tensor | None = None,
) -> Tensor:
    """Integrate from pure noise, caching content and unpooled style descriptions."""

    if steps <= 0:
        raise ValueError("steps must be positive")
    if noise is None:
        noise = torch.randn_like(target_standard)
    if noise.shape != target_standard.shape:
        raise ValueError("noise must match target_standard")

    content = model.encode_content(target_standard)
    style = model.encode_style(reference_handwriting, reference_mask)
    state = noise
    step_size = 1.0 / steps
    batch = target_standard.shape[0]
    for index in range(steps):
        times = torch.full(
            (batch,),
            index * step_size,
            device=state.device,
            dtype=state.dtype,
        )
        velocity = model.predict_velocity(state, times, content, style, reference_mask)
        state = state + step_size * velocity
    return state.clamp(-1, 1)
