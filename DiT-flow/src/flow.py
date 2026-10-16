from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

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
