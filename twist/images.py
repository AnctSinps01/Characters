"""Per-image intensity normalization shared by training data loaders."""
from __future__ import annotations

import torch
from torch import Tensor


NORMALIZATION = "per-image-minmax-v1"


class ConstantImageError(ValueError):
    def __init__(self, indices):
        self.indices = indices
        super().__init__(f"images {indices} have no intensity variation (max == min)")


def normalize_minmax(images: Tensor) -> Tensor:
    """Map each resized [B,C,H,W] image independently to [-1,1].

    Dark ink is -1 and light background is +1. A constant image has no
    usable shape and must be excluded, rather than assigned fake foreground.
    """
    if images.ndim != 4 or images.numel() == 0:
        raise ValueError("images must be a nonempty [B,C,H,W] tensor")
    values = images.to(dtype=torch.float32)
    minimum = values.amin(dim=(1, 2, 3), keepdim=True)
    maximum = values.amax(dim=(1, 2, 3), keepdim=True)
    constant = (minimum == maximum).flatten()
    if bool(constant.any()):
        raise ConstantImageError(constant.nonzero().flatten().tolist())
    return (values - minimum) / (maximum - minimum) * 2 - 1
