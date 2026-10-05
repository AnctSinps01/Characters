"""Inspect unscaled gradients before clipping or changing optimizer state."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn


def json_number(value: float) -> float | str:
    """Keep diagnostic JSON valid even when the quantity being diagnosed is NaN."""
    return value if math.isfinite(value) else str(value)


@dataclass
class GradientSnapshot:
    names: list[str]
    gradients: list[Tensor]
    norms: Tensor
    total: Tensor
    used_float64: bool

    @classmethod
    @torch.no_grad()
    def capture(cls, model: nn.Module) -> GradientSnapshot:
        items = [(name, parameter.grad.detach())
                 for name, parameter in model.named_parameters()
                 if parameter.grad is not None]
        if not items:
            return cls([], [], torch.empty(0), torch.tensor(0.0), False)
        names, gradients = map(list, zip(*items))
        norms = torch.stack([torch.linalg.vector_norm(gradient) for gradient in gradients])
        total = torch.linalg.vector_norm(norms)
        used_float64 = not bool(torch.isfinite(total))
        if used_float64:
            # Squaring large but finite FP32 gradients can overflow the norm
            # reduction. Recompute BEFORE clipping; multiplying by zero first
            # would lose finite gradients and turn infinite ones into NaNs.
            norms = torch.stack([
                torch.linalg.vector_norm(gradient, dtype=torch.float64)
                for gradient in gradients
            ])
            total = torch.linalg.vector_norm(norms)
        return cls(names, gradients, norms, total, used_float64)

    @torch.no_grad()
    def clip(self, max_norm: float) -> None:
        if not math.isfinite(max_norm) or max_norm <= 0:
            raise ValueError("max_norm must be finite and positive")
        if not bool(torch.isfinite(self.total)):
            raise FloatingPointError("cannot clip non-finite gradients")
        coefficient = (max_norm / (self.total + 1e-6)).clamp(max=1.0)
        for gradient in self.gradients:
            gradient.mul_(coefficient.to(dtype=gradient.dtype))

    @torch.no_grad()
    def report(self, max_norm: float, top_k: int = 8) -> dict:
        """Describe raw gradients; call before clip() to retain peak values."""
        values = self.norms.cpu().tolist()
        total = self.total.item()
        modules = {}
        for name, value in zip(self.names, values):
            module = name.rpartition(".")[0] or name
            modules[module] = math.hypot(modules.get(module, 0.0), value)

        def rank(value):
            return value if math.isfinite(value) else math.inf

        largest = sorted(range(len(values)), key=lambda i: rank(values[i]), reverse=True)[:top_k]
        nonfinite = [name for name, value in zip(self.names, values) if not math.isfinite(value)]
        return {
            "event": "gradients" if math.isfinite(total) else "nonfinite_gradients",
            "grad_norm": json_number(total),
            "norm_float64_fallback": self.used_float64,
            "clip_scale": min(1.0, max_norm / (total + 1e-6)) if math.isfinite(total) else None,
            "top_parameters": [
                {"name": self.names[i], "norm": json_number(values[i]),
                 "max_abs": json_number(self.gradients[i].abs().max().item())}
                for i in largest
            ],
            "top_modules": [
                {"name": name, "norm": json_number(value)}
                for name, value in sorted(modules.items(), key=lambda item: rank(item[1]),
                                          reverse=True)[:top_k]
            ],
            "nonfinite_parameter_count": len(nonfinite),
            "nonfinite_parameters": nonfinite[:top_k],
        }
