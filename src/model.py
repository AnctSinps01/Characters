from __future__ import annotations

from torch import Tensor, nn

from .config import ModelConfig
from .generator import ConditionalDiT
from .reference_encoder import ReferenceEncoder


class Glyph2Hand(nn.Module):
    """Reference-conditioned velocity model; the target glyph only seeds the flow."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.reference_encoder = ReferenceEncoder(config)
        self.generator = ConditionalDiT(config)

    def encode_style(
        self,
        reference_handwriting: Tensor,
        reference_standard: Tensor,
        reference_mask: Tensor | None = None,
    ) -> Tensor:
        return self.reference_encoder(
            reference_handwriting, reference_standard, reference_mask
        )

    def predict_velocity(
        self,
        state: Tensor,
        times: Tensor,
        style: Tensor,
    ) -> Tensor:
        return self.generator(state, times, style)

    def forward(
        self,
        state: Tensor,
        times: Tensor,
        reference_handwriting: Tensor,
        reference_standard: Tensor,
        reference_mask: Tensor | None = None,
    ) -> Tensor:
        style = self.encode_style(
            reference_handwriting, reference_standard, reference_mask
        )
        return self.predict_velocity(state, times, style)
