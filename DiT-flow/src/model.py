from __future__ import annotations

from torch import Tensor, nn

from .config import ModelConfig
from .content_encoder import ContentEncoder
from .generator import ConditionalDiT
from .reference_encoder import ReferenceEncoder


class Glyph2Hand(nn.Module):
    """Pure-noise flow conditioned on standard-glyph content and handwriting style."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.reference_encoder = ReferenceEncoder(config)
        self.content_encoder = ContentEncoder(config)
        self.generator = ConditionalDiT(config)

    def encode_style(
        self,
        reference_handwriting: Tensor,
        reference_mask: Tensor | None = None,
    ) -> Tensor:
        return self.reference_encoder(reference_handwriting, reference_mask)

    def encode_content(self, target_standard: Tensor) -> Tensor:
        return self.content_encoder(target_standard)

    def predict_velocity(
        self,
        state: Tensor,
        times: Tensor,
        content: Tensor,
        style: Tensor,
        reference_mask: Tensor | None = None,
    ) -> Tensor:
        style_mask = None
        if reference_mask is not None:
            if reference_mask.ndim != 2:
                raise ValueError("reference_mask must have shape [B, N]")
            style_mask = reference_mask.repeat_interleave(
                self.config.description_tokens, dim=1
            )
        return self.generator(state, times, content, style, style_mask)

    def forward(
        self,
        state: Tensor,
        times: Tensor,
        target_standard: Tensor,
        reference_handwriting: Tensor,
        reference_mask: Tensor | None = None,
    ) -> Tensor:
        style = self.encode_style(reference_handwriting, reference_mask)
        content = self.encode_content(target_standard)
        return self.predict_velocity(state, times, content, style, reference_mask)
