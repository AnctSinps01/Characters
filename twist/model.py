"""Sequential registration, sampling the original A through accumulated maps."""
from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn

from .config import ModelConfig
from .geometry import composed_grid, sample_image
from .stage import RegistrationStage, validate_pair


@dataclass(frozen=True)
class WarpResult:
    image: Tensor
    grid: Tensor                    # [B,H,W,2], normalized (x,y) source coordinates.


@dataclass(frozen=True)
class RegistrationResult:
    source: Tensor
    displacements: tuple[Tensor, ...]  # [B,2,Hc,Wc] for each sequential stage.
    stage_images: tuple[Tensor, ...]
    stage_grids: tuple[Tensor, ...]

    @property
    def image(self) -> Tensor:
        return self.stage_images[-1]

    @property
    def grid(self) -> Tensor:
        return self.stage_grids[-1]

    def render(self, time: float | Tensor) -> WarpResult:
        """Query the cached trajectory without another neural-network forward."""
        grid = composed_grid(self.displacements, self.source, time)
        return WarpResult(sample_image(self.source, grid), grid)


class Twist(nn.Module):
    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        count = 1 if self.config.share_stage_weights else self.config.num_stages
        self.stages = nn.ModuleList(RegistrationStage(self.config) for _ in range(count))

    def forward(self, source: Tensor, target: Tensor) -> RegistrationResult:
        """Fit at t=1; later stages see the preceding stage's actual warped A.

        Gradients propagate through both the sampling maps and the subsequent
        stage inputs. No stage output is detached, and B's image stays fixed.
        """
        validate_pair(source, target, self.config)
        displacements, images, grids = [], [], []
        current_source = source
        for index in range(self.config.num_stages):
            stage = self.stages[0] if self.config.share_stage_weights else self.stages[index]
            displacements.append(stage(current_source, target))
            grid = composed_grid(displacements, source)
            image = sample_image(source, grid)
            current_source = image.to(dtype=source.dtype)
            images.append(image)
            grids.append(grid)
        return RegistrationResult(source, tuple(displacements), tuple(images), tuple(grids))
