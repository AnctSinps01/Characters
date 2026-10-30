"""Core model components for the handwriting generation project."""
import sys
from pathlib import Path

# The shared data preparation package (glyphprep/) lives at the repository
# root, one level above DiT-flow/.  Appending it keeps "python -m src.train"
# working from both the repository root and DiT-flow/.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.append(str(_REPO_ROOT))

from .config import ModelConfig, TrainingConfig
from .flow import (
    FlowMatchingBatch,
    build_flow_matching_batch,
    flow_matching_loss,
    sample_euler,
)
from .model import Glyph2Hand

__all__ = [
    "FlowMatchingBatch",
    "Glyph2Hand",
    "ModelConfig",
    "TrainingConfig",
    "build_flow_matching_batch",
    "flow_matching_loss",
    "sample_euler",
]
