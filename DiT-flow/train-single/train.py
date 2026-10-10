"""Run single-author fitting through the shared DiT-flow training loop."""
from __future__ import annotations

from pathlib import Path
import sys

# This entry point also works from Characters without changing directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.train import main


if __name__ == "__main__":
    main(single_writer=True)
