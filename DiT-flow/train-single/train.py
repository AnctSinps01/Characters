"""Run single-author fitting through the shared DiT-flow training loop."""
from __future__ import annotations

from pathlib import Path
from runpy import run_path
import sys

# This entry point also works from Characters without changing directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.train import main


if __name__ == "__main__":
    local_config = run_path(str(Path(__file__).with_name("config.py")))
    main(single_writer=True, subset_size=local_config["SUBSET_SIZE"])
