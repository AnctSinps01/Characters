"""Shared CASIA-HWDB data preparation for every model in this repository.

    python -m glyphprep.prepare --gnt-root data/gnt \
        --font fonts/WenJinMinchoP0-Regular.ttf --output data/prepared

The output directory is the single ``data/prepared`` dataset read by both
DiT-flow (``--data``) and twist (``train-all-pair``).
"""
from .gnt import iter_gnt
from .image import normalize_image, render_character

__all__ = ["iter_gnt", "normalize_image", "render_character"]
