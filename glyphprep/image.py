"""Crop, scale and render the grayscale glyph bitmaps stored in the dataset."""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFont


def normalize_image(image: Image.Image, size: int, margin: int) -> np.ndarray:
    """Crop background, preserve aspect ratio, center on a white square."""
    if margin < 0 or size - 2 * margin < 1:
        raise ValueError("margin must leave at least one foreground pixel")
    image = image.convert("L")
    bbox = ImageChops.invert(image).getbbox()
    if bbox is None:
        raise ValueError("blank glyph")
    image = image.crop(bbox)
    scale = (size - 2 * margin) / max(image.size)
    shape = tuple(max(1, round(d * scale)) for d in image.size)
    image = image.resize(shape, Image.Resampling.LANCZOS)
    canvas = Image.new("L", (size, size), 255)
    canvas.paste(image, ((size - shape[0]) // 2, (size - shape[1]) // 2))
    return np.asarray(canvas, dtype=np.uint8)


def render_character(character: str, font: ImageFont.FreeTypeFont,
                     size: int, margin: int) -> np.ndarray:
    left, top, right, bottom = font.getbbox(character)
    if right <= left or bottom <= top:
        raise ValueError(f"empty font glyph: {character!r}")
    canvas = Image.new("L", (right - left, bottom - top), 255)
    ImageDraw.Draw(canvas).text((-left, -top), character, font=font, fill=0)
    return normalize_image(canvas, size, margin)
