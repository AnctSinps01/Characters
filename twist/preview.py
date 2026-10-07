"""Shared endpoint, stage, trajectory and sampling-grid previews."""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

from .geometry import compose_displacements, identity_grid


def pixels(tensor):
    return ((tensor.detach().float().cpu()[0, 0].clamp(-1, 1) + 1) * 127.5).round().byte().numpy()


def tile(image, label, size=256):
    if isinstance(image, torch.Tensor):
        image = Image.fromarray(pixels(image))
    image = image.convert("RGB").resize((size, size), Image.Resampling.NEAREST)
    canvas = Image.new("RGB", (size, size + 24), "white")
    canvas.paste(image, (0, 24))
    ImageDraw.Draw(canvas).text((6, 6), label, fill="black")
    return canvas


def strip(items):
    canvas = Image.new("RGB", (sum(image.width for image in items), max(image.height for image in items)), "white")
    offset = 0
    for image in items:
        canvas.paste(image, (offset, 0))
        offset += image.width
    return canvas


@torch.no_grad()
def save_preview(result, target, output: Path, step: int, frames: int, split="train"):
    if type(frames) is not int or frames < 2:
        raise ValueError("GIF needs an integer frame count >= 2")
    output.mkdir(parents=True, exist_ok=True)
    stem = output / f"step_{step:07d}_{split}"
    ink_prediction = (255 - pixels(result.image)).astype(np.float32) / 255
    ink_target = (255 - pixels(target)).astype(np.float32) / 255
    overlay = np.stack((1 - ink_target, 1 - ink_prediction, 1 - ink_prediction), axis=-1)
    overlay = Image.fromarray((overlay * 255).round().astype(np.uint8))
    strip([tile(result.source, "A"), tile(target, "B"), tile(result.image, "fit at t=1"),
           tile(overlay, "red=fit; cyan=B; black=both")]).save(f"{stem}_compare.png")
    strip([tile(result.source, "A")] + [tile(image, f"stage {i+1}")
          for i, image in enumerate(result.stage_images)] + [tile(target, "B")]).save(f"{stem}_stages.png")
    strip([tile(result.render(t).image, f"t={t:.2f}") for t in (0, 0.25, 0.5, 0.75, 1)]).save(f"{stem}_trajectory.png")
    # Unique labels retain the configured count even when images are identical.
    animation = [tile(result.render(float(t)).image, f"t={t:.3f}  frame {index+1}/{frames}")
                 for index, t in enumerate(torch.linspace(0, 1, frames))]
    durations = [80] * frames
    durations[0] = durations[-1] = 600
    animation[0].save(f"{stem}_trajectory.gif", save_all=True, append_images=animation[1:],
                      duration=durations, loop=0)
    nodes = identity_grid(1, (33, 33), device=result.grid.device, dtype=result.grid.dtype)
    grid = compose_displacements(result.displacements, nodes)[0].cpu().numpy() * 255
    grid_image = Image.new("RGB", (256, 256), "white")
    draw = ImageDraw.Draw(grid_image)
    for i in range(0, 33, 2):
        draw.line([tuple(point) for point in grid[i]], fill=(40, 90, 180), width=1)
        draw.line([tuple(point) for point in grid[:, i]], fill=(40, 90, 180), width=1)
    tile(grid_image, "backward sampling grid").save(f"{stem}_grid.png")
