"""Shared GIF trajectory previews."""
from pathlib import Path

from PIL import Image, ImageDraw
import torch


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


@torch.no_grad()
def save_preview(result, output: Path, step: int, frames: int, split="train"):
    if type(frames) is not int or frames < 2:
        raise ValueError("GIF needs an integer frame count >= 2")
    output.mkdir(parents=True, exist_ok=True)
    stem = output / f"step_{step:07d}_{split}"
    # Unique labels retain the configured count even when images are identical.
    animation = [tile(result.render(float(t)).image, f"t={t:.3f}  frame {index+1}/{frames}")
                 for index, t in enumerate(torch.linspace(0, 1, frames))]
    durations = [80] * frames
    durations[0] = durations[-1] = 600
    animation[0].save(f"{stem}_trajectory.gif", save_all=True, append_images=animation[1:],
                      duration=durations, loop=0)
