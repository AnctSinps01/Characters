"""Visualise a trained VAE latent: 4-channel channels and an RGBA composite.

    python -m VAE.visualize_latent --checkpoint runs/VAE/p0/best.pt \
        --output runs/VAE/p0/latent --source handwriting

Writes two sheets. ``rgba.png`` shows input, mean-decoded reconstruction and the
four latent channels mapped to R/G/B/A; ``channels.png`` shows each channel as a
greyscale map. Latent values are unbounded reals, so colours are scaled by the
batch min/max of each channel. That mapping is for inspection only: colour has no
semantic meaning and channel order is arbitrary.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from .data import GlyphStore, ReconstructionDataset, spread_indices
from .model import GlyphVAE, VAEConfig


def to_image(grey: np.ndarray, cell: int) -> Image.Image:
    grey = np.clip(grey, 0.0, 1.0)
    return Image.fromarray((grey * 255).round().astype(np.uint8)).resize(
        (cell, cell), Image.Resampling.NEAREST).convert("RGB")


def normalise(channel: np.ndarray) -> np.ndarray:
    low, high = float(channel.min()), float(channel.max())
    return (channel - low) / (high - low) if high > low else np.zeros_like(channel)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/prepared"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", choices=("handwriting", "standard"), default="handwriting")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--cell", type=int, default=128)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = GlyphVAE(VAEConfig(**checkpoint["model_config"])).to(device).eval()
    model.load_state_dict(checkpoint["model"])

    store = GlyphStore(args.data)
    try:
        dataset = ReconstructionDataset(store, split="val", source=args.source)
        indices = spread_indices(len(dataset), args.samples)[:args.samples]
        batch = [dataset[int(i)] for i in indices]
        images = torch.stack([item["image"] for item in batch]).float().to(device) / 127.5 - 1
        with torch.no_grad():
            mean, _ = model.encode(images)
            recon = model.decode(mean).float().cpu()
        latent = mean.float().cpu().numpy()
        inputs = images.cpu().numpy()[:, 0]
        recon = recon.numpy()[:, 0]
    finally:
        store.close()

    args.output.mkdir(parents=True, exist_ok=True)
    cell, gap, header = args.cell, 4, 22
    channels = latent.shape[1]
    scaled = np.stack([normalise(latent[i, c]) for i in range(len(latent)) for c in range(channels)])
    scaled = scaled.reshape(len(latent), channels, *latent.shape[2:])

    def panel(grid, titles, name):
        width = len(titles) * cell + (len(titles) - 1) * gap
        height = header + len(latent) * cell + (len(latent) - 1) * gap
        canvas = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(canvas)
        for column, title in enumerate(titles):
            draw.text((column * (cell + gap), 5), title, fill="black")
        for row in range(len(latent)):
            y = header + row * (cell + gap)
            for column, tile in enumerate(grid[row]):
                canvas.paste(tile, (column * (cell + gap), y))
        canvas.save(args.output / f"{name}.png")
        print(f"wrote {args.output / (name + '.png')}")

    # Input, reconstruction and the RGBA composite.
    rgba_rows = []
    for row in range(len(latent)):
        rgb = np.stack([scaled[row, c] for c in range(min(3, channels))], axis=-1)
        alpha = scaled[row, 3] if channels > 3 else np.ones_like(scaled[row, 0])
        if rgb.shape[-1] == 1:
            rgb = np.repeat(rgb, 3, axis=-1)
        premultiplied = rgb * alpha[..., None] + (1 - alpha[..., None])
        rgba = Image.fromarray((np.clip(premultiplied, 0, 1) * 255).round().astype(np.uint8)).resize(
            (cell, cell), Image.Resampling.NEAREST)
        rgba_rows.append([to_image(inputs[row], cell), to_image(recon[row], cell), rgba])
    panel(rgba_rows, ["Input", "Posterior-mean decode", "Latent as RGBA (A=channel 4)"], "rgba")

    channel_rows = []
    for row in range(len(latent)):
        channel_rows.append([to_image(scaled[row, c], cell) for c in range(channels)])
    panel(channel_rows, [f"channel {c}" for c in range(channels)], "channels")

    stats = {"checkpoint_step": checkpoint["step"], "source": args.source,
             "samples": int(len(latent)),
             "channel_mean": latent.mean(axis=(0, 2, 3)).round(3).tolist(),
             "channel_std": latent.std(axis=(0, 2, 3)).round(3).tolist(),
             "channel_range": [[float(latent[:, c].min()), float(latent[:, c].max())]
                               for c in range(channels)]}
    import json
    (args.output / "latent_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
