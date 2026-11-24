"""Does the latent encode stroke structure, or only class identity?

Two views of a frozen encoder:

1. ``interp`` – linear interpolation between the latents of two different
   characters. A structural code morphs strokes; a class code tends to fade
   through grey mush.
2. ``pca`` – sweep the leading principal directions of the latent cloud. If a
   direction changes stroke geometry (position, thickness, slant) it is
   structural; if it only changes overall darkness it is appearance.

    python -m VAE.probe_latent_interp --checkpoint runs/VAE/lat2/last.pt \
        --data data/prepared --output runs/VAE/lat2/latent_interp
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from .data import GlyphStore, ReconstructionDataset, spread_indices
from .model import GlyphVAE, VAEConfig


def tile(image, cell):
    array = ((image.detach().float().cpu().squeeze().clamp(-1, 1) + 1) * 127.5)
    return Image.fromarray(array.numpy().round().astype(np.uint8)).resize(
        (cell, cell), Image.Resampling.NEAREST).convert("RGB")


def grid(rows, labels, path, cell, gap=3, header=22):
    width = len(rows[0]) * cell + (len(rows[0]) - 1) * gap
    height = header + len(rows) * cell + (len(rows) - 1) * gap
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    for column, label in enumerate(labels):
        draw.text((column * (cell + gap), 5), str(label), fill="black")
    for row, tiles in enumerate(rows):
        for column, image in enumerate(tiles):
            canvas.paste(image, (column * (cell + gap), header + row * (cell + gap)))
    canvas.save(path)
    print("wrote", path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/prepared"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cell", type=int, default=128)
    parser.add_argument("--steps", type=int, default=9)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = GlyphVAE(VAEConfig(**checkpoint["model_config"])).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    store = GlyphStore(args.data)
    try:
        dataset = ReconstructionDataset(store, split="val", source="handwriting")
        indices = spread_indices(len(dataset), args.samples)
        images = torch.stack([dataset[int(i)]["image"] for i in indices]).float()
        chars = [dataset[int(i)]["class_index"] for i in indices]
        x = (images.to(device) / 127.5 - 1)
        with torch.no_grad():
            mean, _ = model.encode(x)
            mean = mean.float()
    finally:
        store.close()

    args.output.mkdir(parents=True, exist_ok=True)
    flat = mean.flatten(1).cpu().numpy()

    # 1) Interpolate between four pairs of distinct characters.
    rng = np.random.default_rng(args.seed)
    pairs = []
    while len(pairs) < 4:
        a, b = rng.choice(len(flat), 2, replace=False)
        if chars[a] != chars[b]:
            pairs.append((int(a), int(b)))
    rows, labels = [], []
    alphas = np.linspace(0.0, 1.0, args.steps)
    labels = [f"{a:.2f}" for a in alphas]
    with torch.no_grad():
        for a, b in pairs:
            z = torch.tensor(alphas[:, None] * flat[b][None] + (1 - alphas[:, None]) * flat[a][None],
                             device=device, dtype=torch.float32)
            decoded = model.decode(z.reshape(-1, *mean.shape[1:]))
            rows.append([tile(decoded[i], args.cell) for i in range(len(alphas))])
            print(f"pair {a} -> {b}: chars {chars[a]} -> {chars[b]}")
    grid(rows, labels, args.output / "interp.png", args.cell)

    # 2) Sweep the leading principal directions.
    centred = flat - flat.mean(0, keepdims=True)
    _, _, vh = np.linalg.svd(centred, full_matrices=False)
    std = centred.std(0) + 1e-8
    sweep = np.linspace(-2.0, 2.0, args.steps)
    rows, labels = [], [f"{v:+.1f}σ" for v in sweep]
    mu_mean = flat.mean(0)
    with torch.no_grad():
        for component in range(min(3, vh.shape[0])):
            direction = (vh[component] * std).reshape(mean.shape[1:])
            z = torch.tensor(mu_mean[None] + sweep[:, None] * (vh[component] * std)[None],
                             device=device, dtype=torch.float32)
            decoded = model.decode(z.reshape(-1, *mean.shape[1:]))
            rows.append([tile(decoded[i], args.cell) for i in range(len(sweep))])
            print(f"PC{component + 1}: {rows[-1][0] and ''}{float(np.linalg.norm(direction)):.3f} norm")
    grid(rows, labels, args.output / "pca.png", args.cell)


if __name__ == "__main__":
    main()
