"""Render original / generated / reconstruction sheets from a flow checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from VAE.data import GlyphStore
from VAE.flow.data import LatentSet, LatentStats
from VAE.flow.dit import DiTConfig, DiTFlow
from VAE.flow.dit import sample as dit_sample
from VAE.flow.model import FlowConfig, FlowModel
from VAE.flow.model import sample as conv_sample
from VAE.model import GlyphVAE, VAEConfig


def build(flow_config):
    if "depth" in flow_config:
        return DiTFlow(DiTConfig(**flow_config)), dit_sample
    return FlowModel(FlowConfig(**flow_config)), conv_sample


def pick_rows(latents, count, seed, subset):
    """Pick rows with distinct characters, optionally restricted to a memorised subset."""
    rng = np.random.default_rng(0)
    pool = (rng.choice(len(latents), subset, replace=False) if subset
            else np.arange(len(latents)))
    order = np.random.default_rng(seed).permutation(len(pool))
    picked, seen = [], set()
    for position in order:
        row = int(pool[position])
        character = int(latents.label[row])
        if character in seen:
            continue
        seen.add(character)
        picked.append(row)
        if len(picked) == count:
            break
    return picked


def original_pixels(store, latents, row):
    writer_index = int(latents.owner[row])
    info = latents.ranges[writer_index]
    writer = store.writers_by_split[latents.split][writer_index]
    offset = row - info["start"]
    return np.array(store.writer_images(writer)[offset]).copy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--latents", required=True)
    parser.add_argument("--vae", required=True)
    parser.add_argument("--data", default="data/prepared")
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--subset", type=int, default=0,
                        help="restrict rows to the first N rows of a memorised overfit subset")
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--refs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model, sampler = build(checkpoint["flow_config"])
    model = model.to(device).eval()
    model.load_state_dict(checkpoint["model"])
    vae_ckpt = torch.load(args.vae, map_location="cpu", weights_only=False)
    vae = GlyphVAE(VAEConfig(**vae_ckpt["model_config"])).to(device).eval()
    vae.load_state_dict(vae_ckpt["model"])

    latents = LatentSet(args.latents, args.split, in_memory=True)
    stats = LatentStats.from_latent(LatentSet(args.latents, "train", in_memory=True).latent)
    mean, std = stats.mean.to(device), stats.std.to(device)
    channels = model.config.target_channels

    store = GlyphStore(args.data)
    store.writers_by_split = {"train": [w for w in store.writers if w["split"] == "train"],
                              "val": [w for w in store.writers if w["split"] == "val"]}
    latents.split = args.split

    picked = pick_rows(latents, args.count, args.seed, args.subset)
    rng = np.random.default_rng(args.seed)
    standard = torch.from_numpy(np.asarray(latents.standards[
        [int(latents.label[i]) for i in picked]], np.float32))[:, :4].to(device)
    target = torch.from_numpy(np.asarray(latents.latent[picked], np.float32)).to(device)
    refs, counts = [], []
    for row in picked:
        rows = latents.rows[int(latents.owner[row])]
        others = rows[np.asarray(latents.label[rows]) != int(latents.label[row])]
        take = others[rng.choice(len(others), size=min(args.refs, len(others)), replace=False)]
        refs.append(torch.from_numpy(np.asarray(latents.latent[take], np.float32))[:, 2:3])
        counts.append(len(take))
    widest = max(counts)
    padded = torch.zeros(len(refs), widest, 1, 8, 8)
    for i, item in enumerate(refs):
        padded[i, :len(item)] = item
    counts_t = torch.tensor(counts, device=device)
    refs_in = (padded.to(device) - mean[2:3].view(1, 1, 1, 1)) / std[2:3].view(1, 1, 1, 1)
    std_in = (standard - mean[:4].view(1, 4, 1, 1)) / std[:4].view(1, 4, 1, 1)

    with torch.no_grad():
        generated = sampler(model, refs_in, std_in, steps=args.steps, counts=counts_t)
        generated = generated * std[:channels].view(1, channels, 1, 1) + mean[:channels].view(1, channels, 1, 1)
        if channels < 4:
            generated = torch.cat([generated, target[:, 3:4]], dim=1)
        image_gen = vae.decode(generated).float()
        image_rec = vae.decode(target).float()

    cell = 128
    sheet = Image.new("RGB", (3 * (cell + 6), 20 + args.count * (cell + 6)), "white")
    draw = ImageDraw.Draw(sheet)
    for column, title in enumerate(("original", "flow", "reconstruction")):
        draw.text((column * (cell + 6) + 2, 5), title, fill="black")
    to_uint8 = lambda t: ((t.detach().cpu().squeeze().clamp(-1, 1) + 1) * 127.5).numpy().round().astype(np.uint8)
    for index, row in enumerate(picked):
        tiles = [original_pixels(store, latents, row), to_uint8(image_gen[index]), to_uint8(image_rec[index])]
        for column, tile in enumerate(tiles):
            image = Image.fromarray(tile).resize((cell, cell), Image.Resampling.NEAREST)
            sheet.paste(image, (column * (cell + 6), 20 + index * (cell + 6)))
    out = Path(args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    mse = (generated - target[:, :channels]).square().mean().item()
    print(f"wrote {out} | latent mse {mse:.5f}")


if __name__ == "__main__":
    main()
