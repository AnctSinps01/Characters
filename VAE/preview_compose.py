"""Render preview sheets for a blocked VAE checkpoint.

Two sheets are written: reconstruction (input vs decoded) and recombination
(content block from one sample, author block from another).  Works on any
checkpoint saved by ``VAE.train`` or ``VAE.finetune_compose`` because both keep
``model_config`` and ``model``.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from VAE.data import GlyphStore
from VAE.model import GlyphVAE, VAEConfig


def to_uint8(tensor):
    array = ((tensor.detach().cpu().squeeze().clamp(-1, 1) + 1) * 127.5).numpy()
    return array.round().astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", default="data/prepared")
    parser.add_argument("--output", required=True)
    parser.add_argument("--cells", type=int, default=6)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=21)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = GlyphVAE(VAEConfig(**checkpoint["model_config"])).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    training = (checkpoint.get("source_training_config") or checkpoint.get("training_config"))
    if training is None:
        # Fine-tuned checkpoints keep the split only in the sibling run.json.
        sidecar = Path(args.checkpoint).resolve().parent / "run.json"
        if sidecar.exists():
            import json
            training = json.loads(sidecar.read_text(encoding="utf-8"))["source_training_config"]
    if training is None:
        raise ValueError("checkpoint has no content/author channel split")
    content = slice(0, training["content_channels"])
    author = slice(training["content_channels"],
                   training["content_channels"] + training["author_channels"])
    free = slice(author.stop, checkpoint["model_config"]["latent_channels"])

    store = GlyphStore(args.data)
    writers = [w for w in store.writers if w["split"] == "val"]
    common = sorted(set.intersection(*[set(w["labels"]) for w in writers]))
    rng = np.random.default_rng(args.seed)
    n = args.cells
    chars = [int(c) for c in rng.choice(len(common), n, replace=False)]
    authors = [int(a) for a in rng.choice(len(writers), n, replace=False)]

    def sample(writer_index, char_index):
        writer = writers[writer_index]
        row = {k: i for i, k in enumerate(writer["labels"])}[common[char_index]]
        image = np.array(store.writer_images(writer)[row]).copy()
        x = torch.from_numpy(image)[None, None].float().to(device) / 127.5 - 1
        with torch.no_grad():
            mu, _ = model.encode(x)
        return mu[0], image

    cell, gap, head = 128, 8, 26
    out = Path(args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    # reconstruction sheet
    sheet = Image.new("RGB", (2 * (cell + gap), head + n * (cell + gap)), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((4, 7), f"reconstruction  step {checkpoint.get('step')}", fill="black")
    with torch.no_grad():
        for i in range(n):
            writer_index, char_index = authors[i], chars[i]
            mu, image = sample(writer_index, char_index)
            decoded = model.decode(mu[None]).float()[0]
            y = head + i * (cell + gap)
            for j, pixels in enumerate((image, to_uint8(decoded))):
                sheet.paste(Image.fromarray(pixels).resize((cell, cell), Image.Resampling.NEAREST),
                            (j * (cell + gap), y))
            draw.text((2, y + cell + 2), f"w{writer_index} c{char_index}", fill=(90, 90, 90))
    sheet.save(out.parent / f"{out.stem}_recon.png")

    # recombination sheet: rows characters, columns writers, free block random
    style_char = int(rng.choice([c for c in range(len(common)) if c not in chars]))
    grid = Image.new("RGB", ((n + 1) * (cell + gap), head + (n + 1) * (cell + gap)), "white")
    draw = ImageDraw.Draw(grid)
    draw.text((4, 7), f"recombination  step {checkpoint.get('step')}", fill="black")
    for j, writer_index in enumerate(authors):
        _, image = sample(writer_index, style_char)
        grid.paste(Image.fromarray(image).resize((cell, cell), Image.Resampling.NEAREST),
                   ((j + 1) * (cell + gap), head))
        draw.text(((j + 1) * (cell + gap) + 2, head + cell + 2), f"w{writer_index}", fill=(90, 90, 90))
    with torch.no_grad():
        for i, char_index in enumerate(chars):
            zc, image = sample(authors[0], char_index)
            y = head + (i + 1) * (cell + gap)
            grid.paste(Image.fromarray(image).resize((cell, cell), Image.Resampling.NEAREST), (0, y))
            draw.text((2, y + cell + 2), f"c{char_index}", fill=(90, 90, 90))
            for j, writer_index in enumerate(authors):
                za, _ = sample(writer_index, style_char)
                zf, _ = sample(int(rng.integers(len(writers))), int(rng.integers(len(common))))
                mixed = torch.cat([zc[content], za[author], zf[free]])
                decoded = model.decode(mixed[None]).float()[0]
                grid.paste(Image.fromarray(to_uint8(decoded)).resize((cell, cell), Image.Resampling.NEAREST),
                           ((j + 1) * (cell + gap), y))
    grid.save(out.parent / f"{out.stem}_compose.png")
    store.close()
    print(f"wrote {out.parent / (out.stem + '_recon.png')} and {out.parent / (out.stem + '_compose.png')}")


if __name__ == "__main__":
    main()
