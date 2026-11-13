"""Generate characters that never appear in the prepared dataset.

The content encoder only ever saw the 3755 glyphs listed in the manifest, so
this is a content-generalisation test: the target standard glyph is rendered
from the font, while the four references come from that writer's other
characters.  There is no ground truth, so the sheets are for visual inspection.

    PYTHONPATH=DiT-flow python DiT-flow/infer_novel.py \
        --checkpoint runs/DiT-flow/v2-base-flowmse/best.pt --data data/prepared \
        --count 64 --output runs/DiT-flow/novel
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont

from src import Glyph2Hand, ModelConfig
from src.data import PreloadedGlyphs
from src.flow import sample_euler
from glyphprep.image import render_character

REFERENCE_COUNT = 4
COLUMNS = ([f"Ref {i + 1}" for i in range(REFERENCE_COUNT)]
           + [f"Std {i + 1}" for i in range(REFERENCE_COUNT)] + ["Target", "Gen"])


def to_tensor(array, size: int) -> torch.Tensor:
    image = Image.fromarray(array) if array.ndim == 2 else array
    return torch.tensor(np.asarray(image)[None], dtype=torch.float32) / 127.5 - 1


def to_tile(tensor: torch.Tensor, cell: int) -> Image.Image:
    array = ((tensor.detach().float().cpu().squeeze(0).clamp(-1, 1) + 1) * 127.5)
    image = Image.fromarray(array.round().byte().numpy())
    return image.resize((cell, cell), Image.Resampling.NEAREST)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--font", type=Path, default=ROOT / "fonts" / "WenJinMinchoP0-Regular.ttf")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--rows", type=int, default=16)
    parser.add_argument("--cell", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    settings = checkpoint["settings"]
    model = Glyph2Hand(ModelConfig(**settings["model"]))
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    image_size = settings["model"]["image_size"]
    print(f"checkpoint step={checkpoint.get('step')} size={image_size}", flush=True)

    preloaded = PreloadedGlyphs(args.data, verbose=False)
    manifest = preloaded.manifest
    known = set(manifest["characters"])

    with TTFont(str(args.font)) as metadata:
        supported = set(metadata.getBestCmap() or {})
    novel = [chr(code) for code in range(0x4E00, 0x9FA6)
             if code in supported and chr(code) not in known]
    random.Random(args.seed).shuffle(novel)
    novel = novel[:args.count]
    print(f"{len(novel)} characters outside the dataset; e.g. {''.join(novel[:8])}", flush=True)

    font = ImageFont.truetype(str(args.font), image_size * 4)
    rng = random.Random(args.seed + 1)
    writers = [w for w in manifest["writers"] if w["split"] == args.split]
    bank = preloaded.standards

    rows = []
    sheet_index = 1
    for start in range(0, len(novel), args.batch_size):
        chunk = novel[start:start + args.batch_size]
        references, standards, targets = [], [], []
        for character in chunk:
            writer = rng.choice(writers)
            ids = writer["labels"]
            position = {label: index for index, label in enumerate(ids)}
            images = preloaded.writer_images(writer["id"])
            refs = rng.sample(ids, REFERENCE_COUNT)
            references.append(torch.stack([to_tensor(np.asarray(images[position[i]]), image_size)
                                           for i in refs]))
            standards.append(torch.stack([to_tensor(bank[i], image_size) for i in refs]))
            targets.append(to_tensor(render_character(character, font, image_size, 4), image_size))
        refs = torch.stack(references).to(device)
        stds = torch.stack(standards).to(device)
        targets = torch.stack(targets).to(device)
        context = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
        with torch.no_grad(), context:
            generated = sample_euler(model, targets, refs, stds, steps=args.steps)

        for index, character in enumerate(chunk):
            tiles = [to_tile(t, args.cell) for t in refs[index].cpu()]
            tiles += [to_tile(t, args.cell) for t in stds[index].cpu()]
            tiles += [to_tile(targets[index].cpu(), args.cell), to_tile(generated[index].cpu(), args.cell)]
            rows.append((character, tiles))
        print(f"generated {min(start + args.batch_size, len(novel))}/{len(novel)}", flush=True)

    labels = COLUMNS
    gap, header = 3, 20
    width = len(labels) * args.cell + (len(labels) - 1) * gap
    for sheet, offset in enumerate(range(0, len(rows), args.rows), start=1):
        take = rows[offset:offset + args.rows]
        height = header + len(take) * args.cell + (len(take) - 1) * gap
        canvas = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(canvas)
        for column, label in enumerate(labels):
            draw.text((column * (args.cell + gap), 4), label, fill="black")
        for row, (_, tiles) in enumerate(take):
            top = header + row * (args.cell + gap)
            for column, tile in enumerate(tiles):
                canvas.paste(tile, (column * (args.cell + gap), top))
        canvas.save(args.output / f"sheet_{sheet:02d}.png")

    (args.output / "characters.txt").write_text("\n".join(character for character, _ in rows),
                                                encoding="utf-8")
    print(f"wrote {len(rows)} rows to {args.output}", flush=True)


if __name__ == "__main__":
    main()
