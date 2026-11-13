"""Interpolate style tokens and render the resulting gradient.

Style tokens come from ``encode_style``; blending two writers' tokens with

    style(alpha) = (1 - alpha) * style_A + alpha * style_B

shows whether the style space is linear.  A second sheet blends a writer's
tokens with the *null* tokens produced by feeding the standard glyphs as both
reference handwriting and reference standard, which the v4 run was trained on.

    PYTHONPATH=DiT-flow python DiT-flow/infer_style_interp.py \
        --checkpoint runs/DiT-flow/v4-nullstyle-fromscratch/last.pt \
        --data data/prepared --output runs/DiT-flow/style-interp
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from PIL import Image, ImageDraw

from src import Glyph2Hand, ModelConfig
from src.data import PreloadedGlyphs

REFERENCE_COUNT = 4


def to_tensor(array) -> torch.Tensor:
    return torch.tensor(np.asarray(array)[None], dtype=torch.float32) / 127.5 - 1


def to_tile(tensor: torch.Tensor, cell: int) -> Image.Image:
    array = ((tensor.detach().float().cpu().squeeze(0).squeeze(0).clamp(-1, 1) + 1) * 127.5)
    image = Image.fromarray(array.round().byte().numpy())
    return image.resize((cell, cell), Image.Resampling.NEAREST)


def euler(model, content, style, mask, steps, noise):
    state = noise.clone()
    batch = state.shape[0]
    for index in range(steps):
        times = torch.full((batch,), index / steps, device=state.device, dtype=torch.float32)
        velocity = model.predict_velocity(state, times, content, style, mask)
        state = state + velocity / steps
    return state.clamp(-1, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--targets", type=int, default=4, help="target characters per sheet")
    parser.add_argument("--alphas", type=int, default=9)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--cell", type=int, default=128)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    settings = checkpoint["settings"]
    model = Glyph2Hand(ModelConfig(**settings["model"]))
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    print(f"checkpoint step={checkpoint.get('step')}", flush=True)

    preloaded = PreloadedGlyphs(args.data, verbose=False)
    bank = preloaded.standards
    writers = [w for w in preloaded.manifest["writers"] if w["split"] == args.split]
    sets = {w["id"]: set(w["labels"]) for w in writers}

    rng = random.Random(args.seed)
    pair = None
    for i in range(len(writers)):
        for j in range(i + 1, len(writers)):
            shared = sorted(sets[writers[i]["id"]] & sets[writers[j]["id"]])
            if len(shared) >= REFERENCE_COUNT + args.targets:
                pair = (writers[i], writers[j], shared)
                break
        if pair:
            break
    if pair is None:
        raise SystemExit("no writer pair shares enough characters")
    writer_a, writer_b, shared = pair
    refs_ids = shared[:REFERENCE_COUNT]
    targets_ids = shared[REFERENCE_COUNT:REFERENCE_COUNT + args.targets]
    print(f"writers {writer_a['id']} / {writer_b['id']}, {len(targets_ids)} target characters",
          flush=True)

    def references(writer):
        images = preloaded.writer_images(writer["id"])
        position = {c: k for k, c in enumerate(writer["labels"])}
        hands = torch.stack([to_tensor(images[position[c]]) for c in refs_ids]).to(device)
        standards = torch.stack([to_tensor(bank[c]) for c in refs_ids]).to(device)
        return hands, standards

    hands_a, stds_a = references(writer_a)
    hands_b, stds_b = references(writer_b)
    mask = torch.ones(1, REFERENCE_COUNT, dtype=torch.bool, device=device)

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        style_a = model.encode_style(hands_a[None], stds_a[None], mask)
        style_b = model.encode_style(hands_b[None], stds_b[None], mask)
        # Null style: standard glyphs as both reference handwriting and standard.
        style_null = model.encode_style(stds_a[None], stds_a[None], mask)
        contents = [model.encode_content(to_tensor(bank[c])[None].to(device)) for c in targets_ids]

        alphas = [i / (args.alphas - 1) for i in range(args.alphas)]
        generator = torch.Generator(device=device).manual_seed(args.seed)
        noises = [torch.randn(1, 1, settings["model"]["image_size"], settings["model"]["image_size"],
                              generator=generator, device=device) for _ in targets_ids]

        def sheet(style_start, style_end, name, start_label, end_label):
            rows = []
            for content, noise in zip(contents, noises):
                tiles = []
                for alpha in alphas:
                    style = (1 - alpha) * style_start + alpha * style_end
                    generated = euler(model, content, style, mask, args.steps, noise)
                    tiles.append(to_tile(generated, args.cell))
                rows.append(tiles)
            gap, header = 3, 22
            width = len(alphas) * args.cell + (len(alphas) - 1) * gap
            height = header + len(rows) * args.cell + (len(rows) - 1) * gap
            canvas = Image.new("RGB", (width, height), "white")
            draw = ImageDraw.Draw(canvas)
            for column, alpha in enumerate(alphas):
                draw.text((column * (args.cell + gap), 5), f"{alpha:.2f}", fill="black")
            for row, tiles in enumerate(rows):
                top = header + row * (args.cell + gap)
                for column, tile in enumerate(tiles):
                    canvas.paste(tile, (column * (args.cell + gap), top))
            canvas.save(args.output / f"{name}.png")
            print(f"wrote {args.output / (name + '.png')}  ({start_label} -> {end_label})", flush=True)

        sheet(style_a, style_b, "interp_writer_a_to_b", writer_a["id"], writer_b["id"])
        sheet(style_a, style_null, "interp_writer_a_to_printed", writer_a["id"], "printed")


if __name__ == "__main__":
    main()
