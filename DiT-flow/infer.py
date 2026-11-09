"""Sample characters from a trained DiT-flow checkpoint and save contact sheets.

Each row is one generated character.  The layout depends on --ref-mode:

  handwriting  the four reference handwriting glyphs, their four standard glyphs,
               the target standard, the true handwriting and one generation
  standard     the four standard glyphs reused as *reference handwriting* (out of
               distribution: the model only saw handwritten references in
               training), the target standard, the truth and the generation
  both         the first two layouts side by side, sharing the same noise

    cd Characters
    PYTHONPATH=DiT-flow ~/envs/jiya_2044/bin/python DiT-flow/infer.py \
        --checkpoint runs/DiT-flow/v2/best.pt --data data/prepared --count 64
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from PIL import Image, ImageDraw

from src import Glyph2Hand, ModelConfig
from src.data import HandwritingEpisodes, PreloadedGlyphs
from src.flow import sample_euler

REFERENCE_COUNT = 4


def to_tile(tensor, cell: int) -> Image.Image:
    array = ((tensor.detach().float().cpu().squeeze(0).clamp(-1, 1) + 1) * 127.5)
    image = Image.fromarray(array.round().byte().numpy())
    return image.resize((cell, cell), Image.Resampling.NEAREST)


def columns_for(mode: str) -> list[str]:
    refs = [f"Ref {i + 1}" for i in range(REFERENCE_COUNT)]
    standards = [f"Std {i + 1}" for i in range(REFERENCE_COUNT)]
    if mode == "handwriting":
        return refs + standards + ["Target", "Truth", "Gen"]
    if mode == "standard":
        return standards + ["Target", "Truth", "Gen(std ref)"]
    return refs + standards + ["Target", "Truth", "Gen(hw ref)", "Gen(std ref)"]


def render_sheet(rows, labels, path: Path, cell: int, gap: int, header: int) -> None:
    width = len(labels) * cell + (len(labels) - 1) * gap
    height = header + len(rows) * cell + (len(rows) - 1) * gap
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    for column, label in enumerate(labels):
        draw.text((column * (cell + gap), 4), label, fill="black")
    for row, tiles in enumerate(rows):
        top = header + row * (cell + gap)
        for column, tile in enumerate(tiles):
            canvas.paste(tile, (column * (cell + gap), top))
    canvas.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="default: <checkpoint dir>/infer")
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--rows", type=int, default=16, help="rows per sheet")
    parser.add_argument("--ref-mode", choices=("handwriting", "standard", "both"),
                        default="handwriting",
                        help="'standard' feeds the standard glyphs as reference handwriting")
    parser.add_argument("--cell", type=int, default=128,
                        help="pixels per glyph; must be a whole multiple of the model image size, "
                             "otherwise nearest-neighbour upscaling gives uneven pixel blocks")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else ("cpu" if args.device == "auto" else args.device))
    output = args.output or (args.checkpoint.parent / "infer")
    output.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    settings = checkpoint["settings"]
    model = Glyph2Hand(ModelConfig(**settings["model"]))
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    step = checkpoint.get("step")
    size = settings["model"]["image_size"]
    if args.cell < size or args.cell % size:
        parser.error(f"--cell must be a positive multiple of the image size {size} "
                     f"(got {args.cell}); non-integer nearest-neighbour scaling shows up as "
                     f"uneven pixel blocks")
    print(f"checkpoint step={step} device={device} model={settings['model']['hidden_size']}d "
          f"scale={args.cell // size}x ref-mode={args.ref_mode}", flush=True)

    preloaded = PreloadedGlyphs(args.data, verbose=True)
    dataset = HandwritingEpisodes(args.data, args.split, length=args.count,
                                  min_references=4, max_references=4,
                                  seed=args.seed, preloaded=preloaded)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    labels = columns_for(args.ref_mode)

    sheet_rows = []
    sheet_index = 1
    done = 0
    context = (torch.autocast("cuda", dtype=torch.bfloat16)
               if device.type == "cuda" else nullcontext())
    with torch.no_grad():
        for raw in loader:
            batch = {key: value.to(device) for key, value in raw.items()}
            standard = batch["target_standard"]
            mask = batch["reference_mask"]
            generated_hand = generated_standard = None
            with context:
                if args.ref_mode in ("handwriting", "both"):
                    generated_hand = sample_euler(
                        model, standard, batch["reference_handwriting"],
                        batch["reference_standard"], steps=args.steps, reference_mask=mask)
                if args.ref_mode in ("standard", "both"):
                    generated_standard = sample_euler(
                        model, standard, batch["reference_standard"],
                        batch["reference_standard"], steps=args.steps, reference_mask=mask)
            for index in range(standard.shape[0]):
                tiles = []
                if args.ref_mode in ("handwriting", "both"):
                    tiles += [to_tile(t, args.cell) for t in batch["reference_handwriting"][index]]
                if args.ref_mode == "standard":
                    tiles += [to_tile(t, args.cell) for t in batch["reference_standard"][index]]
                else:
                    tiles += [to_tile(t, args.cell) for t in batch["reference_standard"][index]]
                tiles += [to_tile(standard[index], args.cell),
                          to_tile(batch["target_handwriting"][index], args.cell)]
                if generated_hand is not None:
                    tiles.append(to_tile(generated_hand[index], args.cell))
                if generated_standard is not None:
                    tiles.append(to_tile(generated_standard[index], args.cell))
                sheet_rows.append(tiles)
            done += standard.shape[0]
            print(f"generated {done}/{args.count}", flush=True)
            while len(sheet_rows) >= args.rows:
                path = output / f"sheet_{sheet_index:02d}.png"
                render_sheet(sheet_rows[:args.rows], labels, path, args.cell, 3, 20)
                print(f"wrote {path}", flush=True)
                sheet_rows = sheet_rows[args.rows:]
                sheet_index += 1
    if sheet_rows:
        path = output / f"sheet_{sheet_index:02d}.png"
        render_sheet(sheet_rows, labels, path, args.cell, 3, 20)
        print(f"wrote {path}", flush=True)
    print(f"done: {done} characters, step={step}, output={output}")


if __name__ == "__main__":
    main()
