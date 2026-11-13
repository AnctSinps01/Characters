"""Distribution of style-token magnitudes across writers.

For every writer, sample several reference sets of four characters and record

    pooled_norm   L2 norm of the mean-pooled style tokens   (one vector per set)
    token_norm    mean per-token L2 norm

The null style (standard glyphs as both reference handwriting and standard) is
measured on the same reference sets, which shows how far the "printed" mode sits
from the writer manifold in magnitude.

    PYTHONPATH=DiT-flow python DiT-flow/probe_style_norm.py \
        --checkpoint runs/DiT-flow/v4-nullstyle-fromscratch/last.pt \
        --data data/prepared --output runs/DiT-flow/style-norm
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import statistics
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


def quantiles(values):
    values = sorted(values)
    count = len(values)
    return {"median": statistics.median(values),
            "p25": values[count // 4],
            "p75": values[(3 * count) // 4],
            "min": values[0],
            "max": values[-1]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "all"), default="val")
    parser.add_argument("--samples", type=int, default=8, help="reference sets per writer")
    parser.add_argument("--seed", type=int, default=11)
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
    writers = [w for w in preloaded.manifest["writers"]
               if args.split == "all" or w["split"] == args.split]
    rng = random.Random(args.seed)

    rows = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for writer in writers:
            images = preloaded.writer_images(writer["id"])
            position = {c: k for k, c in enumerate(writer["labels"])}
            pooled, tokens, nulls = [], [], []
            for _ in range(args.samples):
                ids = rng.sample(writer["labels"], REFERENCE_COUNT)
                hands = torch.stack([to_tensor(images[position[c]]) for c in ids])[None].to(device)
                stds = torch.stack([to_tensor(bank[c]) for c in ids])[None].to(device)
                mask = torch.ones(1, REFERENCE_COUNT, dtype=torch.bool, device=device)
                style = model.encode_style(hands, stds, mask).float()
                null_style = model.encode_style(stds, stds, mask).float()
                pooled.append(style.mean(1).norm(dim=-1).item())
                tokens.append(style.norm(dim=-1).mean().item())
                nulls.append(null_style.mean(1).norm(dim=-1).item())
            rows.append({"writer": writer["id"], "split": writer["split"],
                         "pooled": pooled, "token": tokens, "null": nulls})

    per_writer = [{"writer": r["writer"], "split": r["split"],
                   "pooled_median": statistics.median(r["pooled"]),
                   "token_median": statistics.median(r["token"]),
                   "null_median": statistics.median(r["null"])} for r in rows]

    pooled_all = [v for r in rows for v in r["pooled"]]
    token_all = [v for r in rows for v in r["token"]]
    null_all = [v for r in rows for v in r["null"]]
    within = statistics.median([statistics.pstdev(r["pooled"]) for r in rows])
    between = statistics.pstdev([p["pooled_median"] for p in per_writer])
    writer_medians = [p["pooled_median"] for p in per_writer]
    null_medians = [p["null_median"] for p in per_writer]

    report = {
        "step": checkpoint.get("step"),
        "writers": len(rows),
        "samples_per_writer": args.samples,
        "pooled_norm": quantiles(pooled_all),
        "token_norm": quantiles(token_all),
        "null_pooled_norm": quantiles(null_all),
        "within_writer_sd": within,
        "between_writer_sd": between,
        "per_writer": per_writer,
    }
    (args.output / "style_norm.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    order = sorted(per_writer, key=lambda p: p["pooled_median"])
    width, height = max(600, 24 * len(order)), 360
    canvas = Image.new("RGB", (width, height + 40), "white")
    draw = ImageDraw.Draw(canvas)
    top = max(p["pooled_median"] for p in order) * 1.1
    scale = (height - 40) / top
    bar = (width - 60) / max(1, len(order))
    for index, item in enumerate(order):
        x = 30 + index * bar
        h = item["pooled_median"] * scale
        colour = (70, 110, 200) if item["split"] == "train" else (200, 120, 60)
        draw.rectangle([x, height - h, x + bar * 0.7, height], fill=colour)
    null_median = statistics.median(null_all)
    y_null = height - null_median * scale
    for x in range(0, width, 8):
        draw.line([x, y_null, x + 4, y_null], fill=(200, 0, 0))
    draw.text((8, 6), f"style pooled-norm per writer (median of {args.samples}); "
                      f"writers {min(writer_medians):.2f}..{max(writer_medians):.2f}, "
                      f"null-style median {null_median:.2f} (red)", fill="black")
    canvas.save(args.output / "style_norm.png")

    print(json.dumps({key: report[key] for key in
                      ("writers", "pooled_norm", "token_norm", "null_pooled_norm",
                       "within_writer_sd", "between_writer_sd")}, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
