"""Measure how far the generated glyph moves when the style tokens are swapped.

For many writer pairs, both writers share the same four reference characters and
the same target character, so the only difference is the style tokens.  The
script reports, per pair:

    disp_gen    MAE between the two generations (same noise, same content)
    disp_truth  MAE between the two writers' real handwriting of that character
    ratio       disp_gen / disp_truth  -- how much of the real style gap survives

    PYTHONPATH=DiT-flow python DiT-flow/probe_style_amplitude.py \
        --checkpoint runs/DiT-flow/v4-nullstyle-fromscratch/last.pt \
        --data data/prepared --pairs 40 --output runs/DiT-flow/style-amplitude.json
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

from src import Glyph2Hand, ModelConfig
from src.data import PreloadedGlyphs

REFERENCE_COUNT = 4


def to_tensor(array) -> torch.Tensor:
    return torch.tensor(np.asarray(array)[None], dtype=torch.float32) / 127.5 - 1


def euler(model, content, style, mask, steps, noise):
    state = noise.clone()
    for index in range(steps):
        times = torch.full((state.shape[0],), index / steps, device=state.device, dtype=torch.float32)
        state = state + model.predict_velocity(state, times, content, style, mask) / steps
    return state.clamp(-1, 1)


def mae(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).abs().mean().item() * 127.5


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=40)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

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

    pairs = []
    for i in range(len(writers)):
        for j in range(i + 1, len(writers)):
            shared = sorted(sets[writers[i]["id"]] & sets[writers[j]["id"]])
            if len(shared) >= REFERENCE_COUNT + 1:
                pairs.append((writers[i], writers[j], shared))
    random.Random(args.seed).shuffle(pairs)
    pairs = pairs[:args.pairs]
    print(f"{len(pairs)} writer pairs", flush=True)

    def refs_of(writer, ids):
        images = preloaded.writer_images(writer["id"])
        position = {c: k for k, c in enumerate(writer["labels"])}
        hands = torch.stack([to_tensor(images[position[c]]) for c in ids]).to(device)
        standards = torch.stack([to_tensor(bank[c]) for c in ids]).to(device)
        return hands, standards

    contents, styles, masks, noises = [], [], [], []
    truth_records = []
    generator = torch.Generator(device=device).manual_seed(args.seed)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for writer_a, writer_b, shared in pairs:
            refs_ids, target = shared[:REFERENCE_COUNT], shared[REFERENCE_COUNT]
            hands_a, stds_a = refs_of(writer_a, refs_ids)
            hands_b, stds_b = refs_of(writer_b, refs_ids)
            mask = torch.ones(1, REFERENCE_COUNT, dtype=torch.bool, device=device)
            content = model.encode_content(to_tensor(bank[target])[None].to(device))
            style_a = model.encode_style(hands_a[None], stds_a[None], mask)
            style_b = model.encode_style(hands_b[None], stds_b[None], mask)
            noise = torch.randn(1, 1, settings["model"]["image_size"], settings["model"]["image_size"],
                                generator=generator, device=device)
            contents += [content, content]
            styles += [style_a, style_b]
            masks += [mask, mask]
            noises += [noise, noise]
            images_a = preloaded.writer_images(writer_a["id"])
            images_b = preloaded.writer_images(writer_b["id"])
            pos_a = {c: k for k, c in enumerate(writer_a["labels"])}
            pos_b = {c: k for k, c in enumerate(writer_b["labels"])}
            truth_records.append((to_tensor(images_a[pos_a[target]]).to(device),
                                  to_tensor(images_b[pos_b[target]]).to(device)))

        generated = euler(model, torch.cat(contents), torch.cat(styles),
                          torch.cat(masks), args.steps, torch.cat(noises)).float().cpu()

    rows = []
    for index, (writer_a, writer_b, shared) in enumerate(pairs):
        gen_a, gen_b = generated[2 * index], generated[2 * index + 1]
        truth_a, truth_b = truth_records[index]
        disp_gen = mae(gen_a, gen_b)
        disp_truth = mae(truth_a.cpu(), truth_b.cpu())
        rows.append({"pair": f"{writer_a['id']}/{writer_b['id']}",
                     "disp_gen": disp_gen, "disp_truth": disp_truth,
                     "ratio": disp_gen / disp_truth if disp_truth > 1e-6 else None})

    ratios = [r["ratio"] for r in rows if r["ratio"] is not None]
    gen_values = [r["disp_gen"] for r in rows]
    truth_values = [r["disp_truth"] for r in rows]

    def summary(values):
        values = sorted(values)
        count = len(values)
        return {"median": statistics.median(values),
                "p25": values[count // 4], "p75": values[(3 * count) // 4],
                "min": values[0], "max": values[-1]}

    report = {"step": checkpoint.get("step"), "pairs": len(rows),
              "disp_gen": summary(gen_values), "disp_truth": summary(truth_values),
              "ratio": summary(ratios), "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps({key: report[key] for key in ("pairs", "disp_gen", "disp_truth", "ratio")},
                     indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
