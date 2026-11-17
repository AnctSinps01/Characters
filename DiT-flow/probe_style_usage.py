"""What does the model actually take from its references?

One checkpoint is probed along four axes on the same writer pairs:

    noise        same references, a second noise draw
    writer       same noise, the other writer's references
    zero_style   same noise and content, style tokens replaced by zeros
    truth        the two writers' real handwriting of the target character

Generated images are then compared with the truth on three cheap shape
statistics (ink ratio, stroke thickness, slant), and the same statistics are
computed for the references, so "copies the reference" and "restores the truth"
can be told apart.

    PYTHONPATH=DiT-flow python DiT-flow/probe_style_usage.py \
        --checkpoint runs/DiT-flow/v7-style-dist-fromscratch/best.pt \
        --data data/prepared --pairs 32 --output runs/DiT-flow/style-usage-v7.json
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


def shape_stats(image: torch.Tensor) -> dict:
    """Ink ratio, stroke thickness and slant of a 64x64 image in [-1, 1]."""
    array = image.detach().cpu().numpy()
    while array.ndim > 2:
        array = array[0]
    ink = (array < 0).astype(np.float32)
    ink_ratio = float(ink.mean())
    runs = []
    for row in ink:
        padded = np.concatenate([[0.0], row, [0.0]])
        edges = np.flatnonzero(np.diff(padded))
        if len(edges):
            runs.extend((edges[1::2] - edges[0::2]).tolist())
    thickness = float(np.mean(runs)) if runs else 0.0
    ys, xs = np.nonzero(ink)
    if len(xs) < 8:
        slant = 0.0
    else:
        x = xs - xs.mean()
        y = ys - ys.mean()
        mu20, mu02 = (x * x).mean(), (y * y).mean()
        mu11 = (x * y).mean()
        slant = float(0.5 * np.arctan2(2 * mu11, mu20 - mu02))
    return {"ink": ink_ratio, "thickness": thickness, "slant": slant}


def correlation(xs, ys) -> float:
    if len(xs) < 3:
        return float("nan")
    a, b = np.asarray(xs), np.asarray(ys)
    if a.std() < 1e-9 or b.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=32)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    settings = checkpoint["settings"]
    model = Glyph2Hand(ModelConfig(**settings["model"]))
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    size = settings["model"]["image_size"]
    print(f"checkpoint step={checkpoint.get('step')}", flush=True)

    preloaded = PreloadedGlyphs(args.data, verbose=False)
    bank = preloaded.standards
    writers = [w for w in preloaded.manifest["writers"] if w["split"] == args.split]
    sets = {w["id"]: set(w["labels"]) for w in writers}
    rng = random.Random(args.seed)

    pairs = []
    for i in range(len(writers)):
        for j in range(i + 1, len(writers)):
            shared = sorted(sets[writers[i]["id"]] & sets[writers[j]["id"]])
            if len(shared) >= REFERENCE_COUNT + 1:
                pairs.append((writers[i], writers[j], shared))
    random.Random(args.seed).shuffle(pairs)
    pairs = pairs[:args.pairs]
    print(f"{len(pairs)} writer pairs", flush=True)

    def refs(writer, ids):
        images = preloaded.writer_images(writer["id"])
        position = {c: k for k, c in enumerate(writer["labels"])}
        hands = torch.stack([to_tensor(images[position[c]]) for c in ids])[None].to(device)
        standards = torch.stack([to_tensor(bank[c]) for c in ids])[None].to(device)
        return hands, standards

    mask = torch.ones(1, REFERENCE_COUNT, dtype=torch.bool, device=device)
    rows = []
    generator = torch.Generator(device=device).manual_seed(args.seed)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for writer_a, writer_b, shared in pairs:
            ids, target = shared[:REFERENCE_COUNT], shared[REFERENCE_COUNT]
            hands_a, stds_a = refs(writer_a, ids)
            hands_b, stds_b = refs(writer_b, ids)
            content = model.encode_content(to_tensor(bank[target])[None].to(device))
            style_a = model.encode_style(hands_a, stds_a, mask)
            style_b = model.encode_style(hands_b, stds_b, mask)
            zeros = torch.zeros_like(style_a)
            noise = torch.randn(1, 1, size, size, generator=generator, device=device)
            noise2 = torch.randn(1, 1, size, size, generator=generator, device=device)
            blank = torch.zeros(1, 1, size, size, device=device)

            gen_a = euler(model, content, style_a, mask, args.steps, noise)
            gen_b = euler(model, content, style_b, mask, args.steps, noise)
            gen_noise = euler(model, content, style_a, mask, args.steps, noise2)
            gen_zero = euler(model, content, zeros, mask, args.steps, noise)
            # Null mode: standard glyphs as references.
            std_mode = euler(model, content, model.encode_style(stds_a, stds_a, mask),
                             mask, args.steps, noise)

            images_a = preloaded.writer_images(writer_a["id"])
            images_b = preloaded.writer_images(writer_b["id"])
            pos_a = {c: k for k, c in enumerate(writer_a["labels"])}
            pos_b = {c: k for k, c in enumerate(writer_b["labels"])}
            truth_a = to_tensor(images_a[pos_a[target]])
            truth_b = to_tensor(images_b[pos_b[target]])
            standard = to_tensor(bank[target])

            ga, gb = gen_a.float().cpu(), gen_b.float().cpu()
            gn, gz = gen_noise.float().cpu(), gen_zero.float().cpu()
            gs = std_mode.float().cpu()
            stats = {
                "gen_a": shape_stats(ga), "gen_b": shape_stats(gb),
                "zero": shape_stats(gz), "null_mode": shape_stats(gs),
                "truth_a": shape_stats(truth_a), "truth_b": shape_stats(truth_b),
                "ref_a": shape_stats(hands_a[0, 0].reshape(1, 1, size, size).float().cpu()),
                "ref_b": shape_stats(hands_b[0, 0].reshape(1, 1, size, size).float().cpu()),
            }
            rows.append({
                "noise": mae(ga, gn),
                "writer": mae(ga, gb),
                "zero_style": mae(ga, gz),
                "truth": mae(truth_a, truth_b),
                "gen_to_truth": (mae(ga, truth_a) + mae(gb, truth_b)) / 2,
                "gen_to_standard": (mae(ga, standard) + mae(gb, standard)) / 2,
                "zero_to_truth": mae(gz, truth_a),
                "null_to_standard": mae(gs, standard),
                "attribution": int(mae(ga, truth_a) < mae(ga, truth_b))
                               + int(mae(gb, truth_b) < mae(gb, truth_a)),
                "stats": stats,
            })

    def median(key):
        return statistics.median([row[key] for row in rows])

    noise, writer_gap, truth_gap = median("noise"), median("writer"), median("truth")
    summary = {
        "step": checkpoint.get("step"),
        "pairs": len(rows),
        "disp_noise": noise,
        "disp_writer": writer_gap,
        "disp_truth": truth_gap,
        "ratio_writer": writer_gap / truth_gap,
        "noise_over_writer": noise / writer_gap,
        "zero_style_effect": median("zero_style"),
        "gen_to_truth": median("gen_to_truth"),
        "gen_to_standard": median("gen_to_standard"),
        "zero_to_truth": median("zero_to_truth"),
        "null_to_standard": median("null_to_standard"),
        "attribution": statistics.mean([row["attribution"] / 2 for row in rows]),
    }
    for key in ("ink", "thickness", "slant"):
        gen_a = [row["stats"]["gen_a"][key] for row in rows]
        summary[f"corr_gen_truth_{key}"] = correlation(
            gen_a, [row["stats"]["truth_a"][key] for row in rows])
        summary[f"corr_gen_ref_{key}"] = correlation(
            gen_a, [row["stats"]["ref_a"][key] for row in rows])
        summary[f"corr_gen_writerB_{key}"] = correlation(
            [row["stats"]["gen_a"][key] for row in rows],
            [row["stats"]["truth_b"][key] for row in rows])
        summary[f"{key}_gen"] = statistics.median(gen_a)
        summary[f"{key}_truth"] = statistics.median([row["stats"]["truth_a"][key] for row in rows])
        summary[f"{key}_zero"] = statistics.median([row["stats"]["zero"][key] for row in rows])
        summary[f"{key}_null"] = statistics.median([row["stats"]["null_mode"][key] for row in rows])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary, "rows": rows},
                                      ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
