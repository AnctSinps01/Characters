"""Test post-hoc style-token transforms before touching the training objective.

Three candidate offsets are measured on the same writer pairs and the same
noise, so only the conditioning changes:

    raw            style tokens as produced by the reference encoder
    minus_null     s - mean(null tokens)
    minus_writer   s - mean(writer tokens)          (removes the shared direction)
    centred_scaled (s - mean(writer tokens)) / std  (moment normalisation)

Reported per variant: how much of the real inter-writer difference survives
(ratio), whether the output still looks handwritten (ink ratio, MAE to truth)
and how close it drifts to the printed glyph (MAE to standard).

    PYTHONPATH=DiT-flow python DiT-flow/probe_style_transform.py \
        --checkpoint runs/DiT-flow/v4-nullstyle-fromscratch/last.pt \
        --data data/prepared --output runs/DiT-flow/style-transform.json
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
    parser.add_argument("--stats-samples", type=int, default=120)
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
    size = settings["model"]["image_size"]
    print(f"checkpoint step={checkpoint.get('step')}", flush=True)

    preloaded = PreloadedGlyphs(args.data, verbose=False)
    bank = preloaded.standards
    writers = [w for w in preloaded.manifest["writers"] if w["split"] == args.split]
    sets = {w["id"]: set(w["labels"]) for w in writers}
    rng = random.Random(args.seed)

    def refs_of(writer, ids):
        images = preloaded.writer_images(writer["id"])
        position = {c: k for k, c in enumerate(writer["labels"])}
        hands = torch.stack([to_tensor(images[position[c]]) for c in ids])[None].to(device)
        standards = torch.stack([to_tensor(bank[c]) for c in ids])[None].to(device)
        return hands, standards

    mask = torch.ones(1, REFERENCE_COUNT, dtype=torch.bool, device=device)

    # ---- statistics over a broad sample of writers -------------------------
    writer_tokens, null_tokens = [], []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(args.stats_samples):
            writer = rng.choice(writers)
            ids = rng.sample(writer["labels"], REFERENCE_COUNT)
            hands, standards = refs_of(writer, ids)
            writer_tokens.append(model.encode_style(hands, standards, mask).float()[0])
            null_tokens.append(model.encode_style(standards, standards, mask).float()[0])
    writer_tokens = torch.stack(writer_tokens)          # [S, Z*N, D]
    null_tokens = torch.stack(null_tokens)
    mean_null = null_tokens.mean(0)
    mean_writer = writer_tokens.mean(0)
    std_writer = writer_tokens.std(0).clamp_min(1e-4)
    print(f"mean null token norm {mean_null.norm():.3f}; "
          f"mean writer token norm {mean_writer.norm():.3f}", flush=True)

    transforms = {
        "raw": lambda s: s,
        "minus_null": lambda s: s - mean_null,
        "minus_writer": lambda s: s - mean_writer,
        "centred_scaled": lambda s: (s - mean_writer) / std_writer,
    }

    # ---- writer pairs ------------------------------------------------------
    pairs = []
    for i in range(len(writers)):
        for j in range(i + 1, len(writers)):
            shared = sorted(sets[writers[i]["id"]] & sets[writers[j]["id"]])
            if len(shared) >= REFERENCE_COUNT + 1:
                pairs.append((writers[i], writers[j], shared))
    random.Random(args.seed).shuffle(pairs)
    pairs = pairs[:args.pairs]
    print(f"{len(pairs)} writer pairs", flush=True)

    contents, styles_a, styles_b, noises, truths = [], [], [], [], []
    generator = torch.Generator(device=device).manual_seed(args.seed)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for writer_a, writer_b, shared in pairs:
            refs_ids, target = shared[:REFERENCE_COUNT], shared[REFERENCE_COUNT]
            hands_a, stds_a = refs_of(writer_a, refs_ids)
            hands_b, stds_b = refs_of(writer_b, refs_ids)
            contents.append(model.encode_content(to_tensor(bank[target])[None].to(device)))
            styles_a.append(model.encode_style(hands_a, stds_a, mask).float()[0])
            styles_b.append(model.encode_style(hands_b, stds_b, mask).float()[0])
            noises.append(torch.randn(1, 1, size, size, generator=generator, device=device))
            images_a = preloaded.writer_images(writer_a["id"])
            images_b = preloaded.writer_images(writer_b["id"])
            pos_a = {c: k for k, c in enumerate(writer_a["labels"])}
            pos_b = {c: k for k, c in enumerate(writer_b["labels"])}
            truths.append((to_tensor(images_a[pos_a[target]]), to_tensor(images_b[pos_b[target]]),
                           to_tensor(bank[target])))

    report = {}
    for name, transform in transforms.items():
        rows = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for index in range(len(pairs)):
                pair_styles = torch.stack([transform(styles_a[index]), transform(styles_b[index])])
                generated = euler(model, contents[index].expand(2, -1, -1), pair_styles,
                                  mask.expand(2, -1), args.steps, noises[index].expand(2, -1, -1, -1))
                gen_a, gen_b = generated[0].float().cpu(), generated[1].float().cpu()
                truth_a, truth_b, standard = truths[index]
                ink_gen = (gen_a < 0).float().mean().item()
                ink_truth = (truth_a < 0).float().mean().item()
                rows.append({
                    "disp_gen": mae(gen_a, gen_b),
                    "disp_truth": mae(truth_a, truth_b),
                    "mae_truth": (mae(gen_a, truth_a) + mae(gen_b, truth_b)) / 2,
                    "mae_standard": (mae(gen_a, standard) + mae(gen_b, standard)) / 2,
                    "ink_ratio": ink_gen / max(ink_truth, 1e-3),
                })
        summary = {
            "ratio_median": statistics.median([r["disp_gen"] / r["disp_truth"] for r in rows]),
            "disp_gen_median": statistics.median([r["disp_gen"] for r in rows]),
            "mae_truth": statistics.median([r["mae_truth"] for r in rows]),
            "mae_standard": statistics.median([r["mae_standard"] for r in rows]),
            "ink_ratio": statistics.median([r["ink_ratio"] for r in rows]),
        }
        report[name] = summary
        print(f"{name:15s} ratio={summary['ratio_median']:.3f}  "
              f"disp_gen={summary['disp_gen_median']:5.2f}  "
              f"MAE->truth={summary['mae_truth']:5.2f}  "
              f"MAE->standard={summary['mae_standard']:6.2f}  "
              f"ink_ratio={summary['ink_ratio']:.2f}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"step": checkpoint.get("step"), "report": report}, indent=2),
                           encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
