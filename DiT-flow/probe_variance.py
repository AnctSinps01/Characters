"""Decompose the variance of generated glyphs: style vs. initial noise.

For a set of (writer A, writer B, shared character) episodes, generate the same
target with writer A's references and 4 noise seeds, and with writer B's
references and the same 4 seeds.

  within   mean pairwise distance among the 4 samples that share references
           -> spread caused by the initial noise
  between  distance between the two reference-conditioned sample means
           -> shift caused by swapping the writer
  truths   distance between the two writers' real handwriting of that character
"""
from __future__ import annotations

import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from src import Glyph2Hand, ModelConfig
from src.data import PreloadedGlyphs
from src.flow import sample_euler

CHECKPOINT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("runs/DiT-flow/v2/best.pt")
DATA = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/prepared")
EPISODES = int(sys.argv[3]) if len(sys.argv) > 3 else 16
NOISES = 4


def tensor(image) -> torch.Tensor:
    return torch.tensor(np.asarray(image)[None], dtype=torch.float32) / 127.5 - 1


def mae(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).abs().mean().item() * 127.5


def main() -> None:
    device = torch.device("cuda")
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model = Glyph2Hand(ModelConfig(**checkpoint["settings"]["model"]))
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()

    preloaded = PreloadedGlyphs(DATA)
    bank = preloaded.standards
    writers = [w for w in preloaded.manifest["writers"] if w["split"] == "train"]
    sets = {w["id"]: set(w["labels"]) for w in writers}

    rng = random.Random(11)
    pairs = []
    for i in range(len(writers)):
        for j in range(i + 1, len(writers)):
            shared = sorted(sets[writers[i]["id"]] & sets[writers[j]["id"]])
            if len(shared) >= 9:
                pairs.append((writers[i], writers[j], shared))
    rng.shuffle(pairs)
    pairs = pairs[:EPISODES]

    keys = ("within", "between", "truths", "mean_a_to_truth_a", "mean_a_to_truth_b",
            "sample_a_to_truth_a", "sample_a_to_truth_b")
    totals = dict.fromkeys(keys, 0.0)
    rows = []
    with torch.no_grad():
        for number, (writer_a, writer_b, shared) in enumerate(pairs, 1):
            refs_ids, target = shared[:4], shared[8]

            def build(writer):
                images = preloaded.writer_images(writer["id"])
                position = {c: k for k, c in enumerate(writer["labels"])}
                hands = torch.stack([tensor(images[position[c]]) for c in refs_ids])
                standards = torch.stack([tensor(bank[c]) for c in refs_ids])
                return hands, standards, tensor(images[position[target]])

            hands_a, std_a, truth_a = build(writer_a)
            hands_b, std_b, truth_b = build(writer_b)
            target_std = tensor(bank[target])

            references = torch.stack([hands_a] * NOISES + [hands_b] * NOISES)
            standards = torch.stack([std_a] * NOISES + [std_b] * NOISES)
            targets = target_std.repeat(NOISES * 2, 1, 1, 1)
            generator = torch.Generator(device=device).manual_seed(1234)
            noise = torch.randn(NOISES * 2, 1, 64, 64, generator=generator, device=device)
            generated = sample_euler(
                model, targets.to(device), references.to(device), standards.to(device),
                steps=32, noise=noise).float().cpu()

            samples_a, samples_b = generated[:NOISES], generated[NOISES:]
            within = 0.0
            for group in (samples_a, samples_b):
                within += sum(mae(group[i], group[j])
                              for i in range(NOISES) for j in range(i + 1, NOISES))
            within /= NOISES * (NOISES - 1)

            mean_a, mean_b = samples_a.mean(0, keepdim=True), samples_b.mean(0, keepdim=True)
            values = {
                "within": within,
                "between": mae(mean_a, mean_b),
                "truths": mae(truth_a, truth_b),
                "mean_a_to_truth_a": mae(mean_a, truth_a),
                "mean_a_to_truth_b": mae(mean_a, truth_b),
                "sample_a_to_truth_a": float(np.mean([mae(s, truth_a) for s in samples_a])),
                "sample_a_to_truth_b": float(np.mean([mae(s, truth_b) for s in samples_a])),
            }
            for key in keys:
                totals[key] += values[key]
            rows.append(values)
            print(f"[{number}/{len(pairs)}] within={values['within']:.1f} "
                  f"between={values['between']:.1f} truths={values['truths']:.1f} "
                  f"meanA->A={values['mean_a_to_truth_a']:.1f} "
                  f"meanA->B={values['mean_a_to_truth_b']:.1f}", flush=True)

    report = {key: value / len(pairs) for key, value in totals.items()}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
