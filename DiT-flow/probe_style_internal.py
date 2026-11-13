"""How much variation is there *inside* one set of style tokens?

Each sample produces ``N * Z`` description tokens (4 references x 24).  If those
tokens are nearly parallel, keeping all of them is wasted computation and the
generator's cross-attention is reading the same vector 96 times.

Reported per sample, aggregated over samples:

    cosine_offdiag   mean pairwise cosine between the 96 tokens
    within_ref       mean cosine between tokens of the same reference
    cross_ref        mean cosine between tokens of different references
    effective_rank   exp(entropy of normalised singular values)
    top1/top4/top8   share of variance on the leading components

    PYTHONPATH=DiT-flow python DiT-flow/probe_style_internal.py \
        --checkpoint runs/DiT-flow/v4-nullstyle-fromscratch/last.pt \
        --data data/prepared --output runs/DiT-flow/style-internal.json
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


def summarise(tokens: torch.Tensor, per_reference: int) -> dict:
    """tokens: [N*Z, D] float tensor on CPU."""
    normalised = torch.nn.functional.normalize(tokens, dim=-1)
    cosine = normalised @ normalised.T
    count = cosine.shape[0]
    off_diagonal = (cosine.sum() - count) / (count * (count - 1))

    grouped = tokens.reshape(REFERENCE_COUNT, per_reference, -1)
    grouped = torch.nn.functional.normalize(grouped, dim=-1)
    within = []
    for index in range(REFERENCE_COUNT):
        block = grouped[index] @ grouped[index].T
        size = block.shape[0]
        within.append(((block.sum() - size) / (size * (size - 1))).item())
    cross = []
    for i in range(REFERENCE_COUNT):
        for j in range(i + 1, REFERENCE_COUNT):
            cross.append((grouped[i] @ grouped[j].T).mean().item())

    centred = tokens - tokens.mean(0, keepdim=True)
    singular = torch.linalg.svdvals(centred)
    energy = singular.square()
    shares = energy / energy.sum().clamp_min(1e-12)
    entropy = -(shares * shares.clamp_min(1e-12).log()).sum()
    return {
        "cosine_offdiag": off_diagonal.item(),
        "within_ref": statistics.mean(within),
        "cross_ref": statistics.mean(cross),
        "effective_rank": float(entropy.exp()),
        "top1": float(shares[:1].sum()),
        "top4": float(shares[:4].sum()),
        "top8": float(shares[:8].sum()),
        "top16": float(shares[:16].sum()),
        "top24": float(shares[:24].sum()),
        "top32": float(shares[:32].sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    settings = checkpoint["settings"]
    model = Glyph2Hand(ModelConfig(**settings["model"]))
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    per_reference = settings["model"]["description_tokens"]
    print(f"checkpoint step={checkpoint.get('step')}  Z={per_reference}", flush=True)

    preloaded = PreloadedGlyphs(args.data, verbose=False)
    bank = preloaded.standards
    writers = [w for w in preloaded.manifest["writers"] if w["split"] == args.split]
    rng = random.Random(args.seed)

    writer_rows, null_rows = [], []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(args.samples):
            writer = rng.choice(writers)
            ids = rng.sample(writer["labels"], REFERENCE_COUNT)
            images = preloaded.writer_images(writer["id"])
            position = {c: k for k, c in enumerate(writer["labels"])}
            hands = torch.stack([to_tensor(images[position[c]]) for c in ids])[None].to(device)
            standards = torch.stack([to_tensor(bank[c]) for c in ids])[None].to(device)
            mask = torch.ones(1, REFERENCE_COUNT, dtype=torch.bool, device=device)
            writer_rows.append(summarise(model.encode_style(hands, standards, mask).float()[0].cpu(),
                                         per_reference))
            null_rows.append(summarise(model.encode_style(standards, standards, mask).float()[0].cpu(),
                                       per_reference))

    def aggregate(rows):
        return {key: statistics.median([row[key] for row in rows]) for key in rows[0]}

    report = {"step": checkpoint.get("step"), "samples": args.samples,
              "tokens_per_sample": REFERENCE_COUNT * per_reference,
              "writer": aggregate(writer_rows), "null": aggregate(null_rows)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
