"""Probe the reference encoder's style tokens for content leakage.

The reference branch computes ``handwriting_tokens - standard_tokens`` and pools
that into Z description tokens per reference.  If the subtraction really removes
the character, the style tokens should cluster by *writer*, not by *character*.

For several pairs of writers that share characters, the probe compares

    d_same_writer   style(A, char set 1) vs style(A, char set 2)
    d_diff_writer   style(A, char set 1) vs style(B, char set 1)
    d_both          style(A, char set 1) vs style(B, char set 2)

and then generates the same target character with A's and B's references to see
which writer's true handwriting the output lands closer to.
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
PAIRS = int(sys.argv[3]) if len(sys.argv) > 3 else 12


def tensor(image) -> torch.Tensor:
    return torch.tensor(np.asarray(image)[None], dtype=torch.float32) / 127.5 - 1


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(1 - torch.nn.functional.cosine_similarity(a, b, dim=-1).mean())


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

    rng = random.Random(3)
    pairs = []
    for i in range(len(writers)):
        for j in range(i + 1, len(writers)):
            shared = sorted(sets[writers[i]["id"]] & sets[writers[j]["id"]])
            if len(shared) >= 9:
                pairs.append((writers[i], writers[j], shared))
    rng.shuffle(pairs)
    pairs = pairs[:PAIRS]
    print(f"writer pairs: {len(pairs)}", flush=True)

    totals = {"d_same_writer": 0.0, "d_diff_writer": 0.0, "d_both": 0.0,
              "gen_closer_to_ref_writer": 0.0, "gen_closer_to_other_writer": 0.0,
              "mae_to_ref_writer": 0.0, "mae_to_other_writer": 0.0,
              "mae_between_generations": 0.0, "mae_between_truths": 0.0,
              "ink_attribution": 0.0}
    count = 0
    with torch.no_grad():
        for writer_a, writer_b, shared in pairs:
            set1, set2, target = shared[:4], shared[4:8], shared[8]

            def references(writer, ids):
                images = preloaded.writer_images(writer["id"])
                position = {c: k for k, c in enumerate(writer["labels"])}
                hands = torch.stack([tensor(images[position[c]]) for c in ids])
                standards = torch.stack([tensor(bank[c]) for c in ids])
                return hands.to(device), standards.to(device)

            hands_a1, std_a1 = references(writer_a, set1)
            hands_a2, std_a2 = references(writer_a, set2)
            hands_b1, std_b1 = references(writer_b, set1)
            hands_b2, std_b2 = references(writer_b, set2)

            def style(hands, standards):
                tokens = model.encode_style(hands.unsqueeze(0), standards.unsqueeze(0))
                return tokens.mean(1).squeeze(0)

            s_a1, s_a2 = style(hands_a1, std_a1), style(hands_a2, std_a2)
            s_b1, s_b2 = style(hands_b1, std_b1), style(hands_b2, std_b2)
            totals["d_same_writer"] += cos(s_a1, s_a2)
            totals["d_diff_writer"] += cos(s_a1, s_b1)
            totals["d_both"] += cos(s_a1, s_b2)

            target_std = tensor(bank[target]).unsqueeze(0).to(device)
            generated = {}
            for name, (hands, standards) in (("a", (hands_a1, std_a1)),
                                             ("b", (hands_b1, std_b1))):
                generated[name] = sample_euler(
                    model, target_std, hands.unsqueeze(0), standards.unsqueeze(0),
                    steps=32).squeeze(0).cpu()

            images_a = preloaded.writer_images(writer_a["id"])
            images_b = preloaded.writer_images(writer_b["id"])
            truth_a = tensor(images_a[{c: k for k, c in enumerate(writer_a["labels"])}[target]])
            truth_b = tensor(images_b[{c: k for k, c in enumerate(writer_b["labels"])}[target]])

            mae_aa = (generated["a"] - truth_a).abs().mean().item() * 127.5
            mae_ab = (generated["a"] - truth_b).abs().mean().item() * 127.5
            mae_ba = (generated["b"] - truth_a).abs().mean().item() * 127.5
            mae_bb = (generated["b"] - truth_b).abs().mean().item() * 127.5
            totals["mae_to_ref_writer"] += (mae_aa + mae_bb) / 2
            totals["mae_to_other_writer"] += (mae_ab + mae_ba) / 2
            totals["mae_between_generations"] += (generated["a"] - generated["b"]).abs().mean().item() * 127.5
            totals["mae_between_truths"] += (truth_a - truth_b).abs().mean().item() * 127.5
            ink = lambda x: (x < 0).float().mean().item()
            totals["ink_attribution"] += ((abs(ink(generated["a"]) - ink(truth_a))
                                           < abs(ink(generated["a"]) - ink(truth_b)))
                                          + (abs(ink(generated["b"]) - ink(truth_b))
                                             < abs(ink(generated["b"]) - ink(truth_a))))
            totals["gen_closer_to_ref_writer"] += (mae_aa < mae_ab) + (mae_bb < mae_ba)
            totals["gen_closer_to_other_writer"] += (mae_aa > mae_ab) + (mae_bb > mae_ba)
            count += 1
            print(f"pair {count}/{len(pairs)}: d_same={cos(s_a1, s_a2):.3f} "
                  f"d_diff={cos(s_a1, s_b1):.3f} mae_ref={mae_aa:.1f}/{mae_bb:.1f} "
                  f"mae_other={mae_ab:.1f}/{mae_ba:.1f}", flush=True)

    report = {key: value / count for key, value in totals.items()
              if key not in ("gen_closer_to_ref_writer", "gen_closer_to_other_writer",
                             "ink_attribution")}
    report["gen_closer_to_ref_writer"] = totals["gen_closer_to_ref_writer"] / (2 * count)
    report["gen_closer_to_other_writer"] = totals["gen_closer_to_other_writer"] / (2 * count)
    report["ink_attribution"] = totals["ink_attribution"] / (2 * count)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
