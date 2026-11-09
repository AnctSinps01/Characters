"""Measure how much of a trained DiT-flow checkpoint generalises vs memorises.

Three groups are generated from the same checkpoint:

  seen     training writers, characters they actually wrote
  unseen   validation writers (never trained on), characters they actually wrote
  novel    training writers, target characters the writer never wrote
           (no ground truth, so only character identity is scored)

Every group reports a nearest-standard character-identity score: the generated
image (or the true handwriting) is matched against all 3755 standard glyphs by
ink IoU.  The true handwriting is scored as well, which calibrates how much of
the score is the metric's own ceiling.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from PIL import Image, ImageDraw

from src import Glyph2Hand, ModelConfig
from src.data import PreloadedGlyphs
from src.flow import sample_euler

REFERENCE_COUNT = 4


def load_model(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = Glyph2Hand(ModelConfig(**checkpoint["settings"]["model"]))
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval(), checkpoint


def sample_group(preloaded, writers, count, rng, novel_characters=False):
    """Build references/target/truth tensors for a list of manifest writers."""
    characters = preloaded.manifest["characters"]
    bank = preloaded.standards

    refs_hand, refs_std, targets, truths, target_ids = [], [], [], [], []
    for _ in range(count):
        writer = rng.choice(writers)
        ids = writer["labels"]                       # global character ids this writer wrote
        position = {character: row for row, character in enumerate(ids)}
        images = preloaded.writer_images(writer["id"])
        if novel_characters:
            seen = set(ids)
            missing = [i for i in range(len(characters)) if i not in seen]
            if not missing:
                raise ValueError(f"writer {writer['id']} covers every character")
            target_id = rng.choice(missing)
            ref_ids = rng.sample(ids, REFERENCE_COUNT)
            truth = None
        else:
            chosen = rng.sample(ids, REFERENCE_COUNT + 1)
            target_id, ref_ids = chosen[0], chosen[1:]
            truth = torch.tensor(np.asarray(images[position[target_id]])[None],
                                 dtype=torch.float32) / 127.5 - 1
        refs_hand.append(torch.tensor(
            np.stack([np.asarray(images[position[i]]) for i in ref_ids])[:, None],
            dtype=torch.float32) / 127.5 - 1)
        refs_std.append(torch.tensor(
            np.stack([bank[i] for i in ref_ids])[:, None],
            dtype=torch.float32) / 127.5 - 1)
        targets.append(torch.tensor(np.asarray(bank[target_id])[None],
                                    dtype=torch.float32) / 127.5 - 1)
        truths.append(truth if truth is not None else torch.zeros(1, 64, 64))
        target_ids.append(target_id)
    return {
        "reference_handwriting": torch.stack(refs_hand),
        "reference_standard": torch.stack(refs_std),
        "target_standard": torch.stack(targets),
        "target_handwriting": torch.stack(truths),
        "target_id": torch.tensor(target_ids),
    }


def nearest_standard(images: torch.Tensor, bank: torch.Tensor):
    """Match binary ink masks against every standard glyph by IoU."""
    ink = (images.squeeze(1) < 0).float().flatten(1)
    ref = (bank < 0).float().flatten(1)
    intersection = ink @ ref.T
    area_a = ink.sum(1, keepdim=True)
    area_b = ref.sum(1).unsqueeze(0)
    iou = intersection / (area_a + area_b - intersection + 1e-6)
    return iou.argmax(1), iou.max(1).values


def ink_iou(a: torch.Tensor, b: torch.Tensor) -> float:
    ia = (a.squeeze(1) < 0).float()
    ib = (b.squeeze(1) < 0).float()
    inter = (ia * ib).sum((1, 2))
    union = ia.sum((1, 2)) + ib.sum((1, 2)) - inter
    return (inter / (union + 1e-6)).mean().item()


def run_group(model, batch, device, steps, bank):
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        generated = sample_euler(
            model, batch["target_standard"].to(device),
            batch["reference_handwriting"].to(device),
            batch["reference_standard"].to(device), steps=steps)
    return generated.float().cpu()


def score(generated, batch, bank):
    target = batch["target_id"]
    result = {}
    predicted, confidence = nearest_standard(generated, bank)
    result["gen_identity"] = (predicted == target).float().mean().item()
    result["gen_confidence"] = confidence.mean().item()
    if batch["target_handwriting"].abs().sum() > 0:
        truth = batch["target_handwriting"]
        predicted_truth, confidence_truth = nearest_standard(truth, bank)
        result["truth_identity"] = (predicted_truth == target).float().mean().item()
        result["truth_confidence"] = confidence_truth.mean().item()
        result["mae_gen_truth"] = (generated - truth).abs().mean().item() * 127.5
        result["mae_std_truth"] = (batch["target_standard"] - truth).abs().mean().item() * 127.5
        result["iou_gen_truth"] = ink_iou(generated, truth)
        result["iou_std_truth"] = ink_iou(batch["target_standard"], truth)
    return result


def to_tile(tensor, cell):
    array = ((tensor.detach().float().cpu().squeeze(0).clamp(-1, 1) + 1) * 127.5)
    return Image.fromarray(array.round().byte().numpy()).resize((cell, cell), Image.Resampling.NEAREST)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=256, help="samples per group")
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    model, checkpoint = load_model(args.checkpoint, device)
    print(f"checkpoint step={checkpoint.get('step')}", flush=True)
    preloaded = PreloadedGlyphs(args.data, verbose=True)
    writers = preloaded.manifest["writers"]
    train_writers = [w for w in writers if w["split"] == "train"]
    val_writers = [w for w in writers if w["split"] == "val"]
    bank = torch.from_numpy(preloaded.standards.astype(np.float32) / 127.5 - 1)

    # A writer who covers every character has no unseen-character target.
    novel_writers = [w for w in train_writers
                     if len(w["labels"]) < len(preloaded.manifest["characters"])]
    groups = {
        "seen": (train_writers, False),
        "unseen": (val_writers, False),
        "novel": (novel_writers, True),
    }
    report = {}
    sheets = {}
    for name, (pool, novel) in groups.items():
        rng = random.Random(f"{args.seed}:{name}")
        all_metrics = []
        rows = []
        for start in range(0, args.count, args.batch_size):
            batch = sample_group(preloaded, pool, min(args.batch_size, args.count - start),
                                 rng, novel_characters=novel)
            generated = run_group(model, batch, device, args.steps, bank)
            metrics = score(generated, batch, bank)
            all_metrics.append((metrics, generated.shape[0]))
            for index in range(generated.shape[0]):
                rows.append((batch, index, generated))
            print(f"{name}: {start + generated.shape[0]}/{args.count}", flush=True)
        keys = all_metrics[0][0].keys()
        report[name] = {key: sum(m[key] * n for m, n in all_metrics) / args.count
                        for key in keys}
        sheets[name] = rows
        print(name, json.dumps(report[name], indent=2), flush=True)

    (args.output / "generalization.json").write_text(json.dumps(
        {"step": checkpoint.get("step"), "count": args.count, "groups": report}, indent=2))

    for name, rows in sheets.items():
        cell, gap, header = 96, 3, 20
        columns = [f"Ref {i + 1}" for i in range(REFERENCE_COUNT)]
        columns += [f"Std {i + 1}" for i in range(REFERENCE_COUNT)] + ["Target", "Truth", "Gen"]
        take = rows[:32]
        canvas = Image.new("RGB", (len(columns) * cell + (len(columns) - 1) * gap,
                                   header + len(take) * cell + (len(take) - 1) * gap), "white")
        draw = ImageDraw.Draw(canvas)
        for column, label in enumerate(columns):
            draw.text((column * (cell + gap), 4), label, fill="black")
        for row, (batch, index, generated) in enumerate(take):
            tiles = [to_tile(t, cell) for t in batch["reference_handwriting"][index]]
            tiles += [to_tile(t, cell) for t in batch["reference_standard"][index]]
            tiles += [to_tile(batch["target_standard"][index], cell),
                      (to_tile(batch["target_handwriting"][index], cell)
                       if batch["target_handwriting"][index].abs().sum() > 0
                       else Image.new("RGB", (cell, cell), "white")),
                      to_tile(generated[index], cell)]
            for column, tile in enumerate(tiles):
                canvas.paste(tile, (column * (cell + gap), header + row * (cell + gap)))
        canvas.save(args.output / f"{name}.png")
    print("wrote", args.output)


if __name__ == "__main__":
    main()
