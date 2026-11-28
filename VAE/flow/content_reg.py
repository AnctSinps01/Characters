"""Deterministic regression from a standard glyph's content code to handwriting content.

No flow and no time conditioning: twelve residual blocks with three 3x3
convolutions each map the two content channels of the printed glyph onto the two
content channels of the writer's handwriting.  A configurable number of
characters can be held out of training entirely, which is what the
generalisation question needs.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from VAE.flow.data import LatentSet, LatentStats


def group_count(channels):
    for groups in (32, 16, 8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class ResBlock3(nn.Module):
    """Residual block with three 3x3 convolutions; the channel count is unchanged."""

    def __init__(self, channels):
        super().__init__()
        self.norm1 = nn.GroupNorm(group_count(channels), channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(group_count(channels), channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm3 = nn.GroupNorm(group_count(channels), channels)
        self.conv3 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        h = F.silu(self.conv1(F.silu(self.norm1(x))))
        h = F.silu(self.conv2(F.silu(self.norm2(h))))
        h = self.conv3(F.silu(self.norm3(h)))
        return x + h


class ContentRegressor(nn.Module):
    def __init__(self, channels=128, blocks=12, in_channels=2, out_channels=2):
        super().__init__()
        self.config = {"channels": channels, "blocks": blocks,
                       "in_channels": in_channels, "out_channels": out_channels}
        self.stem = nn.Sequential(nn.Conv2d(in_channels, channels, 3, padding=1),
                                  nn.GroupNorm(group_count(channels), channels), nn.SiLU())
        self.blocks = nn.ModuleList([ResBlock3(channels) for _ in range(blocks)])
        self.head = nn.Sequential(nn.GroupNorm(group_count(channels), channels), nn.SiLU(),
                                  nn.Conv2d(channels, out_channels, 3, padding=1))

    def forward(self, standard):
        hidden = self.stem(standard)
        for block in self.blocks:
            hidden = block(hidden)
        return self.head(hidden)


@dataclass
class TrainConfig:
    steps: int = 1_000_000
    batch_size: int = 1024
    lr: float = 1e-3
    warmup: int = 2000
    weight_decay: float = 1e-4
    channels: int = 128
    blocks: int = 12
    holdout: int = 256
    seed: int = 0
    log_every: int = 200
    validate_every: int = 10_000
    save_every: int = 10_000
    device: str = "cuda"


def build_arrays(train_set, stats, device):
    labels = np.asarray(train_set.label)
    codes = np.asarray(train_set.latent, np.float32)[:, :2]
    total = len(train_set.characters)
    sums = np.zeros((total, 2, 8, 8), np.float64)
    counts = np.zeros(total, np.float64)
    np.add.at(sums, labels, codes)
    np.add.at(counts, labels, 1)
    centroids = (sums / np.maximum(counts, 1)[:, None, None, None]).astype(np.float32)
    mean, std = stats.mean.to(device), stats.std.to(device)
    standard = torch.from_numpy(np.asarray(train_set.standards, np.float32)[:, :2]).to(device)
    codes_t = torch.from_numpy(codes).to(device)
    cent_t = torch.from_numpy(centroids).to(device)
    scale = std[:2].view(1, 2, 1, 1)
    shift = mean[:2].view(1, 2, 1, 1)
    return {
        "labels": labels, "codes": codes, "centroids": cent_t,
        "standard_n": (standard - shift) / scale,
        "codes_n": (codes_t - shift) / scale,
        "centroids_n": (cent_t - shift) / scale,
        "mean": mean, "std": std, "scale": scale, "shift": shift, "total_chars": total,
    }


@torch.no_grad()
def evaluate(model, arrays, char_ids, device):
    model.eval()
    table = arrays["centroids_n"][char_ids].flatten(1)
    predicted = model(arrays["standard_n"][char_ids])
    predicted_flat = predicted.flatten(1)
    similarity = F.normalize(predicted_flat, dim=1) @ F.normalize(table, dim=1).T
    hit = similarity.argmax(1).eq(torch.arange(len(char_ids), device=device)).float().mean().item() * 100
    decoded = predicted * arrays["scale"] + arrays["shift"]
    mse = (decoded - arrays["centroids"][char_ids]).square().mean().item()
    distance = torch.cdist(predicted_flat, table)
    own = distance.diag()
    masked = distance.clone()
    masked[torch.arange(len(char_ids)), torch.arange(len(char_ids))] = float("inf")
    model.train()
    return {"hit": hit, "mse_to_centroid": mse,
            "own_distance": own.mean().item(),
            "nearest_other": masked.min(1).values.mean().item(),
            "separation": (masked.min(1).values / own.clamp_min(1e-6)).mean().item()}


@torch.no_grad()
def centroid_stats(arrays, char_ids):
    """Cluster tightness of the real coding: within-char spread vs nearest other centroid."""
    device = arrays["centroids"].device
    labels = torch.from_numpy(arrays["labels"]).long().to(device)
    member = torch.isin(labels, char_ids)
    rows = member.nonzero(as_tuple=True)[0]
    codes = arrays["codes_n"][rows]
    centres = arrays["centroids_n"][labels[rows]]
    within = (codes - centres).square().mean().item()
    table = arrays["centroids_n"][char_ids].flatten(1)
    distance = torch.cdist(table, table)
    distance.fill_diagonal_(float("inf"))
    return {"within_msd": within, "centroid_gap": distance.min(1).values.mean().item()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latents", required=True)
    parser.add_argument("--output", required=True)
    for field in TrainConfig.__dataclass_fields__.values():
        parser.add_argument(f"--{field.name.replace('_', '-')}", type=type(field.default),
                            default=field.default)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    config = TrainConfig(**{k: getattr(args, k) for k in TrainConfig.__dataclass_fields__})
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    train_set = LatentSet(args.latents, "train", in_memory=True)
    stats = LatentStats.from_latent(train_set.latent)
    arrays = build_arrays(train_set, stats, device)

    generator = np.random.default_rng(config.seed)
    holdout = np.sort(generator.choice(arrays["total_chars"], config.holdout, replace=False))
    held = set(int(c) for c in holdout)
    train_rows = np.where(~np.isin(arrays["labels"], holdout))[0]
    seen_chars = np.sort(generator.choice(
        np.array([c for c in range(arrays["total_chars"]) if c not in held]),
        min(512, arrays["total_chars"] - len(holdout)), replace=False))
    held_ids = torch.from_numpy(holdout).long().to(device)
    seen_ids = torch.from_numpy(seen_chars).long().to(device)
    print(f"params {sum(p.numel() for p in ContentRegressor(config.channels, config.blocks).parameters()):,} "
          f"| chars {arrays['total_chars']} | holdout {len(holdout)} | train rows {len(train_rows)}", flush=True)

    model = ContentRegressor(config.channels, config.blocks).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    def schedule(step):
        if step < config.warmup:
            return (step + 1) / config.warmup
        progress = min((step - config.warmup) / max(1, config.steps - config.warmup), 1.0)
        return 0.5 * (1 + math.cos(math.pi * progress))

    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "run.json").write_text(json.dumps({
        "config": asdict(config),
        "params": sum(p.numel() for p in model.parameters()),
        "holdout_chars": [int(c) for c in holdout],
    }, indent=2), encoding="utf-8")

    started = time.monotonic()
    best = math.inf
    for step in range(1, config.steps + 1):
        for group in optimizer.param_groups:
            group["lr"] = config.lr * schedule(step)
        rows = np.random.choice(train_rows, config.batch_size, replace=False)
        labels = torch.from_numpy(arrays["labels"][rows]).long().to(device)
        prediction = model(arrays["standard_n"][labels])
        loss = F.mse_loss(prediction, arrays["codes_n"][torch.from_numpy(rows).to(device)])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if step % config.log_every == 0 or step == config.steps:
            print(json.dumps({"step": step, "loss": round(loss.item(), 5),
                              "lr": round(config.lr * schedule(step), 8),
                              "grad_norm": float(grad),
                              "elapsed_seconds": round(time.monotonic() - started, 1)}), flush=True)
        if step % config.validate_every == 0 or step == config.steps:
            held_metrics = evaluate(model, arrays, held_ids, device)
            seen_metrics = evaluate(model, arrays, seen_ids, device)
            clusters = centroid_stats(arrays, held_ids)
            record = {"step": step, **{f"held_{k}": round(v, 5) for k, v in held_metrics.items()},
                      **{f"seen_{k}": round(v, 5) for k, v in seen_metrics.items()},
                      **{f"cluster_{k}": round(v, 5) for k, v in clusters.items()}}
            print(json.dumps(record), flush=True)
            with (out / "metrics.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            if held_metrics["mse_to_centroid"] < best:
                best = held_metrics["mse_to_centroid"]
                torch.save({"model": model.state_dict(), "config": model.config, "step": step},
                           out / "best.pt")
        if step % config.save_every == 0 or step == config.steps:
            torch.save({"model": model.state_dict(), "config": model.config, "step": step},
                       out / "last.pt")


if __name__ == "__main__":
    main()
