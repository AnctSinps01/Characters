"""Overfit a tiny subset: if the flow cannot memorise a few hundred codes, the
architecture or the optimisation is broken rather than the data being hard."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path

import numpy as np
import torch

from VAE.flow.data import LatentSet, LatentStats
from VAE.flow.model import FlowConfig, FlowModel, flow_loss, sample
from VAE.flow.dit import DiTConfig, DiTFlow
from VAE.flow.dit import flow_loss as dit_flow_loss
from VAE.flow.dit import sample as dit_sample


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latents", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--condition", default="standard")
    parser.add_argument("--arch", default="conv")
    parser.add_argument("--depth", type=int, default=24)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    train = LatentSet(args.latents, "train")
    stats = LatentStats.from_latent(train.latent)
    mean, std = stats.mean.to(device), stats.std.to(device)

    rng = np.random.default_rng(0)
    rows = rng.choice(len(train), args.samples, replace=False)
    source = train.condition_source(args.condition)
    target = torch.from_numpy(np.asarray(train.latent[rows], np.float32)).to(device)
    standard = torch.from_numpy(np.asarray(source[np.asarray(train.label[rows])], np.float32)).to(device)
    if args.arch == "dit":
        target = target[:, :4]
        standard = standard[:, :4]
    else:
        target = target[:, :3]
        standard = standard[:, :2]
    refs, counts = [], []
    for row in rows:
        owner = int(train.owner[row])
        pool = train.rows[owner]
        others = pool[np.asarray(train.label[pool]) != int(train.label[row])]
        take = others[rng.choice(len(others), size=min(4, len(others)), replace=False)]
        refs.append(torch.from_numpy(np.asarray(train.latent[take], np.float32))[:, 2:3])
        counts.append(len(take))
    widest = max(counts)
    padded = torch.zeros(len(refs), widest, 1, 8, 8)
    for index, item in enumerate(refs):
        padded[index, :len(item)] = item
    counts = torch.tensor(counts, device=device)
    refs = (padded.to(device) - mean[2:3].view(1, 1, 1, 1)) / std[2:3].view(1, 1, 1, 1)
    n_standard = standard.shape[1]
    n_target = target.shape[1]
    standard = (standard - mean[:n_standard].view(1, n_standard, 1, 1)) / std[:n_standard].view(1, n_standard, 1, 1)
    normalized = (target - mean[:n_target].view(1, n_target, 1, 1)) / std[:n_target].view(1, n_target, 1, 1)

    if args.arch == "dit":
        config = DiTConfig(depth=args.depth)
        model = DiTFlow(config).to(device)
        loss_fn, sampler = dit_flow_loss, dit_sample
    else:
        config = FlowConfig(blocks=6, trunk_channels=24, style_hidden=4, style_dilation=2)
        model = FlowModel(config).to(device)
        loss_fn, sampler = flow_loss, sample
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)
    print(f"params {sum(p.numel() for p in model.parameters())} samples {args.samples}", flush=True)

    total = len(rows)
    for step in range(1, args.steps + 1):
        index = torch.randint(total, (min(args.batch_size, total),))
        loss, _ = loss_fn(model, refs[index], standard[index], normalized[index], counts[index])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        scheduler.step()
        if step % 500 == 0 or step == args.steps:
            with torch.no_grad():
                model.eval()
                generated = sampler(model, refs, standard, steps=16, counts=counts)
                mse = (generated - normalized).square().mean().item()
                model.train()
            print(json.dumps({"step": step, "loss": round(loss.item(), 5),
                              "sample_mse": round(mse, 5)}), flush=True)
    out = Path(args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "flow_config": asdict(config)}, out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
