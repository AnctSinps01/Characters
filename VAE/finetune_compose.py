"""Decoder-only fine-tune that teaches a blocked VAE to accept recombined codes.

The blocked VAE was trained on self-consistent pairs only, so mixing the content
block of one sample with the author block of another produces a weaker author
trace.  Here the encoder stays frozen and only the decoder learns, with three
terms: reconstruction on real pairs, cycle consistency after recombination
(re-encoding a recombined decode must recover both source codes), and an ink
target that keeps recombined strokes as heavy as real ones.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from VAE.data import GlyphStore, ReconstructionDataset
from VAE.model import GlyphVAE, VAEConfig
from VAE.train import GridBatchSampler, pair_table, spread_indices


@dataclass
class FinetuneConfig:
    steps: int = 200_000
    batch_size: int = 256
    grid: int = 16
    lr: float = 5e-5
    weight_recon: float = 1.0
    weight_cycle: float = 0.2
    weight_ink: float = 0.05
    validate_every: int = 2_500
    save_every: int = 2_500
    log_every: int = 100
    val_samples: int = 512
    workers: int = 6
    precision: str = "bf16"
    seed: int = 0
    device: str = "cuda"

    def __post_init__(self):
        if self.grid < 2 or self.grid ** 2 != self.batch_size:
            raise ValueError("batch_size must equal grid**2")
        if self.precision not in ("bf16", "fp32"):
            raise ValueError("precision must be bf16 or fp32")
        for name in ("weight_recon", "weight_cycle", "weight_ink"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")


def split_slices(training, channels):
    content = slice(0, training["content_channels"])
    author = slice(training["content_channels"],
                   training["content_channels"] + training["author_channels"])
    free = slice(author.stop, channels)
    return content, author, free


def cosine(a, b):
    return F.cosine_similarity(a.flatten(1), b.flatten(1), dim=1).mean()


def combine(mu, content, author, free):
    """Content from sample i, author from a different sample, free from anywhere."""
    batch = mu.shape[0]
    shift = max(batch // 2, 1)
    source = (torch.arange(batch, device=mu.device) + shift) % batch
    free_source = torch.randint(batch, (batch,), device=mu.device)
    return torch.cat([mu[:, content], mu[source][:, author], mu[free_source][:, free]], dim=1)


@torch.no_grad()
def validate(model, loader, device, config, content, author, free):
    model.eval()
    stats = {}
    for name in ("matched", "composed"):
        stats[name] = {"ink": 0.0, "sharp": 0.0, "content_cos": 0.0, "author_cos": 0.0, "count": 0}
    for raw in loader:
        images = raw["image"].to(device).float() / 127.5 - 1
        mu, _ = model.encode(images)
        matched = model.decode(mu).float()
        mix = combine(mu, content, author, free)
        composed = model.decode(mix).float()
        for name, out, source in (("matched", matched, mu), ("composed", composed, mix)):
            again, _ = model.encode(out)
            grad = (out[:, :, 1:] - out[:, :, :-1]).abs().mean() + (out[:, :, :, 1:] - out[:, :, :, :-1]).abs().mean()
            entry = stats[name]
            entry["ink"] += (out < 0).float().mean().item() * len(images)
            entry["sharp"] += grad.item() * len(images)
            entry["content_cos"] += cosine(again[:, content], source[:, content]).item() * len(images)
            entry["author_cos"] += cosine(again[:, author], source[:, author]).item() * len(images)
            entry["count"] += len(images)
    result = {}
    for name, entry in stats.items():
        n = max(entry.pop("count"), 1)
        result.update({f"{name}_{k}": v / n for k, v in entry.items()})
    model.train()
    return result


def save(path, model, config, model_config, step, score, extra=None):
    payload = {"model_config": asdict(model_config), "model": model.state_dict(),
               "step": step, "score": score, "finetune_config": asdict(config),
               "version": 1}
    if extra:
        payload.update(extra)
    temporary = str(path) + ".tmp"
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def run(args):
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                          if args.device == "auto" else args.device)
    config = FinetuneConfig(**{k: getattr(args, k) for k in FinetuneConfig.__dataclass_fields__})
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_config = VAEConfig(**checkpoint["model_config"])
    model = GlyphVAE(model_config).to(device)
    model.load_state_dict(checkpoint["model"])
    channels = model_config.latent_channels
    source = checkpoint["training_config"]
    content, author, free = split_slices(source, channels)
    if content.stop == 0 or author.stop == content.stop:
        raise ValueError("checkpoint has no content/author channel split")

    store = GlyphStore(args.data)
    table = pair_table(store, "train")
    characters = sorted(torch.nonzero(torch.as_tensor((table >= 0).all(axis=1))).flatten().tolist())
    train_authors = [w for w in store.writers if w["split"] == "train"]
    sampler = GridBatchSampler(table, characters, list(range(len(train_authors))),
                               config.grid, 0, config.steps, config.seed)
    dataset = ReconstructionDataset(store, split="train", source="handwriting")
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=config.workers,
                        pin_memory=device.type == "cuda", persistent_workers=config.workers > 0)

    val_dataset = ReconstructionDataset(store, split="val", source="handwriting")
    indices = spread_indices(len(val_dataset), config.val_samples * 4)
    val_loader = DataLoader(torch.utils.data.Subset(val_dataset, indices),
                            batch_size=config.batch_size, shuffle=False, num_workers=0)

    for parameter in model.encoder.parameters():
        parameter.requires_grad_(False)
    model.encoder.eval()
    model.decoder.train()
    optimizer = torch.optim.AdamW(model.decoder.parameters(), lr=config.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.steps)

    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "previews").mkdir(exist_ok=True)
    (out / "run.json").write_text(json.dumps({
        "source_checkpoint": str(Path(args.checkpoint).resolve()),
        "model_config": asdict(model_config), "finetune_config": asdict(config),
        "source_training_config": source,
    }, indent=2), encoding="utf-8")

    started = time.monotonic()
    best = -math.inf
    use_amp = config.precision == "bf16" and device.type == "cuda"
    for step, raw in enumerate(loader, start=1):
        images = raw["image"].to(device, non_blocking=True).float() / 127.5 - 1
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            with torch.no_grad():
                mu, _ = model.encode(images)
            recon = model.decode(mu)
            recon_loss = F.mse_loss(recon.float(), images)
            mix = combine(mu, content, author, free)
            composed = model.decode(mix)
            with torch.no_grad():
                again, _ = model.encode(composed.detach().float())
            cycle = (1 - cosine(again[:, content], mix[:, content])) + \
                    (1 - cosine(again[:, author], mix[:, author]))
            ink_gap = ((composed.float() < 0).float().mean() - (images < 0).float().mean()).abs()
            loss = (config.weight_recon * recon_loss + config.weight_cycle * cycle
                    + config.weight_ink * ink_gap)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite fine-tune loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()

        record = {"step": step, "loss": loss.item(), "recon": recon_loss.item(),
                  "cycle": cycle.item(), "ink_gap": ink_gap.item(), "grad_norm": float(grad),
                  "lr": scheduler.get_last_lr()[0], "elapsed_seconds": time.monotonic() - started}
        if step % config.validate_every == 0 or step == config.steps:
            record.update(validate(model, val_loader, device, config, content, author, free))
            score = record["composed_author_cos"] + record["composed_content_cos"]
            record["score"] = score
            if score > best:
                best = score
                save(out / "best.pt", model, config, model_config, step, score)
        if step % config.save_every == 0 or step == config.steps:
            save(out / "last.pt", model, config, model_config, step, best)
        if step % config.log_every == 0 or step == config.steps:
            with (out / "metrics.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in record.items()}),
                  flush=True)
        if step >= config.steps:
            break
    store.close()
    print(f"finished: {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/prepared")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    for field in FinetuneConfig.__dataclass_fields__.values():
        parser.add_argument(f"--{field.name.replace('_', '-')}", type=type(field.default),
                            default=field.default)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
