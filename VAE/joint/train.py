"""Train a decoder over the concatenated content and author codes."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import random
import signal
import tempfile
import time

import torch
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw

from Ref2Latent.data import LatentStore
from VAE.data import GlyphStore, spread_indices
from .config import DecoderConfig
from .data import JointCodeDataset
from .merged import MergedDataset
from .model import JointDecoder


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 100_000
    batch_size: int = 64
    learning_rate: float = 2e-4
    warmup_steps: int = 1000
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    precision: str = "bf16"
    log_every: int = 100
    validate_every: int = 2500
    save_every: int = 2500
    val_samples: int = 1024
    workers: int = 4
    seed: int = 42

    def __post_init__(self):
        for name in ("steps", "batch_size", "log_every", "validate_every", "save_every"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("warmup_steps", "val_samples", "workers"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.warmup_steps >= self.steps:
            raise ValueError("warmup_steps must be smaller than steps")
        if self.precision not in ("bf16", "fp32"):
            raise ValueError("precision must be bf16 or fp32")


def learning_rate(config, step):
    if config.warmup_steps and step <= config.warmup_steps:
        return config.learning_rate * step / config.warmup_steps
    progress = (step - config.warmup_steps) / (config.steps - config.warmup_steps)
    return config.learning_rate * (config.min_lr_ratio + (1 - config.min_lr_ratio) *
                                   (1 + math.cos(math.pi * progress)) / 2)


def amp(device, precision):
    from contextlib import nullcontext
    return (torch.autocast("cuda", dtype=torch.bfloat16)
            if device.type == "cuda" and precision == "bf16" else nullcontext())


def atomic_save(state, path):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".joint-", suffix=".tmp")
    os.close(fd)
    try:
        torch.save(state, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def preview(images, normal, swapped, path):
    n = min(6, len(images))
    cell = images.shape[-1] * 2
    labels = ("Input", "Both codes", "Author code shuffled", "Content code shuffled")
    canvas = Image.new("RGB", (cell * len(labels), 22 + n * cell), "white")
    draw = ImageDraw.Draw(canvas)
    for j, label in enumerate(labels):
        draw.text((j * cell + 2, 4), label, fill="black")
    for i in range(n):
        for j, tile in enumerate((images[i], normal[i], swapped["author"][i], swapped["content"][i])):
            pixels = ((tile.detach().float().cpu().squeeze().clamp(-1, 1) + 1) * 127.5)
            canvas.paste(Image.fromarray(pixels.round().byte().numpy()).resize(
                (cell, cell), Image.Resampling.NEAREST), (j * cell, 22 + i * cell))
    canvas.save(path)


@torch.no_grad()
def evaluate(model, loader, device, precision, *, preview_path=None):
    model.eval()
    totals = {"n": 0.0, "mae": 0.0, "mae_shuffled_author": 0.0, "mae_shuffled_content": 0.0}
    views = None
    generator = torch.Generator().manual_seed(7)
    for raw in loader:
        content = raw["content"].to(device).float()
        author = raw["author"].to(device).float()
        image = raw["image"].to(device).float() / 127.5 - 1
        with amp(device, precision):
            normal = model(content, author)
            perm = torch.randperm(len(content), generator=generator).to(device)
            swapped_author = model(content, author[perm])
            swapped_content = model(content[perm], author)
        n = len(image)
        totals["n"] += n
        totals["mae"] += (normal - image).abs().mean().item() * 127.5 * n
        totals["mae_shuffled_author"] += (swapped_author - image).abs().mean().item() * 127.5 * n
        totals["mae_shuffled_content"] += (swapped_content - image).abs().mean().item() * 127.5 * n
        if preview_path is not None and views is None:
            views = (image, normal, {"author": swapped_author, "content": swapped_content})
    n = max(totals["n"], 1.0)
    result = {key: value / n for key, value in totals.items() if key != "n"}
    result["samples"] = int(totals["n"])
    if preview_path is not None and views is not None:
        preview(*views, preview_path)
    return result


def run(args):
    device = torch.device(args.device if args.device != "auto" else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    config = TrainConfig(**{k: getattr(args, k) for k in TrainConfig.__dataclass_fields__})
    decoder_config = DecoderConfig(latent_channels=args.latent_channels,
                                   latent_size=args.latent_size,
                                   base_channels=args.base_channels)
    store = GlyphStore(args.data)
    content = LatentStore(args.content_latents) if args.content_latents else None
    author = LatentStore(args.author_latents) if args.author_latents else None
    if args.merged is None and (content is None or author is None):
        raise ValueError("pass --merged, or both --content-latents and --author-latents")
    out = Path(args.output).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError("output must be new/empty; refusing overwrite")
    out.mkdir(parents=True, exist_ok=True)
    (out / "previews").mkdir(exist_ok=True)
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    model = JointDecoder(decoder_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                  betas=(0.9, 0.999), weight_decay=config.weight_decay)
    train_data = (MergedDataset(args.merged, "train") if args.merged
                  else JointCodeDataset(store, content, author, split="train"))
    loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True,
                        num_workers=config.workers, pin_memory=device.type == "cuda",
                        drop_last=True, persistent_workers=config.workers > 0,
                        generator=torch.Generator().manual_seed(config.seed + 50))
    val_data = (MergedDataset(args.merged, "val") if args.merged
                else JointCodeDataset(store, content, author, split="val"))
    from torch.utils.data import Subset
    val_loader = DataLoader(Subset(val_data, spread_indices(len(val_data), config.val_samples)),
                            batch_size=config.batch_size, num_workers=0)
    (out / "run.json").write_text(json.dumps({
        "decoder": asdict(decoder_config), "training": asdict(config),
        "parameters": sum(p.numel() for p in model.parameters()),
        "merged": str(Path(args.merged).resolve()) if args.merged else None,
        "content_latents": str(Path(args.content_latents).resolve()) if args.content_latents else None,
        "author_latents": str(Path(args.author_latents).resolve()) if args.author_latents else None},
        indent=2), encoding="utf-8")
    print(f"device={device} parameters={sum(p.numel() for p in model.parameters()):,} "
          f"train_pairs={len(train_data)} val_pairs={len(val_data)}", flush=True)

    stop = [False]
    def request_stop(_signum, _frame):
        stop[0] = True
    previous = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    step, best = 0, float("inf")

    def save():
        atomic_save({"version": 1, "step": step, "decoder_config": asdict(decoder_config),
                     "training_config": asdict(config), "best_val_mae": best,
                     "model": model.state_dict(), "optimizer": optimizer.state_dict()},
                    out / "last.pt")

    def endless(source):
        while True:
            for batch in source:
                yield batch

    started = time.monotonic()
    model.train()
    try:
        for raw in endless(loader):
            if stop[0] or step >= config.steps:
                break
            step += 1
            lr = learning_rate(config, step)
            for group in optimizer.param_groups:
                group["lr"] = lr
            content_code = raw["content"].to(device).float()
            author_code = raw["author"].to(device).float()
            image = raw["image"].to(device).float() / 127.5 - 1
            optimizer.zero_grad(set_to_none=True)
            with amp(device, config.precision):
                decoded = model(content_code, author_code)
            loss = (decoded.float() - image).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite loss; existing checkpoints preserved")
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip,
                                                  error_if_nonfinite=True)
            optimizer.step()
            record = {"step": step, "mse": loss.item(), "lr": lr, "grad_norm": float(grad),
                      "elapsed_seconds": time.monotonic() - started}
            validate = step % config.validate_every == 0 or step == config.steps
            if validate and not stop[0]:
                record["val"] = evaluate(model, val_loader, device, config.precision,
                                         preview_path=out / "previews" / f"val_{step:07d}.png")
                improved = record["val"]["mae"] < best
                best = min(best, record["val"]["mae"])
                model.train()
            else:
                improved = False
            if step % config.log_every == 0 or validate or stop[0]:
                with (out / "metrics.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record) + "\n")
                print(json.dumps(record), flush=True)
            if step % config.save_every == 0 or validate or stop[0]:
                save()
                if improved:
                    state = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
                    atomic_save(state, out / "best.pt")
        save()
        print(f"Saved step {step}; {'stopped' if stop[0] else 'completed'}: {out / 'last.pt'}")
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        store.close()
        if content is not None:
            content.close()
        if author is not None:
            author.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/prepared"))
    parser.add_argument("--content-latents", type=Path, default=None)
    parser.add_argument("--author-latents", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--merged", type=Path, default=None,
                        help="directory from VAE.joint.prepare_merged; single-reader inputs")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--latent-channels", type=int, default=4)
    parser.add_argument("--latent-size", type=int, default=8)
    parser.add_argument("--base-channels", type=int, default=64)
    for name, field in TrainConfig.__dataclass_fields__.items():
        parser.add_argument("--" + name.replace("_", "-"), type=type(field.default),
                            default=field.default)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
