"""Train Ref2Latent: reference latents + standard-glyph latent -> target latent mean."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import signal
import tempfile
import time

import torch
from torch.utils.data import DataLoader, Subset
from PIL import Image, ImageDraw

from VAE.data import GlyphStore, spread_indices
from VAE.model import GlyphVAE, VAEConfig
from .data import LatentStore, RefTargetDataset
from .model import ModelConfig, Ref2Latent


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 200_000
    batch_size: int = 64
    learning_rate: float = 2e-4
    warmup_steps: int = 1000
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    latent_weight: float = 1.0
    image_weight: float = 1.0
    precision: str = "bf16"
    log_every: int = 100
    validate_every: int = 2500
    save_every: int = 2500
    val_samples: int = 1024
    workers: int = 4
    references: int = 4
    seed: int = 42

    def __post_init__(self):
        for name in ("steps", "batch_size", "log_every", "validate_every", "save_every", "references"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("warmup_steps", "val_samples", "workers"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.warmup_steps >= self.steps:
            raise ValueError("warmup_steps must be smaller than steps")
        for name in ("learning_rate", "gradient_clip"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("weight_decay", "latent_weight", "image_weight"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative")
        if self.latent_weight == 0 and self.image_weight == 0:
            raise ValueError("at least one loss term must be nonzero")
        if self.precision not in ("bf16", "fp32"):
            raise ValueError("precision must be bf16 or fp32")


def learning_rate(config, step):
    if config.warmup_steps and step <= config.warmup_steps:
        return config.learning_rate * step / config.warmup_steps
    progress = (step - config.warmup_steps) / (config.steps - config.warmup_steps)
    return config.learning_rate * (config.min_lr_ratio + (1 - config.min_lr_ratio) *
                                   (1 + math.cos(math.pi * progress)) / 2)


def amp(device, precision):
    return (torch.autocast("cuda", dtype=torch.bfloat16)
            if device.type == "cuda" and precision == "bf16" else nullcontext())


def atomic_save(state, path):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".ref2latent-", suffix=".tmp")
    os.close(fd)
    try:
        torch.save(state, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def preview(inputs, predicted_pixels, target_pixels, path):
    n = min(6, len(inputs))
    cell = inputs.shape[-1] * 2
    columns = ("Input", "Decoded prediction", "Decoded target latent")
    canvas = Image.new("RGB", (cell * len(columns), 22 + n * cell), "white")
    draw = ImageDraw.Draw(canvas)
    for j, label in enumerate(columns):
        draw.text((j * cell + 2, 4), label, fill="black")
    for i in range(n):
        for j, tile in enumerate((inputs[i], predicted_pixels[i], target_pixels[i])):
            pixels = ((tile.detach().float().cpu().squeeze().clamp(-1, 1) + 1) * 127.5)
            image = Image.fromarray(pixels.round().byte().numpy()).resize(
                (cell, cell), Image.Resampling.NEAREST)
            canvas.paste(image, (j * cell, 22 + i * cell))
    canvas.save(path)


@torch.no_grad()
def evaluate(model, decoder, loader, device, precision):
    was_training = model.training
    model.eval()
    totals = {"n": 0.0, "latent_mse": 0.0, "image_mse": 0.0, "image_mae": 0.0,
              "baseline_mae": 0.0, "decoded": None}
    for raw in loader:
        reference = raw["reference"].to(device).float()
        standard = raw["standard"].to(device).float()
        target = raw["target"].to(device).float()
        image = raw["image"].to(device).float() / 127.5 - 1
        with amp(device, precision):
            predicted = model(reference, standard)
        n = len(predicted)
        totals["n"] += n
        totals["latent_mse"] += (predicted.float() - target).square().mean().item() * n
        if decoder is not None:
            decoded_predicted = decoder.decode(predicted.float())
            decoded_target = decoder.decode(target.float())
            totals["image_mse"] += (decoded_predicted - image).square().mean().item() * n
            totals["image_mae"] += (decoded_predicted - image).abs().mean().item() * 127.5 * n
            totals["baseline_mae"] += (decoded_target - image).abs().mean().item() * 127.5 * n
            if totals["decoded"] is None:
                totals["decoded"] = (image, decoded_predicted, decoded_target)
    model.train(was_training)
    n = max(totals["n"], 1.0)
    result = {key: value / n for key, value in totals.items()
              if key not in ("n", "decoded")}
    result["samples"] = int(totals["n"])
    return result, totals["decoded"]


def endless(loader):
    """Repeat epochs until the step budget is reached."""
    while True:
        for batch in loader:
            yield batch


def run(args):
    device = torch.device(args.device if args.device != "auto" else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    config = TrainConfig(**{k: getattr(args, k) for k in TrainConfig.__dataclass_fields__})
    model_config = ModelConfig(latent_channels=args.latent_channels,
                               latent_size=args.latent_size, references=config.references,
                               dim=args.dim, heads=args.heads, layers=args.layers,
                               mlp_ratio=args.mlp_ratio)
    store = GlyphStore(args.data)
    latents = LatentStore(args.latents)
    expected = (model_config.latent_channels, model_config.latent_size, model_config.latent_size)
    if tuple(latents.shape) != expected:
        raise ValueError(f"latent store shape {latents.shape} does not match model {expected}")

    decoder = None
    if args.vae is not None:
        checkpoint = torch.load(args.vae, map_location="cpu", weights_only=False)
        decoder_config = VAEConfig(**checkpoint["model_config"])
        if tuple(decoder_config.latent_shape) != expected:
            raise ValueError("VAE checkpoint latent shape does not match the model")
        decoder = GlyphVAE(decoder_config).to(device).eval()
        decoder.load_state_dict(checkpoint["model"])
        for parameter in decoder.parameters():
            parameter.requires_grad_(False)

    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise ValueError("output must be new/empty; refusing overwrite")
    out.mkdir(parents=True, exist_ok=True)
    (out / "previews").mkdir(exist_ok=True)
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    model = Ref2Latent(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                  betas=(0.9, 0.999), weight_decay=config.weight_decay)
    train_data = RefTargetDataset(store, latents, split="train",
                                  references=config.references, seed=config.seed)
    loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True,
                        num_workers=config.workers, pin_memory=device.type == "cuda",
                        drop_last=True, generator=torch.Generator().manual_seed(config.seed + 50))
    val_data = RefTargetDataset(store, latents, split="val",
                                references=config.references, seed=config.seed)
    val_loader = DataLoader(Subset(val_data, spread_indices(len(val_data), config.val_samples)),
                            batch_size=config.batch_size, num_workers=0)
    (out / "run.json").write_text(json.dumps({
        "model": asdict(model_config), "training": asdict(config),
        "parameters": sum(p.numel() for p in model.parameters()),
        "latent_store": str(Path(args.latents).resolve()),
        "latent_manifest": latents.manifest,
        "vae": str(args.vae) if args.vae else None}, indent=2), encoding="utf-8")
    print(f"device={device} parameters={sum(p.numel() for p in model.parameters()):,} "
          f"train_pairs={len(train_data)} val_pairs={len(val_data)} "
          f"latent={expected}", flush=True)

    stop = [False]
    def request_stop(_signum, _frame):
        stop[0] = True
    previous = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    step, best = 0, float("inf")

    def save():
        atomic_save({"version": 1, "step": step, "model_config": asdict(model_config),
                     "training_config": asdict(config), "best_val_latent_mse": best,
                     "model": model.state_dict(), "optimizer": optimizer.state_dict()},
                    out / "last.pt")

    started = time.monotonic()
    model.train()
    try:
        for raw in endless(loader):
            if stop[0]:
                break
            step += 1
            lr = learning_rate(config, step)
            for group in optimizer.param_groups:
                group["lr"] = lr
            reference = raw["reference"].to(device).float()
            standard = raw["standard"].to(device).float()
            target = raw["target"].to(device).float()
            image = raw["image"].to(device).float() / 127.5 - 1
            optimizer.zero_grad(set_to_none=True)
            with amp(device, config.precision):
                predicted = model(reference, standard)
            latent_loss = (predicted.float() - target).square().mean()
            image_loss = predicted.new_zeros(())
            if decoder is not None and config.image_weight > 0:
                decoded = decoder.decode(predicted.float())
                image_loss = (decoded - image).square().mean()
            loss = config.latent_weight * latent_loss + config.image_weight * image_loss
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite loss; existing checkpoints preserved")
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip,
                                                  error_if_nonfinite=True)
            optimizer.step()
            record = {"step": step, "loss": loss.item(), "latent_mse": latent_loss.item(),
                      "image_mse": image_loss.item(), "lr": lr, "grad_norm": float(grad),
                      "elapsed_seconds": time.monotonic() - started}
            validate = step % config.validate_every == 0 or step == config.steps
            if validate and not stop[0]:
                result, decoded_views = evaluate(model, decoder, val_loader, device,
                                                 config.precision)
                record["val"] = result
                if decoded_views is not None:
                    preview(decoded_views[0], decoded_views[1], decoded_views[2],
                            out / "previews" / f"val_{step:07d}.png")
                improved = result["latent_mse"] < best
                best = min(best, result["latent_mse"])
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
            if step >= config.steps:
                break
        save()
        print(f"Saved step {step}; {'stopped' if stop[0] else 'completed'}: {out / 'last.pt'}")
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        store.close()
        latents.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/prepared"))
    parser.add_argument("--latents", type=Path, required=True)
    parser.add_argument("--vae", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--latent-channels", type=int, default=4)
    parser.add_argument("--latent-size", type=int, default=8)
    parser.add_argument("--dim", type=int, default=384)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    for name, field in TrainConfig.__dataclass_fields__.items():
        parser.add_argument("--" + name.replace("_", "-"), type=type(field.default),
                            default=field.default)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
