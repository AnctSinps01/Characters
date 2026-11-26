"""Train the rectified-flow model that maps standard glyphs to handwriting codes."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import os
import signal
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from VAE.data import GlyphStore
from VAE.flow.data import FlowDataset, LatentSet, LatentStats, collate
from VAE.flow.model import FlowConfig, FlowModel, flow_loss, sample
from VAE.model import GlyphVAE, VAEConfig


@dataclass
class TrainConfig:
    steps: int = 200_000
    batch_size: int = 64
    lr: float = 2e-4
    warmup: int = 2_000
    ema_decay: float = 0.999
    blocks: int = 6
    time_dim: int = 128
    ref_hidden: int = 16
    trunk_channels: int = 6
    style_hidden: int = 0
    style_dilation: int = 2
    clip: float = 1.0
    ref_min: int = 1
    ref_max: int = 8
    sample_steps: int = 16
    validate_every: int = 5_000
    save_every: int = 5_000
    log_every: int = 100
    workers: int = 6
    precision: str = "bf16"
    seed: int = 0
    device: str = "cuda"


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for key, value in model.state_dict().items():
            self.shadow[key].mul_(self.decay).add_(value.detach(), alpha=1 - self.decay)

    def copy_to(self, model):
        model.load_state_dict(self.shadow)


def learning_rate(step, config):
    if step < config.warmup:
        return config.lr * (step + 1) / config.warmup
    progress = (step - config.warmup) / max(1, config.steps - config.warmup)
    return config.lr * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))


def normalize(batch, stats, device):
    mean, std = stats.mean.to(device), stats.std.to(device)
    return (batch["standard"].to(device) - mean[:2].view(1, 2, 1, 1)) / std[:2].view(1, 2, 1, 1), \
           (batch["target"].to(device) - mean[:3].view(1, 3, 1, 1)) / std[:3].view(1, 3, 1, 1), \
           (batch["refs"].to(device) - mean[2:3].view(1, 1, 1, 1)) / std[2:3].view(1, 1, 1, 1)


def prepare_validation(latents, store, count, seed=0):
    rng = np.random.default_rng(seed)
    rows = rng.choice(len(latents), count, replace=False)
    owners = np.asarray(latents.owner[rows])
    images, used = [], []
    for row, owner in zip(rows, owners):
        writer = [w for w in store.writers if w["split"] == "val"][int(owner)]
        absolute = int(row) - latents.rows[int(owner)][0]
        images.append(np.array(store.writer_images(writer)[absolute]).copy())
        used.append((int(row), int(owner), absolute))
    return rows, np.stack(images), used


def write_preview(real, generated, oracle, path):
    from PIL import Image, ImageDraw
    count = min(8, len(real))
    cell = 128
    canvas = Image.new("RGB", (3 * (cell + 6), 20 + count * (cell + 6)), "white")
    draw = ImageDraw.Draw(canvas)
    for column, title in enumerate(("real", "flow", "oracle")):
        draw.text((column * (cell + 6) + 2, 5), title, fill="black")
    to_uint8 = lambda t: ((t.detach().cpu().squeeze().clamp(-1, 1) + 1) * 127.5).numpy().round().astype(np.uint8)
    for row in range(count):
        for column, tensor in enumerate((real, generated, oracle)):
            tile = Image.fromarray(to_uint8(tensor[row])).resize((cell, cell), Image.Resampling.NEAREST)
            canvas.paste(tile, (column * (cell + 6), 20 + row * (cell + 6)))
    canvas.save(path)


@torch.no_grad()
def validate(model, ema, latents, stats, decoder, rows, images, used, store, device, config,
             seed=0, preview=None):
    live = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict({k: v.detach().clone() for k, v in ema.shadow.items()})
    model.eval()
    rng = np.random.default_rng(seed)
    writer_list = [w for w in store.writers if w["split"] == "val"]
    standard = torch.from_numpy(np.asarray(latents.standards[[int(latents.label[r]) for r in rows]],
                                           np.float32))[:, :2].to(device)
    target = torch.from_numpy(np.asarray(latents.latent[rows], np.float32))[:, :3].to(device)
    refs, counts = [], []
    for row, owner, absolute in used:
        pool = latents.rows[owner]
        others = pool[np.asarray(latents.label[pool]) != int(latents.label[row])]
        take = others[rng.choice(len(others), size=min(4, len(others)), replace=False)]
        refs.append(torch.from_numpy(np.asarray(latents.latent[take], np.float32))[:, 2:3])
        counts.append(len(take))
    widest = max(counts)
    padded = torch.zeros(len(refs), widest, 1, 8, 8)
    for index, item in enumerate(refs):
        padded[index, :len(item)] = item
    counts = torch.tensor(counts, device=device)
    mean, std = stats.mean.to(device), stats.std.to(device)
    padded = padded.to(device)
    std_in = (standard - mean[:2].view(1, 2, 1, 1)) / std[:2].view(1, 2, 1, 1)
    refs_in = (padded - mean[2:3].view(1, 1, 1, 1)) / std[2:3].view(1, 1, 1, 1)
    generated = sample(model, refs_in, std_in, steps=config.sample_steps, counts=counts)
    denormalized = generated * std[:3].to(device).view(1, 3, 1, 1) + mean[:3].to(device).view(1, 3, 1, 1)
    free = torch.from_numpy(np.asarray(latents.latent[rows], np.float32))[:, 3:4].to(device)
    codes = torch.cat([denormalized, free], dim=1)
    decoded = decoder(codes).float()
    real = torch.from_numpy(images).float().to(device)[:, None] / 127.5 - 1
    mae = (decoded - real).abs().mean().item() * 127.5
    latent_mse = (denormalized - target.to(device)).square().mean().item()
    target_codes = torch.from_numpy(np.asarray(latents.latent[rows], np.float32)).to(device)
    oracle = decoder(target_codes).float()
    oracle_mae = (oracle - real).abs().mean().item() * 127.5
    if preview is not None:
        write_preview(real, decoded, oracle, preview)
    model.load_state_dict(live)
    model.train()
    return {"mae": mae, "oracle_mae": oracle_mae, "latent_mse": latent_mse}


def save(path, model, config, flow_config, step, score):
    payload = {"model": model.state_dict(), "flow_config": asdict(flow_config),
               "train_config": asdict(config), "step": step, "score": score, "version": 1}
    temporary = str(path) + ".tmp"
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def run(args):
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device)
    config = TrainConfig(**{k: getattr(args, k) for k in TrainConfig.__dataclass_fields__})
    flow_config = FlowConfig(blocks=config.blocks, time_dim=config.time_dim,
                             ref_hidden=config.ref_hidden, trunk_channels=config.trunk_channels,
                             style_hidden=config.style_hidden, style_dilation=config.style_dilation)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    checkpoint = torch.load(args.vae, map_location="cpu", weights_only=False)
    vae_config = VAEConfig(**checkpoint["model_config"])
    vae = GlyphVAE(vae_config).to(device).eval()
    vae.load_state_dict(checkpoint["model"])
    decoder = vae.decode
    for parameter in vae.parameters():
        parameter.requires_grad_(False)

    store = GlyphStore(args.data)
    train_set = LatentSet(args.latents, "train")
    stats = LatentStats.from_latent(train_set.latent)
    dataset = FlowDataset(train_set, config.ref_min, config.ref_max, config.seed)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True,
                        num_workers=config.workers, collate_fn=collate, drop_last=True,
                        persistent_workers=config.workers > 0)
    val_set = LatentSet(args.latents, "val")
    rows, images, used = prepare_validation(val_set, store, 128, config.seed)

    model = FlowModel(flow_config).to(device)
    ema = EMA(model, config.ema_decay)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)

    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "previews").mkdir(exist_ok=True)
    (out / "run.json").write_text(json.dumps({
        "flow_config": asdict(flow_config), "train_config": asdict(config),
        "vae": str(Path(args.vae).resolve()), "latents": str(Path(args.latents).resolve()),
        "latent_stats": stats.to_dict(),
    }, indent=2), encoding="utf-8")

    started = time.monotonic()
    best = math.inf
    use_amp = config.precision == "bf16" and device.type == "cuda"
    step = 0
    previous = {}

    def stop(signum, _frame):
        raise KeyboardInterrupt

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous[signum] = signal.signal(signum, stop)
    try:
        step, best = train_loop(config, loader, model, ema, optimizer, val_set, stats, decoder,
                                rows, images, used, store, device, out, started, best, use_amp,
                                flow_config)
    except KeyboardInterrupt:
        save(out / "last.pt", model, config, flow_config, step, best)
        print(f"interrupted at step {step}; saved {out / 'last.pt'}", flush=True)
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
    store.close()
    print(f"finished: {out}")


def train_loop(config, loader, model, ema, optimizer, val_set, stats, decoder, rows, images, used,
               store, device, out, started, best, use_amp, flow_config):
    step = 0
    while step < config.steps:
        for batch in loader:
            step += 1
            lr = learning_rate(step, config)
            for group in optimizer.param_groups:
                group["lr"] = lr
            standard, target, refs = normalize(batch, stats, device)
            counts = batch["counts"].to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                loss, _ = flow_loss(model, refs, standard, target, counts)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite flow loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip, error_if_nonfinite=True)
            optimizer.step()
            ema.update(model)
            record = {"step": step, "loss": loss.item(), "lr": lr, "grad_norm": float(grad),
                      "elapsed_seconds": time.monotonic() - started}
            if step % config.validate_every == 0 or step == config.steps:
                record.update(validate(model, ema, val_set, stats, decoder, rows, images, used,
                                       store, device, config, config.seed,
                                       preview=out / "previews" / f"val_{step:07d}.png"))
                if record["mae"] < best:
                    best = record["mae"]
                    save(out / "best.pt", model, config, flow_config, step, best)
            if step % config.save_every == 0 or step == config.steps:
                save(out / "last.pt", model, config, flow_config, step, best)
            if step % config.log_every == 0 or step == config.steps:
                with (out / "metrics.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record) + "\n")
                print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v)
                                  for k, v in record.items()}), flush=True)
            if step >= config.steps:
                break
    return step, best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/prepared")
    parser.add_argument("--latents", required=True)
    parser.add_argument("--vae", required=True)
    parser.add_argument("--output", required=True)
    for field in TrainConfig.__dataclass_fields__.values():
        parser.add_argument(f"--{field.name.replace('_', '-')}", type=type(field.default),
                            default=field.default)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
