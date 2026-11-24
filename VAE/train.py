"""Independent glyph VAE training. Execute from Characters/: python -m VAE.train."""
import argparse
from contextlib import nullcontext
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
from torch.utils.data import DataLoader, Subset
from PIL import Image, ImageDraw

from .data import GlyphStore, ReconstructionDataset, spread_indices
from .model import GlyphVAE, VAEConfig, vae_loss


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 1_000_000
    batch_size: int = 64
    standard_ratio: float = 0.0
    learning_rate: float = 1e-4
    warmup_steps: int = 1000
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.01
    beta: float = 1e-4
    kl_warmup_steps: int = 10_000
    gradient_clip: float = 1.0
    workers: int = 0
    precision: str = "bf16"
    log_every: int = 100
    validate_every: int = 2500
    save_every: int = 2500
    val_samples: int = 1024  # evenly spaced across all validation authors; 0 = full
    seed: int = 42
    char_cos_weight: float = 0.0
    char_cos_warmup: int = 10_000
    char_cos_temperature: float = 0.1
    chars_per_batch: int = 0
    writers_per_char: int = 0

    def __post_init__(self):
        for name in ("steps", "batch_size", "log_every", "validate_every", "save_every"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("workers", "warmup_steps", "kl_warmup_steps", "val_samples",
                     "char_cos_warmup", "chars_per_batch", "writers_per_char"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.warmup_steps >= self.steps:
            raise ValueError("warmup_steps must be smaller than steps")
        for name in ("learning_rate", "gradient_clip"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("beta", "weight_decay", "char_cos_weight"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in ("standard_ratio", "min_lr_ratio"):
            if not math.isfinite(getattr(self, name)) or not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.precision not in ("bf16", "fp32"):
            raise ValueError("precision must be bf16 or fp32")
        if not math.isfinite(self.char_cos_temperature) or self.char_cos_temperature <= 0:
            raise ValueError("char_cos_temperature must be finite and positive")
        if self.chars_per_batch > 0 or self.char_cos_weight > 0:
            if self.chars_per_batch < 2 or self.writers_per_char < 2:
                raise ValueError("grouped batches need chars_per_batch and "
                                 "writers_per_char of at least 2")
            if self.chars_per_batch * self.writers_per_char != self.batch_size:
                raise ValueError("chars_per_batch * writers_per_char must equal batch_size")
            if self.standard_ratio > 0:
                raise ValueError("grouped batches need standard_ratio 0")


def schedule(config, step):
    if config.warmup_steps and step <= config.warmup_steps:
        lr = config.learning_rate * step / config.warmup_steps
    else:
        progress = (step - config.warmup_steps) / (config.steps - config.warmup_steps)
        lr = config.learning_rate * (config.min_lr_ratio + (1 - config.min_lr_ratio) *
                                     (1 + math.cos(math.pi * progress)) / 2)
    beta = config.beta * min(1.0, step / max(1, config.kl_warmup_steps))
    cos_weight = config.char_cos_weight * min(1.0, step / max(1, config.char_cos_warmup))
    return lr, beta, cos_weight


def char_cosine_terms(mean, labels, temperature=0.1):
    """Supervised contrastive loss over same-class positives and the rest as negatives.

    Both groups are L2-normalised posterior means; ``temperature`` scales the dot
    products inside the softmax. Returns the loss and the mean positive/negative
    cosines, which expose collapse (both near 1) versus separation (gap grows).
    Returns zeros when the batch holds no same-class pair.
    """
    if not torch.is_tensor(labels) or len(labels) < 2:
        return mean.new_zeros(()), mean.new_zeros(()), mean.new_zeros(())
    unit = torch.nn.functional.normalize(mean.flatten(1).float(), dim=1)
    cosine = unit @ unit.t()
    device = cosine.device
    labels = labels.to(device)
    same = labels[:, None] == labels[None, :]
    self_mask = torch.eye(len(labels), dtype=torch.bool, device=device)
    positive = same & ~self_mask
    negative = ~same
    if not positive.any():
        return mean.new_zeros(()), mean.new_zeros(()), mean.new_zeros(())
    logits = cosine / temperature
    logits = logits.masked_fill(self_mask, float("-inf"))
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    # masked_select avoids -inf * 0, which would turn the loss into NaN.
    loss = -torch.masked_select(log_prob, positive).mean()
    return loss, cosine[positive].mean(), cosine[negative].mean()


class GroupedBatchSampler:
    """Batches of `chars_per_batch` classes times `writers_per_char` authors each.

    Indices are drawn from a fixed per-(seed, step) RNG, so a resumed run sees the
    same batches and workers never share mutable state.
    """
    def __init__(self, class_indices, chars_per_batch, writers_per_char,
                 start_step, steps, seed):
        self.class_indices = class_indices
        self.classes = sorted(class_indices)
        self.chars_per_batch = chars_per_batch
        self.writers_per_char = writers_per_char
        self.start_step = start_step
        self.steps = steps
        self.seed = seed
        if chars_per_batch > len(self.classes):
            raise ValueError("chars_per_batch exceeds the number of classes")

    def __len__(self):
        return max(0, self.steps - self.start_step)

    def __iter__(self):
        for step in range(self.start_step, self.steps):
            rng = random.Random(f"{self.seed}:group:{step}")
            batch = []
            for label in rng.sample(self.classes, self.chars_per_batch):
                pool = self.class_indices[label]
                if len(pool) < self.writers_per_char:
                    raise ValueError(f"class {label} has only {len(pool)} writers")
                batch.extend(pool[i] for i in
                             rng.sample(range(len(pool)), self.writers_per_char))
            yield batch


def class_index(store, split="train"):
    """Map class -> flat dataset indices, matching ReconstructionDataset ordering."""
    writers = [w for w in store.writers if w["split"] == split]
    ends = [0]
    for w in writers:
        ends.append(ends[-1] + len(w["labels"]))
    table = {}
    for wi, w in enumerate(writers):
        base = ends[wi]
        for row, label in enumerate(w["labels"]):
            table.setdefault(label, []).append(base + row)
    return table



def amp(device, precision):
    return (torch.autocast("cuda", dtype=torch.bfloat16)
            if device.type == "cuda" and precision == "bf16" else nullcontext())


def atomic_save(state, path):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".vae-", suffix=".tmp")
    os.close(fd)
    try:
        torch.save(state, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def preview(inputs, means, samples, path):
    n = min(8, len(inputs))
    cell = inputs.shape[-1] * 2
    canvas = Image.new("RGB", (cell * 3, 22 + n * cell), "white")
    draw = ImageDraw.Draw(canvas)
    for j, label in enumerate(("Input", "Decode posterior mean", "Decode posterior sample")):
        draw.text((j * cell + 2, 4), label, fill="black")
    for i in range(n):
        for j, tensor in enumerate((inputs, means, samples)):
            pixels = ((tensor[i].detach().float().cpu().squeeze().clamp(-1, 1) + 1) * 127.5)
            tile = Image.fromarray(pixels.round().byte().numpy()).resize(
                (cell, cell), Image.Resampling.NEAREST)
            canvas.paste(tile, (j * cell, 22 + i * cell))
    canvas.save(path)


@torch.no_grad()
def evaluate(model, loader, device, precision, beta, *, seed, preview_path=None):
    """Fixed posterior noise, independent of training RNG. No prior-sampling claim."""
    was_training = model.training
    model.eval()
    totals = {k: 0.0 for k in ("mse_mean", "mse_sample", "mae_mean_255", "kl", "kl_per_image",
                               "char_pos", "char_neg")}
    count = 0
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.manual_seed(seed)
            for raw in loader:
                images = raw["image"].to(device).float() / 127.5 - 1
                with amp(device, precision):
                    mu, logvar = model.encode(images)
                    decoded_mean = model.decode(mu)
                    decoded_sample = model.decode(model.reparameterize(mu, logvar))
                    terms = vae_loss(decoded_sample, images, mu, logvar, beta)
                dm = decoded_mean.float()
                if not all(torch.isfinite(x).all() for x in (dm, decoded_sample, mu, logvar)):
                    raise FloatingPointError("non-finite validation output")
                n = len(images)
                totals["mse_mean"] += (dm - images).square().mean().item() * n
                totals["mse_sample"] += terms["mse"].item() * n
                totals["mae_mean_255"] += (dm - images).abs().mean().item() * 127.5 * n
                for k in ("kl", "kl_per_image"):
                    totals[k] += terms[k].item() * n
                labels = raw.get("class_index")
                if torch.is_tensor(labels) and len(labels) > 1:
                    _, positive, negative = char_cosine_terms(mu, labels)
                    totals["char_pos"] += positive.item() * n
                    totals["char_neg"] += negative.item() * n
                if count == 0 and preview_path is not None:
                    preview(images, dm, decoded_sample, preview_path)
                count += n
        if not count:
            raise ValueError("empty validation set")
        result = {k: v / count for k, v in totals.items()}
        result["char_gap"] = result.pop("char_pos") - result.pop("char_neg")
        result.update(samples=count,
                      total_sample=result["mse_sample"] + beta * result["kl"],
                      psnr_mean_db=10 * math.log10(4 / max(result["mse_mean"], 1e-12)))
        return result
    finally:
        model.train(was_training)


def run(args):
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                          if args.device == "auto" else args.device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    config = TrainConfig(**{k: getattr(args, k) for k in TrainConfig.__dataclass_fields__})
    store = GlyphStore(args.data)
    mc = VAEConfig(image_size=store.image_size, base_channels=args.base_channels,
                   latent_channels=args.latent_channels,
                   latent_downsample=args.latent_downsample)
    if device.type == "cuda" and config.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("GPU does not support bf16; use --precision fp32")
    out = Path(args.output)
    checkpoint = None
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        # Checkpoints written before latent_downsample existed used 8.
        saved_model = dict(checkpoint["model_config"])
        saved_model.setdefault("latent_downsample", 8)
        defaults = TrainConfig()
        saved_training = dict(checkpoint["training_config"])
        for field in TrainConfig.__dataclass_fields__:
            saved_training.setdefault(field, getattr(defaults, field))
        saved_run = dict(json.loads((Path(args.output) / "run.json").read_text(encoding="utf-8"))["model"]) \
            if (Path(args.output) / "run.json").exists() else None
        if saved_run is not None:
            saved_run.setdefault("latent_downsample", 8)
        expected = (asdict(mc), asdict(config), store.fingerprint)
        actual = (saved_model, saved_training, checkpoint["data_fingerprint"])
        if actual != expected:
            raise ValueError("resume model/training settings or dataset differ; repeat saved CLI settings")
        if not out.exists() or not (out / "run.json").exists():
            raise ValueError("resume requires the original run output directory")
        metadata = json.loads((out / "run.json").read_text(encoding="utf-8"))
        if saved_run != expected[0] or metadata["training"] != expected[1] or metadata["data_fingerprint"] != expected[2]:
            raise ValueError("resume output belongs to another run")
        if checkpoint.get("device_type", device.type) != device.type:
            raise ValueError("resume must use the same device type as the checkpoint "
                             f"({checkpoint.get('device_type')} != {device.type}); "
                             "RNG streams are per device")
        if (out / "last.pt").exists():
            latest = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
            if latest["step"] > checkpoint["step"]:
                raise ValueError("resume checkpoint is older than last.pt")
            del latest
    elif out.exists() and any(out.iterdir()):
        raise ValueError("output must be new/empty; refusing overwrite")
    step = checkpoint["step"] if checkpoint is not None else 0
    if step >= config.steps:
        raise ValueError("checkpoint has already reached the step budget")
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    model = GlyphVAE(mc).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                  betas=(0.9, 0.999), weight_decay=config.weight_decay)
    best = float("inf")
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        best = checkpoint["best_handwriting_mse"]
        torch.set_rng_state(checkpoint["cpu_rng"])
        if device.type == "cuda" and checkpoint["cuda_rng"] is not None:
            torch.cuda.set_rng_state(checkpoint["cuda_rng"], device)
    del checkpoint
    data = ReconstructionDataset(store, standard_ratio=config.standard_ratio,
                                 length=config.steps * config.batch_size, seed=config.seed)
    # Dedicated loader generators keep iterator creation/prefetch out of the
    # model's posterior-noise RNG, including after checkpoint resume.
    if config.chars_per_batch > 0:
        grouped = ReconstructionDataset(store, split="train", source="handwriting")
        sampler = GroupedBatchSampler(class_index(store), config.chars_per_batch,
                                      config.writers_per_char, step, config.steps,
                                      config.seed)
        loader = DataLoader(grouped, batch_sampler=sampler, num_workers=config.workers,
                            pin_memory=device.type == "cuda")
    else:
        loader = DataLoader(data, batch_size=config.batch_size,
                            sampler=range(step * config.batch_size, len(data)),
                            num_workers=config.workers, pin_memory=device.type == "cuda",
                            generator=torch.Generator().manual_seed(config.seed + 50))
    val_loaders = {}
    for source in ("handwriting", "standard"):
        ds = ReconstructionDataset(store, split="val", source=source)
        indices = spread_indices(len(ds), config.val_samples)
        val_loaders[source] = DataLoader(Subset(ds, indices), batch_size=config.batch_size,
                                         num_workers=0, generator=torch.Generator().manual_seed(100))
    out.mkdir(parents=True, exist_ok=True)
    (out / "previews").mkdir(exist_ok=True)
    if args.resume is None:
        (out / "run.json").write_text(json.dumps({"model": asdict(mc), "training": asdict(config),
            "data_fingerprint": store.fingerprint, "parameters": sum(p.numel() for p in model.parameters()),
            "latent_shape": mc.latent_shape, "standard_validation_note":
            "Same printed glyph bank as training when ratio > 0; not unseen-font generalization."}, indent=2), encoding="utf-8")
    print(f"device={device} parameters={sum(p.numel() for p in model.parameters()):,} "
          f"latent={mc.latent_shape} standard_ratio={config.standard_ratio} start_step={step}", flush=True)
    stop = [False]
    def request_stop(_signum, _frame):
        stop[0] = True
    previous_handlers = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    def save():
        atomic_save({"version": 1, "step": step, "model_config": asdict(mc),
                     "training_config": asdict(config), "data_fingerprint": store.fingerprint,
                     "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                     "best_handwriting_mse": best, "cpu_rng": torch.get_rng_state(),
                     "device_type": device.type,
                     "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None}, out / "last.pt")
    started = time.monotonic()
    model.train()
    try:
        for raw in loader:
            if stop[0]:
                break
            current = step + 1
            lr, beta, cos_weight = schedule(config, current)
            for group in optimizer.param_groups:
                group["lr"] = lr
            images = raw["image"].to(device).float() / 127.5 - 1
            optimizer.zero_grad(set_to_none=True)
            with amp(device, config.precision):
                decoded, mu, logvar = model(images, sample=True)
                terms = vae_loss(decoded, images, mu, logvar, beta)
            cos_loss, cos_positive, cos_negative = char_cosine_terms(
                mu, raw["class_index"], config.char_cos_temperature)
            total = terms["loss"] + cos_weight * cos_loss
            if not torch.isfinite(total):
                raise FloatingPointError("non-finite VAE training loss; previous checkpoints preserved")
            total.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip, error_if_nonfinite=True)
            optimizer.step()
            step = current
            record = {"step": step, **{k: v.item() for k, v in terms.items()},
                      "beta": beta, "lr": lr, "grad_norm": float(grad),
                      "char_cos_weight": cos_weight, "char_cos_loss": cos_loss.item(),
                      "char_cos_pos": cos_positive.item(), "char_cos_neg": cos_negative.item(),
                      "char_cos_gap": (cos_positive - cos_negative).item(), "total_loss": total.item(),
                      "standard_fraction": raw["is_standard"].float().mean().item(),
                      "elapsed_seconds": time.monotonic() - started}
            validate = step % config.validate_every == 0 or step == config.steps
            improved = False
            if validate and not stop[0]:
                for source, val_loader in val_loaders.items():
                    record["val_" + source] = evaluate(model, val_loader, device, config.precision,
                        config.beta, seed=config.seed + 1000,
                        preview_path=out / "previews" / f"{source}_{step:07d}.png")
                score = record["val_handwriting"]["mse_mean"]
                improved = score < best
                best = min(best, score)
            if step % config.log_every == 0 or validate or stop[0]:
                with (out / "metrics.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record, allow_nan=False) + "\n")
                print(json.dumps(record, allow_nan=False), flush=True)
            if step % config.save_every == 0 or validate or stop[0]:
                save()
                if improved:
                    # Separate atomic writes, so best.pt is never a hard link to last.pt.
                    state = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
                    atomic_save(state, out / "best.pt")
            if stop[0]:
                break
        save()
        print(f"Saved step {step}; {'stopped' if stop[0] else 'completed'}: {out / 'last.pt'}", flush=True)
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
        store.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/prepared"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--latent-channels", type=int, default=4)
    parser.add_argument("--latent-downsample", type=int, default=8)
    for name, field in TrainConfig.__dataclass_fields__.items():
        parser.add_argument("--" + name.replace("_", "-"), type=type(field.default), default=field.default)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
