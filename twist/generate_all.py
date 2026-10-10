"""Render one same-character transition for every character with >=2 valid images."""
from __future__ import annotations

import argparse
from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import asdict
import hashlib
import html
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from twist import ModelConfig, Twist
from twist.images import NORMALIZATION, normalize_minmax
from twist.preview import tile


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Trained version=3 best.pt or last.pt")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "prepared")
    parser.add_argument("--output", type=Path, help="Default: checkpoint directory/all-character-transitions")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda or cuda:N")
    parser.add_argument("--precision", choices=("auto", "fp32", "bf16"), default="auto")
    parser.add_argument("--split", choices=("all", "train", "val"), default="all",
                        help="all includes pairs across train/val when needed")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--frames", type=int, default=21)
    parser.add_argument("--display-size", type=int, default=256, help="Display pixels per panel; model resolution is unchanged")
    parser.add_argument("--seed", type=int, default=42, help="Reproducible two-writer sampling per character")
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--resume", action="store_true", help="Continue the same export and skip completed characters")
    return parser.parse_args(argv)


def resize_pixels(pixels, image_size):
    images = torch.from_numpy(np.array(pixels, copy=True)).unsqueeze(1).float()
    if images.shape[-2:] != (image_size, image_size):
        images = F.interpolate(images, size=(image_size, image_size), mode="bilinear",
                               align_corners=False, antialias=True)
    return images


class CharacterImages:
    """Scan one writer at a time; keep two samples per character, not all pixels."""

    def __init__(self, root, image_size):
        self.root = Path(root).resolve()
        raw = (self.root / "manifest.json").read_bytes()
        self.fingerprint = hashlib.sha256(raw).hexdigest()
        self.manifest = json.loads(raw)
        if self.manifest.get("version") != 1:
            raise ValueError("expected prepared manifest version 1")
        self.characters = self.manifest["characters"]
        self.writers = self.manifest["writers"]
        self.prepared_size = self.manifest["image_size"]
        self.image_size = image_size
        self.maps = OrderedDict()
        if not self.characters or len(set(self.characters)) != len(self.characters):
            raise ValueError("characters must be nonempty and unique")
        if any(not isinstance(char, str) or len(char) != 1 for char in self.characters):
            raise ValueError("each character label must contain one character")
        if not self.writers or len({w["id"] for w in self.writers}) != len(self.writers):
            raise ValueError("writers must have unique IDs")
        if type(self.prepared_size) is not int or self.prepared_size < 2:
            raise ValueError("invalid prepared image_size")
        for writer in self.writers:
            labels = writer["labels"]
            if (writer["split"] not in ("train", "val") or len(set(labels)) != len(labels)
                    or any(type(label) is not int or not 0 <= label < len(self.characters) for label in labels)):
                raise ValueError(f"invalid split or labels for writer {writer['id']}")

    def open_writer(self, number):
        if number in self.maps:
            self.maps.move_to_end(number)
            return self.maps[number]
        writer = self.writers[number]
        images = np.load(self.root / writer["file"], mmap_mode="r", allow_pickle=False)
        expected = (len(writer["labels"]), self.prepared_size, self.prepared_size)
        if images.dtype != np.uint8 or images.shape != expected:
            images._mmap.close()
            raise ValueError(f"invalid prepared pixels for writer {writer['id']}")
        self.maps[number] = images
        # Selected samples are copied before eviction; no Tensor aliases a mmap.
        if len(self.maps) > 8:
            _, previous = self.maps.popitem(last=False)
            previous._mmap.close()
        return images

    def close(self):
        for images in self.maps.values():
            images._mmap.close()
        self.maps.clear()

    def select_pairs(self, split, seed):
        generator = random.Random(seed)
        counts = [0] * len(self.characters)
        reservoirs = [[] for _ in self.characters]
        scanned = excluded = global_start = 0
        for number, writer in enumerate(self.writers):
            if split != "all" and writer["split"] != split:
                global_start += len(writer["labels"])
                continue
            images = self.open_writer(number)
            for offset in range(0, len(images), 128):
                block = images[offset:offset + 128]
                if self.image_size == self.prepared_size:
                    valid = block.min(axis=(1, 2)) != block.max(axis=(1, 2))
                else:
                    resized = resize_pixels(block, self.image_size)
                    valid = (resized.amin(dim=(1, 2, 3)) != resized.amax(dim=(1, 2, 3))).numpy()
                scanned += len(block)
                excluded += int((~valid).sum())
                for local in np.flatnonzero(valid):
                    row = offset + int(local)
                    label = writer["labels"][row]
                    counts[label] += 1
                    reference = {"writer_number": number, "writer": writer["id"],
                                 "split": writer["split"], "file": writer["file"],
                                 "row": row, "index": global_start + row}
                    samples = reservoirs[label]
                    if counts[label] <= 2:
                        samples.append(reference)
                    else:
                        position = generator.randrange(counts[label])
                        if position < 2:
                            samples[position] = reference
            global_start += len(writer["labels"])
            if (number + 1) % 20 == 0 or number + 1 == len(self.writers):
                print(f"Scanning writers {number+1}/{len(self.writers)}; images={scanned:,}; constant={excluded:,}", flush=True)
        self.close()
        records, skipped = [], []
        for label, (character, samples) in enumerate(zip(self.characters, reservoirs)):
            if counts[label] < 2:
                skipped.append({"label": label, "character": character, "valid_images": counts[label]})
                continue
            generator.shuffle(samples)
            if samples[0]["writer"] == samples[1]["writer"]:
                raise ValueError("selected pair must use two different writers")
            stem = f"u{ord(character):04x}"
            records.append({"label": label, "character": character, "unicode": f"U+{ord(character):04X}",
                            "valid_images": counts[label], "source": samples[0], "target": samples[1],
                            "gif": f"transitions/{stem}.gif", "registration": f"registration/{stem}.png"})
        return records, {"scanned_images": scanned, "constant_images": excluded,
                         "total_characters": len(self.characters), "eligible_characters": len(records),
                         "skipped_characters": skipped}

    def batch(self, records, device):
        pixels = []
        for side in ("source", "target"):
            for record in records:
                ref = record[side]
                pixels.append(np.array(self.open_writer(ref["writer_number"])[ref["row"]], copy=True))
        images = normalize_minmax(resize_pixels(np.stack(pixels), self.image_size)).to(device)
        return images[:len(records)], images[len(records):]


def file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def image_bytes(images):
    return ((images.float().clamp(-1, 1) + 1) * 127.5).round().byte().cpu().numpy()


def row_canvas(images, labels, size):
    panels = [tile(Image.fromarray(pixels), label, size=size) for pixels, label in zip(images, labels)]
    canvas = Image.new("RGB", (size * len(panels), size + 24), "white")
    for column, panel in enumerate(panels):
        canvas.paste(panel, (column * size, 0))
    return canvas


def save_character(record, output, frames, endpoints, times, display_size):
    source, registered_source, target, registered_target = endpoints
    comparisons = [row_canvas((source, registered_source, target), ("A", "A(exp(v))", "B"), display_size),
                   row_canvas((target, registered_target, source), ("B", "B(exp(-v))", "A"), display_size)]
    comparison = Image.new("RGB", (comparisons[0].width, comparisons[0].height * 2), "white")
    for row, panel in enumerate(comparisons):
        comparison.paste(panel, (0, row * panel.height))
    png = output / record["registration"]
    temporary = png.with_suffix(".png.tmp")
    comparison.save(temporary, format="PNG")
    temporary.replace(png)
    animation = [row_canvas((source, frame, target),
                            ("A", f"t={t:.3f}  #{index+1}/{len(times)}", "B"), display_size)
                 for index, (frame, t) in enumerate(zip(frames, times))]
    durations = [80] * len(times)
    durations[0] = durations[-1] = 600
    gif = output / record["gif"]
    temporary = gif.with_suffix(".gif.tmp")
    animation[0].save(temporary, format="GIF", save_all=True, append_images=animation[1:],
                      duration=durations, loop=0)
    temporary.replace(gif)


def completed(record, output):
    return (output / record["gif"]).is_file() and (output / record["registration"]).is_file()


def write_gallery(output, records, step):
    cards = []
    for record in records:
        source, target = record["source"], record["target"]
        caption = f"{record['character']} · {record['unicode']} · {source['writer']} ({source['split']}) → {target['writer']} ({target['split']})"
        cards.append(f'<article data-search="{html.escape(caption.lower(), quote=True)}">'
                     f'<h2>{html.escape(record["character"])} <small>{record["unicode"]}</small></h2>'
                     f'<p>{html.escape(caption)}</p><a href="{record["gif"]}" target="_blank">'
                     f'<img loading="lazy" src="{record["gif"]}" alt="{html.escape(caption, quote=True)}"></a>'
                     f'<details><summary>查看双向未混合配准</summary><a href="{record["registration"]}" target="_blank">'
                     f'<img loading="lazy" src="{record["registration"]}" alt="双向配准"></a></details></article>')
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Twist 汉字过渡</title>
<style>body{margin:24px;background:#f4f5f7;color:#18202d;font:16px system-ui,sans-serif}h1{margin-bottom:8px}
input{width:min(440px,90%);padding:12px;border:1px solid #cbd0da;border-radius:8px;margin:12px 0}
main{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(420px,100%),1fr));gap:20px}
article{padding:16px;background:white;border-radius:12px}h2{margin:0}small{color:#687080;font:14px system-ui}
p{color:#596273;font-size:14px;overflow-wrap:anywhere}img{width:100%;height:auto}summary{cursor:pointer;margin-top:10px}
[hidden]{display:none}</style>
'''
    page += (f'<h1>Twist 汉字过渡</h1><p>检查点 step={step}，已生成 {len(records)} 个汉字。'
             'GIF 左侧为 A，中间为过渡，右侧为 B；展开配准图可查看未混合端点。</p>'
             '<input id="search" placeholder="搜索汉字、Unicode 或书写者"><p id="count"></p><main>'
             + "\n".join(cards) + '</main><script>const cards=[...document.querySelectorAll("article")];'
             'const search=document.querySelector("#search"),count=document.querySelector("#count");'
             'function filter(){const query=search.value.trim().toLowerCase();let shown=0;'
             'for(const card of cards){card.hidden=!card.dataset.search.includes(query);if(!card.hidden)shown++;}'
             'count.textContent=`显示 ${shown} / ${cards.length} 个汉字`;}'
             'search.addEventListener("input",filter);filter();</script></html>')
    (output / "index.html").write_text(page, encoding="utf-8")


@torch.inference_mode()
def generate(args):
    if (args.batch_size < 1 or args.frames < 2 or args.display_size < 128 or args.cpu_threads < 1
            or not 0 <= args.seed < 2**63):
        raise ValueError("batch-size/threads must be positive, frames>=2, display-size>=128, seed in [0,2**63)")
    torch.set_num_threads(args.cpu_threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    if device.type not in ("cpu", "cuda") or (device.type == "cuda" and not torch.cuda.is_available()):
        raise ValueError("choose an available cpu or cuda device")
    if device.type == "cuda":
        device = torch.device("cuda", torch.cuda.current_device() if device.index is None else device.index)
        torch.cuda.set_device(device)
    supports_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    precision = ("bf16" if supports_bf16 else "fp32") if args.precision == "auto" else args.precision
    if precision == "bf16" and not supports_bf16:
        raise ValueError("bf16 is unavailable; use --precision fp32")
    checkpoint_path = args.checkpoint.resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("kind") != "twist-all-pair" or checkpoint.get("version") != 3:
        raise ValueError("expected pixel-resolution twist-all-pair checkpoint version 3")
    if checkpoint.get("normalization") != NORMALIZATION:
        raise ValueError("checkpoint uses an unsupported normalization")
    config = ModelConfig(**checkpoint["model_config"])
    if config.in_channels != 1:
        raise ValueError("prepared handwriting requires in_channels=1")
    output = (args.output or checkpoint_path.parent / "all-character-transitions").resolve()
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise ValueError(f"Output is nonempty: {output}; use --resume or a new --output")
    if args.resume and not (output / "generation.json").is_file():
        raise ValueError("--resume requires generation.json in the output directory")
    data = CharacterImages(args.data, config.image_size)
    settings = {"checkpoint_sha256": file_digest(checkpoint_path), "data_fingerprint": data.fingerprint,
                "model_config": asdict(config), "normalization": NORMALIZATION, "split": args.split,
                "seed": args.seed, "frames": args.frames, "display_size": args.display_size,
                "precision": precision}
    if args.resume:
        plan = json.loads((output / "generation.json").read_text(encoding="utf-8"))
        if plan.get("version") != 1 or plan["settings"] != settings:
            raise ValueError("Export settings/checkpoint/data differ; use a new --output")
    else:
        records, selection = data.select_pairs(args.split, args.seed)
        plan = {"version": 1, "checkpoint": str(checkpoint_path), "step": checkpoint["step"],
                "data_root": str(data.root), "settings": settings, "selection": selection, "pairs": records}
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / "generation.json", plan)
    for folder in ("transitions", "registration"):
        (output / folder).mkdir(exist_ok=True)
    model = Twist(config).to(device).eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    del checkpoint
    records = plan["pairs"]
    pending = [record for record in records if not completed(record, output)]
    done = len(records) - len(pending)
    print(f"Device={device}; precision={precision}; eligible characters={len(records):,}; "
          f"already complete={done:,}; frames={args.frames}; output={output}", flush=True)
    times = torch.linspace(0, 1, args.frames).tolist()
    started = time.monotonic()
    interrupted = False
    try:
        for offset in range(0, len(pending), args.batch_size):
            chunk = pending[offset:offset + args.batch_size]
            source, target = data.batch(chunk, device)
            context = torch.autocast("cuda", dtype=torch.bfloat16) if precision == "bf16" else nullcontext()
            with context:
                result = model(source, target)
            rendered = torch.stack([result.render(t).image[:, 0] for t in times])
            endpoints = torch.stack((source[:, 0], result.image[:, 0], target[:, 0], result.reverse_image[:, 0]))
            if not bool(torch.isfinite(rendered).all() & torch.isfinite(endpoints).all()):
                raise FloatingPointError(f"Nonfinite images in characters starting at {chunk[0]['unicode']}")
            frame_pixels, endpoint_pixels = image_bytes(rendered), image_bytes(endpoints)
            for position, record in enumerate(chunk):
                save_character(record, output, frame_pixels[:, position], endpoint_pixels[:, position], times, args.display_size)
                done += 1
            print(f"Generated {done:,}/{len(records):,} characters; elapsed={time.monotonic()-started:.1f}s", flush=True)
    except KeyboardInterrupt:
        interrupted = True
        print("Interrupted; completed files are preserved. Run the same command with --resume to continue.", flush=True)
    finally:
        data.close()
        finished = [record for record in records if completed(record, output)]
        write_gallery(output, finished, plan["step"])
        write_json(output / "summary.json", {"status": "interrupted" if interrupted else (
            "completed" if len(finished) == len(records) else "incomplete"),
            "eligible_characters": len(records), "generated_characters": len(finished),
            "elapsed_seconds_this_invocation": time.monotonic() - started})
    print(f"Gallery: {output / 'index.html'}", flush=True)
    return output


def main():
    generate(parse_args())


if __name__ == "__main__":
    main()
