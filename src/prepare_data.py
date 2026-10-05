"""Prepare extracted CASIA HWDB training GNTs; never include official test GNTs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random

from fontTools.ttLib import TTFont
import numpy as np
from PIL import ImageFont

from .data import iter_gnt, normalize_image, render_character
from .config import ModelConfig, TrainingConfig


def prepare(args):
    if not 0 < args.val_fraction < 1:
        raise ValueError("val-fraction must be between 0 and 1")
    if args.min_characters < 2:
        raise ValueError("min-characters must be at least 2")
    if args.image_size < 4 or not 0 <= args.margin < args.image_size / 2:
        raise ValueError("invalid image-size or margin")
    files = sorted(args.gnt_root.rglob("*"))
    files = [p for p in files if p.is_file() and p.suffix.lower() == ".gnt"]
    if len(files) < 2:
        raise ValueError("need at least two writer GNT files")
    if len({p.stem for p in files}) != len(files):
        raise ValueError("duplicate GNT stems: use original files, one file per writer")
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("output must be a new or empty directory; refusing to overwrite")
    # Check cmap, otherwise Pillow silently renders missing characters as boxes.
    with TTFont(str(args.font), fontNumber=args.font_index) as metadata:
        supported = set(metadata.getBestCmap() or {})
    font = ImageFont.truetype(str(args.font), args.image_size * 4, index=args.font_index)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "writers").mkdir()
    characters, char_to_index, standards, writers = [], {}, [], []
    skipped = {"non_cjk": 0, "missing_font": 0, "blank": 0,
               "duplicate_character": 0, "small_writer": 0}
    for file_number, path in enumerate(files):
        images, labels, seen = [], [], set()
        for character, image in iter_gnt(path):
            if len(character) != 1 or not 0x4E00 <= ord(character) <= 0x9FFF:
                skipped["non_cjk"] += 1
                continue
            if ord(character) not in supported:
                skipped["missing_font"] += 1
                continue
            if character in seen:
                skipped["duplicate_character"] += 1
                continue
            try:
                normalized = normalize_image(image, args.image_size, args.margin)
                if character not in char_to_index:
                    standard = render_character(character, font, args.image_size, args.margin)
                    char_to_index[character] = len(characters)
                    characters.append(character)
                    standards.append(standard)
            except ValueError:
                skipped["blank"] += 1
                continue
            seen.add(character)
            images.append(normalized)
            labels.append(char_to_index[character])
        if len(images) < args.min_characters:
            skipped["small_writer"] += 1
            continue
        relative = f"writers/{file_number:05d}.npy"
        output_file = args.output / relative
        np.save(output_file, np.stack(images))
        writers.append({"id": path.stem, "file": relative, "labels": labels,
                        "sha256": hashlib.sha256(output_file.read_bytes()).hexdigest()})
        print(f"[{file_number + 1}/{len(files)}] {path.name}: {len(images)} characters", flush=True)
    if len(writers) < 2:
        raise ValueError("fewer than two usable writers; check font and min-characters")
    order = list(range(len(writers)))
    random.Random(args.seed).shuffle(order)
    val_count = min(len(writers) - 1, max(1, round(len(writers) * args.val_fraction)))
    validation = set(order[:val_count])
    for index, writer in enumerate(writers):
        writer["split"] = "val" if index in validation else "train"
    np.save(args.output / "standards.npy", np.stack(standards))
    manifest = {
        "version": 1, "image_size": args.image_size, "margin": args.margin,
        "normalization": "white=+1, black=-1; tight crop, aspect-fit, centered",
        "font_name": args.font.name, "font_index": args.font_index,
        "font_sha256": hashlib.sha256(args.font.read_bytes()).hexdigest(),
        "standards_sha256": hashlib.sha256((args.output / "standards.npy").read_bytes()).hexdigest(),
        "seed": args.seed, "characters": characters, "writers": writers, "skipped": skipped,
    }
    # Manifest is the completion marker: an interrupted preparation cannot train.
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Prepared {len(writers) - val_count} train / {val_count} val writers; "
          f"{len(characters)} characters. Skipped: {skipped}")


def main():
    model_config = ModelConfig()
    training_config = TrainingConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gnt-root", type=Path, required=True)
    parser.add_argument("--font", type=Path, required=True)
    parser.add_argument("--font-index", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=model_config.image_size)
    parser.add_argument("--margin", type=int, default=4)
    parser.add_argument("--min-characters", type=int, default=training_config.max_references + 1)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=training_config.seed)
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
