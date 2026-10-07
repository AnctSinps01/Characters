"""Export the exact A/B pairs used by a full-dataset run's GIF previews."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from twist.images import NORMALIZATION, normalize_minmax
from twist.preview import tile


def read_input(root, manifest, writer, label, image_size, normalization):
    """Read one row, then apply the run's actual image preprocessing on CPU."""
    images = np.load(root / writer["file"], mmap_mode="r", allow_pickle=False)
    try:
        size = manifest["image_size"]
        if images.dtype != np.uint8 or images.shape != (len(writer["labels"]), size, size):
            raise ValueError(f"Invalid prepared pixels for writer {writer['id']}")
        pixels = np.array(images[writer["labels"].index(label)], copy=True)
    finally:
        del images
    image = torch.from_numpy(pixels)[None, None].float()
    # Older runs normalized the fixed uint8 range before resizing.
    if normalization != NORMALIZATION:
        image = image / 127.5 - 1
    if image_size != size:
        image = F.interpolate(image, size=(image_size, image_size), mode="bilinear",
                              align_corners=False, antialias=True)
    if normalization == NORMALIZATION:
        return normalize_minmax(image)
    return image


def export_inputs(run: Path, data: Path | None = None, output: Path | None = None) -> Path:
    run = run.resolve()
    info = json.loads((run / "run.json").read_text(encoding="utf-8"))
    pairs = json.loads((run / "preview_pairs.json").read_text(encoding="utf-8"))
    if not pairs:
        raise ValueError("This run has no recorded GIF preview pairs")
    root = (data if data is not None else Path(info["data_root"])).resolve()
    raw_manifest = (root / "manifest.json").read_bytes()
    if hashlib.sha256(raw_manifest).hexdigest() != info["data_fingerprint"]:
        raise ValueError("Prepared manifest differs from this training run")
    manifest = json.loads(raw_manifest)
    normalization = info.get("normalization", "fixed-uint8-range")
    if normalization not in (NORMALIZATION, "fixed-uint8-range"):
        raise ValueError(f"Unsupported normalization: {normalization}")
    image_size = info["model"]["image_size"]
    writers = {}
    start = 0
    for writer in manifest["writers"]:
        writers[writer["id"]] = (writer, start)
        start += len(writer["labels"])

    rows = []
    for number, pair in enumerate(pairs):
        label = pair["label"]
        character = manifest["characters"][label]
        if character != pair["character"]:
            raise ValueError(f"Pair {number} has an inconsistent character label")
        row = []
        splits = []
        for side, name in (("source", "A"), ("target", "B")):
            writer, start = writers[pair[f"{side}_writer"]]
            if label not in writer["labels"] or start + writer["labels"].index(label) != pair[f"{side}_index"]:
                raise ValueError(f"Pair {number} has an inconsistent {name} image index")
            splits.append(writer["split"])
            image = read_input(root, manifest, writer, label, image_size, normalization)
            row.append(tile(image, f"{number:02d} {name}: {writer['id']} | U+{ord(character):04X}"))
        if pair["source_writer"] == pair["target_writer"] or splits[0] != splits[1]:
            raise ValueError(f"Pair {number} must use different writers in the same split")
        rows.append(row)
    canvas = Image.new("RGB", (rows[0][0].width * 2, sum(row[0].height for row in rows)), "white")
    y = 0
    for a, b in rows:
        canvas.paste(a, (0, y))
        canvas.paste(b, (a.width, y))
        y += a.height
    output = (output if output is not None else ROOT / "inputs.png").resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG")
    print(f"Saved {len(pairs)} GIF input pairs (left=A, right=B): {output}", flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Existing training output directory")
    parser.add_argument("--data", type=Path, help="Override the prepared-data path saved in run.json")
    parser.add_argument("--output", type=Path, help="PNG destination; default: repository root/inputs.png")
    args = parser.parse_args()
    torch.set_num_threads(1)
    export_inputs(args.run, args.data, args.output)


if __name__ == "__main__":
    main()
