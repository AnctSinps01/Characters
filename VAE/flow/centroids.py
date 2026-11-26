"""Per-character mean codes over the training writers, used as an oracle condition.

Only for diagnostics: the flow gets the true character centroid instead of the
standard glyph's content code, which separates "the condition is too weak" from
"the architecture cannot learn the transport".
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latents", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    root = Path(args.latents)
    latent = np.load(root / "train_latent.npy", mmap_mode="r")
    label = np.load(root / "train_label.npy", mmap_mode="r")
    count = int(max(label)) + 1
    sums = np.zeros((count, 4, 8, 8), np.float64)
    counts = np.zeros(count, np.float64)
    chunk = 20_000
    for start in range(0, len(latent), chunk):
        block = np.asarray(latent[start:start + chunk], np.float32)
        rows = np.asarray(label[start:start + chunk])
        np.add.at(sums, rows, block)
        np.add.at(counts, rows, 1)
    means = sums / np.maximum(counts, 1)[:, None, None, None]
    means[counts == 0] = 0
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "centroid_latent.npy", means.astype(np.float16))
    np.save(out / "centroid_count.npy", counts)
    print(f"wrote {out / 'centroid_latent.npy'} shape={means.shape} covered={(counts > 0).sum()}/{count}")


if __name__ == "__main__":
    main()
