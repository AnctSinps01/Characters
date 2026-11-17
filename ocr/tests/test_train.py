"""Small end-to-end training smoke test using a synthetic prepared dataset."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from ocr.train import run


class OCRTrainingTests(unittest.TestCase):
    def test_short_training_writes_finite_metrics_and_checkpoints(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            data.mkdir()
            images = np.random.default_rng(4).integers(0, 256, (4, 16, 16), dtype=np.uint8)
            np.save(data / "train.npy", images[:2])
            np.save(data / "val.npy", images[2:])
            manifest = {
                "version": 1,
                "image_size": 16,
                "characters": ["甲", "乙"],
                "writers": [
                    {"id": "writer-train", "file": "train.npy", "labels": [0, 1], "split": "train"},
                    {"id": "writer-val", "file": "val.npy", "labels": [0, 1], "split": "val"},
                ],
            }
            (data / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            output = root / "run"
            args = SimpleNamespace(
                seed=3, device="cpu", data=data, output=output,
                image_size=16, patch_size=4, hidden_size=16, depth=1,
                num_heads=4, mlp_ratio=2.0, dropout=0.0,
                steps=2, batch_size=2, learning_rate=1e-3, min_lr_ratio=0.1,
                warmup_steps=1, weight_decay=0.0, gradient_clip=1.0,
                workers=0, validate_every=1, val_batches=1, save_every=1,
                log_every=1, precision="fp32", resume=None,
            )
            run(args)
            rows = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
            self.assertEqual([row["step"] for row in rows], [1, 2])
            self.assertTrue(all(np.isfinite(row["train_loss"]) for row in rows))
            self.assertTrue(all(np.isfinite(row["val_loss"]) for row in rows))
            self.assertTrue((output / "last.pt").is_file())
            self.assertTrue((output / "best.pt").is_file())
            checkpoint = torch.load(output / "last.pt", map_location="cpu", weights_only=False)
            self.assertEqual(checkpoint["step"], 2)
            self.assertEqual(checkpoint["characters"], ["甲", "乙"])


if __name__ == "__main__":
    unittest.main()
