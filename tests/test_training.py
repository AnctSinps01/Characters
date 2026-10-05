from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

from src.prepare_data import prepare
from src import ModelConfig, TrainingConfig
from src.data import HandwritingEpisodes, iter_gnt, normalize_image
from src.train import learning_rate, optimizer_update, run


def record(character="啊", width=5, height=7):
    pixels = np.full((height, width), 255, dtype=np.uint8)
    pixels[1:-1, 1:-1] = 60
    return struct.pack("<I2sHH", 10 + width * height, character.encode("gbk"), width, height) + pixels.tobytes()


def fixture_data(root):
    root.mkdir()
    standards = np.stack([np.full((16, 16), i * 30, dtype=np.uint8) for i in range(6)])
    np.save(root / "standards.npy", standards)
    writers = []
    for writer in range(3):
        name = f"writer{writer}.npy"
        np.save(root / name, standards + writer)
        writers.append({"id": str(writer), "file": name, "labels": list(range(6)),
                        "split": "train" if writer < 2 else "val"})
    (root / "manifest.json").write_text(json.dumps({
        "version": 1, "image_size": 16, "writers": writers,
        "characters": list("啊阿埃挨哎唉"),
    }), encoding="utf-8")


class DataTests(unittest.TestCase):
    def test_reference_range_includes_one_and_configured_maximum(self):
        self.assertEqual(TrainingConfig().min_references, 1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data"
            fixture_data(root)
            for split in ("train", "val"):
                for maximum in (1, 5):
                    config = TrainingConfig(max_references=maximum)
                    dataset = HandwritingEpisodes(
                        root, split, length=64, min_references=config.min_references,
                        max_references=config.max_references, seed=config.seed)
                    try:
                        counts = set()
                        for index in range(len(dataset)):
                            episode = dataset[index]
                            counts.add(episode["reference_mask"].sum().item())
                            self.assertEqual(episode["reference_handwriting"].shape[0], maximum)
                        self.assertEqual(counts, set(range(1, maximum + 1)))
                    finally:
                        dataset.close()
        for values in ({"max_references": 0}, {"min_references": 0},
                       {"min_references": 3, "max_references": 2}):
            with self.assertRaises(ValueError):
                TrainingConfig(**values)

    def test_gnt_endianness_multiple_records_and_truncation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "1001.gnt"
            raw = record("啊") + record("永", 9, 3)
            path.write_bytes(raw)
            items = list(iter_gnt(path))
            self.assertEqual([item[0] for item in items], ["啊", "永"])
            self.assertEqual(items[1][1].size, (9, 3))
            self.assertEqual(np.array(items[0][1])[1, 1], 60)
            for invalid in (raw[:-1], raw[:6], b"\x00\x00\x00\x00" + raw[4:]):
                path.write_bytes(invalid)
                with self.assertRaises(ValueError):
                    list(iter_gnt(path))

    def test_preprocess_preserves_aspect_and_rejects_blank(self):
        image = Image.new("L", (4, 8), 0)
        result = normalize_image(image, 16, 2)
        self.assertEqual(result.shape, (16, 16))
        ys, xs = np.where(result < 128)
        self.assertEqual((xs.max() - xs.min() + 1, ys.max() - ys.min() + 1), (6, 12))
        self.assertTrue((result[:2] == 255).all())
        with self.assertRaises(ValueError):
            normalize_image(Image.new("L", (8, 8), 255), 16, 2)

    def test_episode_exclusion_mask_and_worker_independence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data"
            fixture_data(root)
            dataset = HandwritingEpisodes(root, "train", length=12, min_references=1, max_references=3, seed=5)
            self.assertEqual({w["id"] for w in dataset.writers}, {"0", "1"})
            for index in range(12):
                item = dataset[index]
                valid = item["reference_mask"]
                references = item["reference_standard"][valid, 0, 0, 0]
                target = item["target_standard"][0, 0, 0]
                self.assertFalse((references == target).any())
                self.assertEqual(len(references.unique()), len(references))
                self.assertTrue((item["reference_handwriting"][~valid] == 1).all())
                torch.testing.assert_close(item["target_standard"], dataset[index]["target_standard"])
            for direct, worker in zip(DataLoader(dataset, batch_size=3, num_workers=0),
                                      DataLoader(dataset, batch_size=3, num_workers=2)):
                for key in direct:
                    torch.testing.assert_close(direct[key], worker[key])
            dataset.close()

    def test_prepare_real_font_and_writer_split(self):
        font = Path(os.environ.get("GLYPH2HAND_TEST_FONT", "C:/Windows/Fonts/msyh.ttc"))
        if not font.exists():
            self.skipTest("set GLYPH2HAND_TEST_FONT to a font containing CJK glyphs")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "gnt"
            source.mkdir()
            for writer in range(3):
                (source / f"{writer}.gnt").write_bytes(b"".join(record(c) for c in "啊永水木"))
            args = argparse.Namespace(gnt_root=source, output=root / "ready", font=font,
                                      font_index=0, image_size=16, margin=2,
                                      val_fraction=.34, seed=7, min_characters=3)
            with redirect_stdout(io.StringIO()):
                prepare(args)
            manifest = json.loads((args.output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(len(manifest["writers"]), 3)
            self.assertEqual(sum(w["split"] == "val" for w in manifest["writers"]), 1)
            self.assertEqual(set(manifest["characters"]), set("啊永水木"))
            self.assertTrue((np.load(args.output / "standards.npy") < 128).any())
            with self.assertRaises(ValueError):
                prepare(args)


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_schedule_boundaries(self):
        config = TrainingConfig(steps=10, warmup_steps=2)
        self.assertAlmostEqual(learning_rate(1, config), config.learning_rate / 2)
        self.assertAlmostEqual(learning_rate(2, config), config.learning_rate)
        self.assertAlmostEqual(learning_rate(10, config), config.learning_rate * config.min_lr_ratio)

    def test_scaler_overflow_retries_without_advancing_optimizer_or_noise(self):
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scaler = torch.amp.GradScaler("cpu", init_scale=2.0**120)
        config = TrainingConfig(accumulation_steps=1)
        noises = []

        def large_loss(model, batch):
            noise = torch.rand(1, 1)
            noises.append(noise.clone())
            return model(noise).sum() * 1e5

        with patch("src.train.batch_loss", side_effect=large_loss):
            loss, norm = optimizer_update(model, optimizer, scaler, [{}],
                                          torch.device("cpu"), "fp32", config)
        self.assertGreater(len(noises), 1)
        for noise in noises[1:]:
            torch.testing.assert_close(noises[0], noise, rtol=0, atol=0)
        self.assertTrue(np.isfinite(loss) and np.isfinite(norm))
        self.assertTrue(all(state["step"].item() == 1 for state in optimizer.state.values()))

    def test_train_preview_and_resume_equal_uninterrupted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            fixture_data(data)
            model = ModelConfig(image_size=16, patch_size=4, hidden_size=16, num_heads=2,
                                mlp_ratio=2, local_depth=1, generator_depth=1,
                                description_tokens=2, style_tokens=2, time_frequency_size=16)
            config = TrainingConfig(steps=3, warmup_steps=1, batch_size=2, accumulation_steps=2,
                                    min_references=1, max_references=2, workers=0, val_batches=1,
                                    validate_every=3, save_every=1, log_every=1, sample_steps=2,
                                    precision="fp32")
            args = argparse.Namespace(data=data, output=root / "full", resume=None,
                                      device="cpu", workers=0, stop_after=None)
            with redirect_stdout(io.StringIO()):
                # New runs read Python defaults; no configuration file is needed.
                with patch("src.train.ModelConfig", return_value=model), \
                     patch("src.train.TrainingConfig", return_value=config):
                    run(args)
                    args.output = root / "split"
                    args.stop_after = 1
                    run(args)
                # Restore the actual (different) defaults before resuming. Saved
                # settings must win, otherwise image size/reference counts fail.
                args.resume = args.output / "last.pt"
                args.stop_after = None
                run(args)
            full = torch.load(root / "full/last.pt", weights_only=True)
            resumed = torch.load(root / "split/last.pt", weights_only=True)
            self.assertEqual(resumed["step"], 3)
            for section in ("model", "ema"):
                for name in full[section]:
                    torch.testing.assert_close(full[section][name], resumed[section][name], rtol=0, atol=0)
            torch.testing.assert_close(full["cpu_rng"], resumed["cpu_rng"], rtol=0, atol=0)
            for key, state in full["optimizer"]["state"].items():
                for name, value in state.items():
                    torch.testing.assert_close(value, resumed["optimizer"]["state"][key][name], rtol=0, atol=0)
            with Image.open(root / "split/previews/step_0000003.png") as preview:
                self.assertGreater(preview.width, 0)
            self.assertTrue((root / "split/best.pt").exists())
            # Check data identity before continuation, even when the path is unchanged.
            manifest_path = data / "manifest.json"
            manifest_path.write_text(manifest_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "dataset differs"):
                run(args)


if __name__ == "__main__":
    unittest.main()
