from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from dataclasses import asdict
import io
import json
import math
import multiprocessing
import os
from pathlib import Path
import random
import struct
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

from src.prepare_data import prepare
from src import Glyph2Hand, ModelConfig, TrainingConfig
from src.data import HandwritingEpisodes, PreloadedGlyphs, iter_gnt, normalize_image, read_manifest
from src.train import (evaluate, is_unstarted_run, learning_rate, make_grad_scaler,
                       metrics_json, optimizer_update, resolve_device, run)


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


def change_shared_pixel(store):
    # A child writes through the backing buffer only to prove storage is shared.
    assert not store.writer_images("0").flags.writeable
    assert not store.standards.flags.writeable
    np.frombuffer(store._buffer, dtype=np.uint8)[0] = 213


class DataTests(unittest.TestCase):
    def test_preload_shared_storage_and_sampling_without_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data"
            fixture_data(root)
            store = PreloadedGlyphs(root)
            self.assertEqual(store.nbytes, (3 * 6 + 6) * 16 * 16)
            self.assertEqual(store.standards.dtype, np.uint8)
            self.assertFalse(store.standards.flags.writeable)
            for writer in range(3):
                np.testing.assert_array_equal(store.writer_images(str(writer)), store.standards + writer)
            datasets = [HandwritingEpisodes(root, split, length=6, min_references=1,
                                             max_references=3, seed=5, preloaded=store)
                        for split in ("train", "val")]
            self.assertIs(datasets[0]._preloaded, datasets[1]._preloaded)
            # Spawn must transfer the shared allocation, not a private pixel copy.
            child = multiprocessing.get_context("spawn").Process(target=change_shared_pixel, args=(store,))
            child.start()
            child.join(timeout=30)
            if child.is_alive():
                child.terminate()
                child.join()
                self.fail("shared-memory child timed out")
            self.assertEqual(child.exitcode, 0)
            self.assertEqual(store.writer_images("0")[0, 0, 0], 213)
            for path in root.glob("*.npy"):
                path.unlink()
            # Neither parent nor newly spawned loader workers may reopen NPYs.
            for dataset in datasets:
                expected = list(DataLoader(dataset, batch_size=3, num_workers=0))
                actual = list(DataLoader(dataset, batch_size=3, num_workers=2,
                                         multiprocessing_context="spawn"))
                self.assertEqual(len(actual), len(expected))
                for direct, worker in zip(expected, actual):
                    for key in direct:
                        torch.testing.assert_close(direct[key], worker[key], rtol=0, atol=0)
                dataset.close()

    def test_vectorized_episodes_preserve_sampling_and_pixels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data"
            fixture_data(root)
            manifest_path = root / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for writer in manifest["writers"]:
                writer["labels"] = [4, 0, 5, 2, 1, 3]
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            dataset = HandwritingEpisodes(root, "train", length=24, min_references=1,
                                          max_references=5, seed=5)
            standards = np.load(root / "standards.npy")
            for index in range(len(dataset)):
                rng = random.Random(f"5:{index}")
                writer = dataset.writers[rng.randrange(len(dataset.writers))]
                count = rng.randint(1, 5)
                selected = rng.sample(range(len(writer["labels"])), count + 1)
                images = np.load(root / writer["file"])
                def convert(array):
                    return torch.from_numpy(array.copy()).float().unsqueeze(0) / 127.5 - 1
                expected = {
                    "target_standard": convert(standards[writer["labels"][selected[0]]]),
                    "target_handwriting": convert(images[selected[0]]),
                    "reference_handwriting": torch.ones(5, 1, 16, 16),
                    "reference_standard": torch.ones(5, 1, 16, 16),
                    "reference_mask": torch.arange(5) < count,
                }
                for slot, sample in enumerate(selected[1:]):
                    expected["reference_handwriting"][slot] = convert(images[sample])
                    expected["reference_standard"][slot] = convert(standards[writer["labels"][sample]])
                actual = dataset[index]
                self.assertEqual(set(actual), set(expected))
                for key in expected:
                    torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
            dataset.close()

    def test_reference_range_includes_one_and_configured_maximum(self):
        self.assertEqual(TrainingConfig().min_references, 4)
        self.assertEqual(TrainingConfig().max_references, 4)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data"
            fixture_data(root)
            for split in ("train", "val"):
                for maximum in (1, 5):
                    config = TrainingConfig(min_references=1, max_references=maximum)
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
                references = item["reference_handwriting"][valid, 0, 0, 0]
                target = item["target_handwriting"][0, 0, 0]
                self.assertFalse((references == target).any())
                self.assertEqual(len(references.unique()), len(references))
                self.assertTrue((item["reference_handwriting"][~valid] == 1).all())
                self.assertTrue((item["reference_standard"][~valid] == 1).all())
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

    def test_cuda_aliases_resolve_to_explicit_device(self):
        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.current_device", return_value=1) as current, \
             patch("torch.cuda.set_device") as select:
            for name, expected in (("cuda", "cuda:1"), ("auto", "cuda:1"), ("cuda:2", "cuda:2")):
                with self.subTest(device=name):
                    current.reset_mock()
                    self.assertEqual(resolve_device(name), torch.device(expected))
                    select.assert_called_with(torch.device(expected))
                    if name == "cuda:2":
                        current.assert_not_called()
            select.reset_mock()
            self.assertEqual(resolve_device("cpu"), torch.device("cpu"))
            select.assert_not_called()
        with patch("torch.cuda.is_available", return_value=False), \
             patch("torch.cuda.set_device") as select:
            self.assertEqual(resolve_device("auto"), torch.device("cpu"))
            with self.assertRaisesRegex(ValueError, "CUDA unavailable"):
                resolve_device("cuda")
            select.assert_not_called()

    def test_cuda_selection_failure_leaves_no_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            args = argparse.Namespace(device="cuda", output=output)
            with patch("torch.cuda.is_available", return_value=True), \
                 patch("torch.cuda.current_device", return_value=0), \
                 patch("torch.cuda.set_device", side_effect=RuntimeError("device initialization failed")), \
                 patch("torch.cuda.is_bf16_supported") as bf16:
                with self.assertRaisesRegex(RuntimeError, "device initialization failed"):
                    run(args)
                bf16.assert_not_called()
            self.assertFalse(output.exists())

    def test_gpu_preload_requires_cuda_and_is_runtime_only(self):
        with self.assertRaises(ValueError):
            TrainingConfig(preload_to_gpu=1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            fixture_data(data)
            output = root / "run"
            config = TrainingConfig(steps=2, warmup_steps=0, batch_size=1,
                                    min_references=1, max_references=2,
                                    preload_to_gpu=True, workers=2)
            model = ModelConfig(image_size=16, patch_size=4)
            args = argparse.Namespace(data=data, output=output, resume=None,
                                      device="cpu", workers=None, stop_after=1)
            with patch("src.train.TrainingConfig", return_value=config), \
                 patch("src.train.ModelConfig", return_value=model):
                with self.assertRaisesRegex(ValueError, "requires CUDA"):
                    run(args)
            self.assertFalse(output.exists())

    def test_retry_does_not_overwrite_existing_results(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "previews").mkdir()
            (output / "run.json").write_text(json.dumps({
                "settings": {"seed": 1}, "data_fingerprint": "dataset",
            }), encoding="utf-8")
            self.assertTrue(is_unstarted_run(output, {"seed": 1}, "dataset"))
            self.assertFalse(is_unstarted_run(output, {"seed": 2}, "dataset"))
            self.assertFalse(is_unstarted_run(output, {"seed": 1}, "other-dataset"))
            for name in ("last.pt", "best.pt", "metrics.jsonl", "previews/image.png", "notes.txt"):
                artifact = output / name
                artifact.write_bytes(b"existing result")
                self.assertFalse(is_unstarted_run(output, {"seed": 1}, "dataset"))
                self.assertEqual(artifact.read_bytes(), b"existing result")
                artifact.unlink()

    def test_schedule_boundaries(self):
        config = TrainingConfig(steps=10, warmup_steps=2)
        self.assertAlmostEqual(learning_rate(1, config), config.learning_rate / 2)
        self.assertAlmostEqual(learning_rate(2, config), config.learning_rate)
        self.assertAlmostEqual(learning_rate(10, config), config.learning_rate * config.min_lr_ratio)

    def test_old_architecture_checkpoints_are_rejected_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "old.pt"
            output = root / "new_run"
            args = argparse.Namespace(device="cpu", resume=checkpoint, output=output)
            for version in (1, 2, 3, 4, 5):
                with self.subTest(version=version):
                    torch.save({"version": version}, checkpoint)
                    with self.assertRaisesRegex(ValueError, "old checkpoint architecture"):
                        run(args)
                    self.assertFalse(output.exists())

    def test_metrics_json_has_bounded_precision_and_units(self):
        line = metrics_json({
            "step": 7220,
            "train_loss": 0.6108397245407104,
            "lr": 0.00029737878008084513,
            "grad_norm": 1.5258227801325568e16,
            "elapsed": "2287.18407843 s",
            "peak_allocated": "3.69401121 GB",
        })
        self.assertEqual(
            line,
            '{"step": 7220, "train_loss": 0.61083972, "lr": 0.00029738, '
            '"grad_norm": 1.52582278e+16, "elapsed": "2287.18407843 s", '
            '"peak_allocated": "3.69401121 GB"}',
        )
        parsed = json.loads(line)
        self.assertIsInstance(parsed["train_loss"], float)
        self.assertEqual(parsed["peak_allocated"], "3.69401121 GB")

        with self.assertRaisesRegex(ValueError, "must be finite"):
            metrics_json({"train_loss": float("nan")})

    def test_scaler_overflow_retries_without_advancing_optimizer_or_noise(self):
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scaler = make_grad_scaler(torch.device("cpu"), enabled=True,
                                  init_scale=2.0**120)
        config = TrainingConfig()
        noises = []

        def large_loss(model, batch):
            noise = torch.rand(1, 1)
            noises.append(noise.clone())
            return model(noise).sum() * 1e5

        with patch("src.train.batch_loss", side_effect=large_loss):
            loss, norm = optimizer_update(model, optimizer, scaler, {},
                                          torch.device("cpu"), "fp32", config)
        self.assertGreater(len(noises), 1)
        for noise in noises[1:]:
            torch.testing.assert_close(noises[0], noise, rtol=0, atol=0)
        self.assertTrue(np.isfinite(loss) and np.isfinite(norm))
        self.assertTrue(all(state["step"].item() == 1 for state in optimizer.state.values()))

    def test_validation_restores_current_model_mode_and_training_rng(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            fixture_data(data)
            model = Glyph2Hand(ModelConfig(
                image_size=16, patch_size=4, hidden_size=16, num_heads=2,
                local_depth=1, content_depth=1, generator_depth=1,
                description_tokens=2, time_frequency_size=16,
            ))
            config = TrainingConfig(batch_size=2, val_batches=1, sample_steps=2)
            dataset = HandwritingEpisodes(data, "val", length=2,
                                           min_references=1, max_references=2, seed=43)
            loader = DataLoader(dataset, batch_size=2,
                                generator=torch.Generator().manual_seed(43))
            for training_mode in (True, False):
                model.train(training_mode)
                before = torch.get_rng_state().clone()
                value = evaluate(model, loader, torch.device("cpu"), "fp32",
                                 config, root / "preview.png")
                self.assertTrue(math.isfinite(value))
                self.assertEqual(model.training, training_mode)
                torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)
            model.train()
            before = torch.get_rng_state().clone()
            with patch("src.train.batch_loss", return_value=torch.tensor(float("inf"))):
                with self.assertRaises(FloatingPointError):
                    evaluate(model, loader, torch.device("cpu"), "fp32",
                             config, root / "failed.png")
            self.assertTrue(model.training)
            torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)
            dataset.close()

    def test_train_preview_and_resume_equal_uninterrupted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            fixture_data(data)
            model = ModelConfig(image_size=16, patch_size=4, hidden_size=16, num_heads=2,
                                mlp_ratio=2, local_depth=1, content_depth=1, generator_depth=1,
                                description_tokens=2, time_frequency_size=16)
            config = TrainingConfig(steps=3, warmup_steps=1, batch_size=2,
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
                    # Reproduce the failed cloud startup: metadata and an empty
                    # preview directory already exist, but no training happened.
                    (args.output / "previews").mkdir(parents=True)
                    (args.output / "run.json").write_text(json.dumps({
                        "settings": {"model": asdict(model), "training": asdict(config)},
                        "data_fingerprint": read_manifest(data)[1], "device": "cuda",
                    }), encoding="utf-8")
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
            self.assertEqual(resumed["version"], 6)
            self.assertNotIn("ema", resumed)
            self.assertNotIn("preload_to_gpu", resumed["settings"]["training"])
            for name in full["model"]:
                torch.testing.assert_close(full["model"][name], resumed["model"][name], rtol=0, atol=0)
            torch.testing.assert_close(full["cpu_rng"], resumed["cpu_rng"], rtol=0, atol=0)
            for key, state in full["optimizer"]["state"].items():
                self.assertEqual(state["step"].item(), 3)
                for name, value in state.items():
                    torch.testing.assert_close(value, resumed["optimizer"]["state"][key][name], rtol=0, atol=0)
            with Image.open(root / "split/previews/step_0000003.png") as preview:
                self.assertGreater(preview.width, 0)
            self.assertTrue((root / "split/best.pt").exists())
            reports = [json.loads(line) for line in
                       (root / "split/gradient_diagnostics.jsonl").read_text().splitlines()]
            self.assertEqual([report["step"] for report in reports], [1, 2, 3])
            self.assertTrue(all(report["event"] == "gradients" for report in reports))
            self.assertTrue(all(report["top_parameters"] for report in reports))
            metrics = [json.loads(line) for line in
                       (root / "split/metrics.jsonl").read_text().splitlines()]
            self.assertTrue(all(0 < item["grad_clip_scale"] <= 1 for item in metrics))
            self.assertTrue(any("val_loss" in item for item in metrics))
            self.assertTrue(all("val_loss_ema" not in item for item in metrics))
            # Check data identity before continuation, even when the path is unchanged.
            manifest_path = data / "manifest.json"
            manifest_path.write_text(manifest_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "dataset differs"):
                run(args)


if __name__ == "__main__":
    unittest.main()
