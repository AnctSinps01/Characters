from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader

from src import ModelConfig, TrainingConfig
from src.data import HandwritingEpisodes, PreloadedGlyphs, read_manifest
from src.train import is_unstarted_run, run, select_data


def fixture_data(root: Path):
    root.mkdir()
    standards = np.stack([np.full((16, 16), index * 18, dtype=np.uint8)
                          for index in range(12)])
    np.save(root / "standards.npy", standards)
    writers = []
    for writer_id, count, split, offset in (("small", 5, "train", 1),
                                            ("rich", 9, "train", 2),
                                            ("val-rich", 12, "val", 3)):
        np.save(root / f"{writer_id}.npy", standards[:count] + offset)
        writers.append({"id": writer_id, "file": f"{writer_id}.npy",
                        "labels": list(range(count)), "split": split})
    (root / "manifest.json").write_text(json.dumps({
        "version": 1, "image_size": 16, "characters": list("啊阿埃挨哎唉爱碍安岸暗昂"),
        "writers": writers,
    }), encoding="utf-8")


def tiny_settings():
    model = ModelConfig(image_size=16, patch_size=4, hidden_size=16, num_heads=2,
                        local_depth=1, content_depth=1, generator_depth=1,
                        description_tokens=2, time_frequency_size=16)
    training = TrainingConfig(steps=3, warmup_steps=1, batch_size=3,
                              min_references=1, max_references=2, workers=0,
                              val_batches=1, validate_every=3, save_every=1,
                              log_every=1, sample_steps=2, precision="fp32")
    return model, training


class SingleWriterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_selection_prefers_largest_eligible_training_writer_and_stable_ties(self):
        _, config = tiny_settings()
        manifest = {"writers": [
            {"id": "z", "labels": list(range(9)), "split": "train"},
            {"id": "val", "labels": list(range(12)), "split": "val"},
            {"id": "invalid", "labels": [0] * 20, "split": "train"},
            {"id": "a", "labels": list(range(9)), "split": "train"},
            {"id": "small", "labels": list(range(4)), "split": "train"},
        ]}
        def choose(writer_id=None, checkpoint=None, single=True):
            return select_data(manifest, config, single_writer=single,
                               writer_id=writer_id, checkpoint=checkpoint)
        chosen = choose()
        self.assertEqual(chosen["writer_id"], "a")
        self.assertEqual(chosen["character_count"], 9)
        manifest["writers"].reverse()
        self.assertEqual(choose(), chosen)
        explicit = choose("small")
        self.assertEqual(choose(checkpoint={"data_selection": explicit}), explicit)
        with self.assertRaisesRegex(ValueError, "writer-id differs"):
            choose("z", {"data_selection": explicit})
        with self.assertRaisesRegex(ValueError, "single-author checkpoint"):
            choose(checkpoint={})
        with self.assertRaisesRegex(ValueError, "train-single"):
            choose(checkpoint={"data_selection": explicit}, single=False)
        for writer_id in ("val", "invalid", "unknown"):
            with self.subTest(writer_id=writer_id), self.assertRaises(ValueError):
                choose(writer_id)
        self.assertIsNone(choose(checkpoint={}, single=False))

    def test_selected_preload_and_episodes_only_use_chosen_author_and_keep_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data"
            fixture_data(root)
            before = (root / "manifest.json").read_bytes()
            with patch("src.data.np.load", wraps=np.load) as load:
                store = PreloadedGlyphs(root, writer_ids=("rich",))
            self.assertEqual({Path(call.args[0]).name for call in load.call_args_list},
                             {"standards.npy", "rich.npy"})
            self.assertEqual(store.handwriting_count, 9)
            self.assertEqual(store.nbytes, (9 + 12) * 16 * 16)
            self.assertEqual(set(store.writer_ranges), {"rich"})
            datasets = [HandwritingEpisodes(root, split, length=12, min_references=1,
                                             max_references=2, seed=seed,
                                             preloaded=store, writer_id="rich")
                        for split, seed in (("train", 42), ("val", 43))]
            for dataset in datasets:
                for index in range(len(dataset)):
                    item = dataset[index]
                    self.assertEqual(item["author_index"].item(), 1)
                    target_pixel = item["target_handwriting"][0, 0, 0]
                    reference_pixels = item["reference_handwriting"][item["reference_mask"], 0, 0, 0]
                    self.assertFalse((reference_pixels == target_pixel).any())
                    self.assertEqual(reference_pixels.unique().numel(), reference_pixels.numel())
                    torch.testing.assert_close(item["target_handwriting"] - item["target_standard"],
                                               torch.full_like(item["target_standard"], 2 / 127.5))
            self.assertFalse(torch.equal(datasets[0][0]["reference_handwriting"],
                                         datasets[1][0]["reference_handwriting"]))
            for direct, worker in zip(DataLoader(datasets[0], batch_size=3, num_workers=0),
                                      DataLoader(datasets[0], batch_size=3, num_workers=1,
                                                 multiprocessing_context="spawn")):
                for key in direct:
                    torch.testing.assert_close(direct[key], worker[key], rtol=0, atol=0)
            for dataset in datasets:
                dataset.close()
            self.assertEqual((root / "manifest.json").read_bytes(), before)

    def test_startup_metadata_cannot_be_reused_for_other_data_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "previews").mkdir()
            selection = {"mode": "single_writer_fit", "writer_id": "rich"}
            settings = {"model": {}, "training": {}}
            (root / "run.json").write_text(json.dumps({
                "settings": settings, "data_fingerprint": "fingerprint",
                "data_selection": selection,
            }), encoding="utf-8")
            self.assertTrue(is_unstarted_run(root, settings, "fingerprint", data_selection=selection))
            self.assertFalse(is_unstarted_run(root, settings, "fingerprint"))
            self.assertFalse(is_unstarted_run(root, settings, "fingerprint", data_selection={
                "mode": "single_writer_fit", "writer_id": "small",
            }))

    def test_single_author_training_preview_and_resume_match_uninterrupted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            fixture_data(data)
            model, config = tiny_settings()
            args = argparse.Namespace(data=data, output=root / "full", resume=None,
                                      device="cpu", workers=0, stop_after=None, writer_id=None)
            with redirect_stdout(io.StringIO()):
                with patch("src.train.ModelConfig", return_value=model), \
                     patch("src.train.TrainingConfig", return_value=config):
                    run(args, single_writer=True)
                    args.output = root / "split"
                    args.stop_after = 1
                    run(args, single_writer=True)
                args.resume = args.output / "last.pt"
                args.stop_after = None
                saved = torch.load(args.resume, weights_only=True)
                before = args.resume.read_bytes()
                with self.assertRaisesRegex(ValueError, "train-single"):
                    run(args)
                args.writer_id = "small"
                with self.assertRaisesRegex(ValueError, "writer-id differs"):
                    run(args, single_writer=True)
                self.assertEqual(args.resume.read_bytes(), before)
                args.writer_id = None
                run(args, single_writer=True)
            full = torch.load(root / "full/last.pt", weights_only=True)
            resumed = torch.load(root / "split/last.pt", weights_only=True)
            self.assertEqual(resumed["data_selection"], saved["data_selection"])
            self.assertEqual(resumed["data_selection"]["writer_id"], "rich")
            self.assertEqual(resumed["data_selection"]["character_count"], 9)
            self.assertEqual(resumed["settings"]["training"]["author_style_loss_weight"], 0.05)
            self.assertEqual(resumed["step"], 3)
            for name in full["model"]:
                torch.testing.assert_close(full["model"][name], resumed["model"][name], rtol=0, atol=0)
            torch.testing.assert_close(full["cpu_rng"], resumed["cpu_rng"], rtol=0, atol=0)
            for key, state in full["optimizer"]["state"].items():
                for name, value in state.items():
                    torch.testing.assert_close(value, resumed["optimizer"]["state"][key][name],
                                               rtol=0, atol=0)
            metadata = json.loads((root / "split/run.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["data_selection"], resumed["data_selection"])
            self.assertTrue((root / "split/previews/step_0000003.png").is_file())
            self.assertTrue((root / "split/best.pt").is_file())
            self.assertEqual(resumed["data_fingerprint"], read_manifest(data)[1])

    def test_entry_point_and_documented_arguments_work_from_characters(self):
        implementation = Path(__file__).resolve().parents[1]
        characters = implementation.parent
        relative_entry = "DiT-flow/train-single/train.py"
        process = subprocess.run([sys.executable, relative_entry, "--help"],
                                 cwd=characters, capture_output=True, text=True, check=True)
        self.assertIn("--writer-id", process.stdout)
        self.assertIn("--resume", process.stdout)
        argv = [relative_entry, "--data", "data/prepared", "--output", "runs/DiT-flow/single-author",
                "--device", "cuda", "--workers", "0", "--stop-after", "10000"]
        with patch.object(sys, "argv", argv), patch("src.train.run") as shared_run:
            runpy.run_path(str(implementation / "train-single/train.py"), run_name="__main__")
        shared_run.assert_called_once()
        args = shared_run.call_args.args[0]
        self.assertTrue(shared_run.call_args.kwargs["single_writer"])
        self.assertEqual(args.data, Path("data/prepared"))
        self.assertEqual(args.stop_after, 10000)
        self.assertIsNone(args.writer_id)


if __name__ == "__main__":
    unittest.main()
