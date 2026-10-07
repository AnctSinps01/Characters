"""Small prepared-data fixtures exercise loading, training, previews and resume."""
from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import signal
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

from twist import LossConfig, ModelConfig, RegistrationResult
from twist.geometry import identity_grid, lipschitz_bound, sample_image
from twist.preview import save_preview
from glyph_pair import load_glyphs
from settings import TrainingConfig
from train import parse_args, train


def prepared_fixture(root):
    root.mkdir()
    (root / "writers").mkdir()
    standard = np.full((16, 16), 255, dtype=np.uint8)
    standard[3:13, 5:7] = 0
    standard[7:9, 3:13] = 0
    np.save(root / "standards.npy", np.stack((standard, np.rot90(standard))))
    writers = []
    for index in range(5):
        hand = np.roll(standard, index % 3 - 1, axis=1)
        # 每个 writer 两个字；交错标签顺序检查按标签取字而非固定行号。
        labels = [0, 1] if index % 2 == 0 else [1, 0]
        images = np.stack((hand, np.rot90(standard))) if labels[0] == 0 else np.stack((np.rot90(standard), hand))
        relative = f"writers/{index:05d}.npy"
        np.save(root / relative, images)
        writers.append({"id": f"writer{index}", "file": relative, "labels": labels,
                        "split": "train" if index < 3 else "val"})
    (root / "manifest.json").write_text(json.dumps({"version": 1, "image_size": 16,
                 "characters": ["中", "十"], "writers": writers}), encoding="utf-8")
    return standard


def tiny_model():
    return ModelConfig(image_size=16, patch_size=4, token_dim=16, num_heads=4,
                       blocks_per_stage=1, num_stages=2)


class TrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_all_selected_writers_preloaded_with_correct_character_and_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "prepared"
            standard = prepared_fixture(root)
            with redirect_stdout(io.StringIO()):
                glyphs = load_glyphs(root, tiny_model(), LossConfig(), TrainingConfig())
            self.assertEqual(glyphs["targets"].shape, (5, 1, 16, 16))
            self.assertEqual(glyphs["train_indices"].tolist(), [0, 1, 2])
            self.assertEqual(glyphs["val_indices"].tolist(), [3, 4])
            expected = torch.from_numpy(np.roll(standard, 0, axis=1).copy()).float() / 127.5 - 1
            torch.testing.assert_close(glyphs["targets"][1, 0], expected)
            # 后续抽样只用缓存，文件已经移走也不受影响。
            (root / "writers").rename(root / "moved_writers")
            self.assertTrue(torch.isfinite(glyphs["targets"][glyphs["train_indices"]]).all())

    def test_faint_glyphs_use_minmax_and_constant_handwriting_is_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "prepared"
            prepared_fixture(root)
            for path in [root / "standards.npy", *sorted((root / "writers").glob("*.npy"))]:
                pixels = np.load(path)
                np.save(path, np.where(pixels == 0, 220, 255).astype(np.uint8))
            path = root / "writers" / "00001.npy"
            pixels = np.load(path)
            pixels[1] = 255  # writer1 的「中」，标签顺序为 [十, 中]。
            np.save(path, pixels)
            with redirect_stdout(io.StringIO()):
                glyphs = load_glyphs(root, tiny_model(), LossConfig(), TrainingConfig())
            self.assertEqual(glyphs["metadata"]["handwriting_count"], 4)
            self.assertEqual(glyphs["metadata"]["excluded_images"][0]["writer"], "writer1")
            self.assertEqual(glyphs["metadata"]["train_ids"], ["writer0", "writer2"])
            self.assertEqual(glyphs["metadata"]["val_ids"], ["writer3", "writer4"])
            for images in (glyphs["source"], glyphs["targets"]):
                self.assertTrue(bool((images.amin(dim=(1, 2, 3)) == -1).all()))
                self.assertTrue(bool((images.amax(dim=(1, 2, 3)) == 1).all()))
            self.assertTrue(bool(torch.isfinite(glyphs["distance"]).all()))
            with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "constant image"):
                load_glyphs(root, tiny_model(), LossConfig(), TrainingConfig(), "writer1")

    def test_synthetic_target_is_nontrivial_and_has_fixed_boundary_and_known_grid(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "prepared"
            prepared_fixture(root)
            settings = TrainingConfig(mode="synthetic")
            with redirect_stdout(io.StringIO()):
                glyphs = load_glyphs(root, tiny_model(), LossConfig(), settings)
            self.assertEqual(glyphs["metadata"]["handwriting_count"], 5)
            self.assertEqual(glyphs["train_indices"].tolist(), [0])
            self.assertFalse(len(glyphs["val_indices"]))
            field = glyphs["true_field"]
            self.assertLessEqual(float(lipschitz_bound(field)[0]), settings.synthetic_lipschitz)
            self.assertEqual(int(torch.count_nonzero(field[:, :, 0])), 0)
            self.assertGreater(float((glyphs["targets"] - glyphs["source"]).abs().max()), 0.05)
            torch.testing.assert_close(glyphs["targets"], sample_image(glyphs["source"], glyphs["true_grid"]))

    def test_writer_overfit_still_loads_all_and_does_not_validate_training_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "prepared"
            prepared_fixture(root)
            with redirect_stdout(io.StringIO()):
                glyphs = load_glyphs(root, tiny_model(), LossConfig(), TrainingConfig(), "writer2")
            self.assertEqual(glyphs["metadata"]["handwriting_count"], 5)
            self.assertEqual(glyphs["train_indices"].tolist(), [2])
            self.assertEqual(glyphs["val_indices"].tolist(), [])
            with self.assertRaisesRegex(ValueError, "not prepared"):
                load_glyphs(root, tiny_model(), LossConfig(), replace(TrainingConfig(), character="无"))

    def run_training(self, arguments):
        with patch.object(sys, "argv", ["train.py", *map(str, arguments)]), redirect_stdout(io.StringIO()):
            args = parse_args()
            train(args)

    def test_training_outputs_and_resume_match_uninterrupted_updates_without_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared_fixture(root / "prepared")
            common = ["--data", root / "prepared", "--device", "cpu", "--image-size", 16,
                      "--patch-size", 4, "--token-dim", 16, "--num-heads", 4,
                      "--blocks-per-stage", 1, "--num-stages", 2, "--batch-size", 2,
                      "--preview-every", 2, "--log-every", 1, "--animation-frames", 3]
            self.run_training([*common, "--output", root / "whole", "--steps", 5])
            self.run_training([*common, "--output", root / "split", "--steps", 3])
            (root / "prepared").rename(root / "moved_prepared")
            self.run_training(["--resume", root / "split" / "last.pt", "--steps", 5, "--device", "cpu"])
            whole = torch.load(root / "whole" / "last.pt", weights_only=True)
            resumed = torch.load(root / "split" / "last.pt", weights_only=True)
            self.assertEqual(resumed["step"], 5)
            for name, value in whole["model"].items():
                torch.testing.assert_close(value, resumed["model"][name], rtol=0, atol=0)
            summary = json.loads((root / "split" / "summary.json").read_text())
            self.assertEqual(summary["final"]["train"]["count"], 3)
            self.assertEqual(summary["final"]["val"]["count"], 2)
            self.assertEqual(summary["status"], "completed")
            self.assertTrue((root / "split" / "best.pt").is_file())
            with Image.open(root / "split" / "previews" / "step_0000005_val_trajectory.gif") as animation:
                self.assertEqual(animation.n_frames, 3)
            self.assertFalse(list((root / "split").rglob("*.png")))
            for metric in (summary["final"]["train"], summary["final"]["val"]):
                self.assertTrue(all(bound <= 0.8 for bound in metric["stage_lipschitz_max"]))
            with self.assertRaisesRegex(ValueError, "nonempty"):
                self.run_training(["--output", root / "split", "--device", "cpu"])

    def test_stop_request_saves_a_completed_optimizer_step(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared_fixture(root / "prepared")
            original_step = torch.optim.Adam.step
            original_handler = signal.getsignal(signal.SIGINT)
            def interrupted_step(optimizer, *args, **kwargs):
                # 请求在更新进行中到达；处理器应允许本次更新完成。
                signal.raise_signal(signal.SIGINT)
                return original_step(optimizer, *args, **kwargs)
            with patch.object(torch.optim.Adam, "step", interrupted_step):
                self.run_training(["--data", root / "prepared", "--output", root / "run",
                    "--device", "cpu", "--steps", 5, "--image-size", 16, "--patch-size", 4,
                    "--token-dim", 16, "--num-heads", 4, "--blocks-per-stage", 1,
                    "--num-stages", 1, "--batch-size", 2, "--animation-frames", 3])
            summary = json.loads((root / "run" / "summary.json").read_text())
            saved = torch.load(root / "run" / "last.pt", weights_only=True)
            self.assertEqual(summary["status"], "interrupted")
            self.assertEqual(saved["step"], 1)
            self.assertTrue(all(float(state["step"]) == 1 for state in saved["optimizer"]["state"].values()))
            self.assertEqual(signal.getsignal(signal.SIGINT), original_handler)

    def test_gif_frame_count_follows_setting_even_when_the_image_does_not_move(self):
        with tempfile.TemporaryDirectory() as directory:
            source = torch.ones(1, 1, 16, 16)
            source[:, :, 3:13, 7:9] = -1
            grid = identity_grid(1, (16, 16), device=source.device)
            result = RegistrationResult(source, (torch.zeros(1, 2, 6, 6),), (source,), (grid,))
            for frames in (7, TrainingConfig().animation_frames, 121):
                output = Path(directory) / str(frames)
                save_preview(result, output, 0, frames)
                with Image.open(output / "step_0000000_train_trajectory.gif") as animation:
                    self.assertEqual(animation.n_frames, frames)
                self.assertEqual({path.suffix for path in output.iterdir()}, {".gif"})


if __name__ == "__main__":
    unittest.main()
