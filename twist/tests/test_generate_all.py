from contextlib import redirect_stdout
from dataclasses import asdict
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from twist import ModelConfig, Twist
from twist.generate_all import CharacterImages, file_digest, generate, parse_args, save_character
from twist.images import NORMALIZATION


def fixture(root):
    root.mkdir()
    (root / "writers").mkdir()
    writers = []
    for number, (split, labels) in enumerate((("train", [0, 1, 2, 3]), ("val", [0, 1, 2]),
                                               ("train", [1]), ("val", [1]))):
        pixels = np.full((len(labels), 16, 16), 245, dtype=np.uint8)
        for row, label in enumerate(labels):
            if (number, label) in ((0, 1), (1, 2)):
                continue
            pixels[row, 3:13, 4 + number:7 + number] = 180
            pixels[row, 7:9, 3:13] = 180
        file = f"writers/{number:05d}.npy"
        np.save(root / file, pixels)
        writers.append({"id": f"writer_{number}", "file": file, "split": split, "labels": labels})
    (root / "manifest.json").write_text(json.dumps({"version": 1, "image_size": 16,
        "characters": list("中十独少"), "writers": writers}), encoding="utf-8")


def checkpoint(path):
    torch.manual_seed(91)
    config = ModelConfig(image_size=16, patch_size=4, token_dim=16, num_heads=4,
                         num_blocks=1, squaring_steps=3)
    torch.save({"kind": "twist-all-pair", "version": 3, "step": 123, "normalization": NORMALIZATION,
                "model_config": asdict(config), "model": Twist(config).state_dict()}, path)


class GenerateAllTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def arguments(self, root, *extra):
        return parse_args(["--checkpoint", str(root / "best.pt"), "--data", str(root / "data"),
                           "--output", str(root / "output"), "--device", "cpu", "--precision", "fp32",
                           "--batch-size", "2", "--frames", "3", "--display-size", "128", *extra])

    def run_export(self, root, *extra):
        with redirect_stdout(io.StringIO()):
            return generate(self.arguments(root, *extra))

    def test_all_characters_include_cross_split_pairs_and_exclude_constant_images(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory) / "data"
            fixture(root)
            data = CharacterImages(root, 16)
            records, summary = data.select_pairs("all", 42)
            self.assertEqual([record["character"] for record in records], list("中十"))
            self.assertEqual([record["valid_images"] for record in records], [2, 3])
            self.assertEqual({records[0][side]["split"] for side in ("source", "target")}, {"train", "val"})
            self.assertTrue(all(record["source"]["writer"] != record["target"]["writer"] for record in records))
            self.assertEqual(summary["scanned_images"], 9)
            self.assertEqual(summary["constant_images"], 2)
            self.assertEqual([row["valid_images"] for row in summary["skipped_characters"]], [1, 1])
            self.assertEqual(records, data.select_pairs("all", 42)[0])
            self.assertEqual(data.select_pairs("train", 42)[0], [])
            self.assertEqual([record["character"] for record in data.select_pairs("val", 42)[0]], ["十"])
            self.assertFalse(data.maps)

    def test_selected_faint_images_resize_and_normalize_with_checkpoint_resolution(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory) / "data"
            fixture(root)
            data = CharacterImages(root, 8)
            try:
                records, _ = data.select_pairs("all", 42)
                source, target = data.batch(records, "cpu")
                self.assertEqual(source.shape, (2, 1, 8, 8))
                for images in (source, target):
                    torch.testing.assert_close(images.amin(dim=(1, 2, 3)), -torch.ones(2))
                    torch.testing.assert_close(images.amax(dim=(1, 2, 3)), torch.ones(2))
                    self.assertTrue(bool(torch.isfinite(images).all()))
            finally:
                data.close()

    def test_checkpoint_to_batched_gifs_gallery_and_exact_endpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root / "data")
            checkpoint(root / "best.pt")
            forward = Twist.forward
            calls = []
            def counted(model, source, target):
                calls.append(len(source))
                return forward(model, source, target)
            with patch.object(Twist, "forward", counted), \
                    patch("twist.distance.distance_fields", side_effect=AssertionError("inference must not compute EDT")):
                output = self.run_export(root)
            self.assertEqual(calls, [2])
            plan = json.loads((output / "generation.json").read_text(encoding="utf-8"))
            self.assertEqual(plan["settings"]["model_config"]["image_size"], 16)
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["status"], "completed")
            self.assertEqual(summary["generated_characters"], 2)
            page = (output / "index.html").read_text(encoding="utf-8")
            self.assertTrue(all(character in page for character in "中十"))
            self.assertIn("查看双向未混合配准", page)
            self.assertIn('id="search"', page)
            for record in plan["pairs"]:
                with Image.open(output / record["gif"]) as gif:
                    self.assertEqual(gif.n_frames, 3)
                    self.assertEqual(gif.size, (384, 152))
                    first = np.asarray(gif.convert("RGB"))
                    np.testing.assert_array_equal(first[24:, :128], first[24:, 128:256])
                    gif.seek(2)
                    last = np.asarray(gif.convert("RGB"))
                    np.testing.assert_array_equal(last[24:, 128:256], last[24:, 256:])
                with Image.open(output / record["registration"]) as image:
                    self.assertEqual(image.size, (384, 304))
            with patch.object(Twist, "forward", side_effect=AssertionError("completed characters must be skipped")):
                self.run_export(root, "--resume")

    def test_interrupted_export_resumes_without_overwriting_completed_character(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root / "data")
            checkpoint(root / "best.pt")
            def interrupted(*args, **kwargs):
                save_character(*args, **kwargs)
                raise KeyboardInterrupt
            with patch("twist.generate_all.save_character", interrupted):
                output = self.run_export(root)
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["status"], "interrupted")
            self.assertEqual(summary["generated_characters"], 1)
            first = next((output / "transitions").glob("*.gif"))
            digest = file_digest(first)
            self.run_export(root, "--resume", "--batch-size", "1")
            self.assertEqual(file_digest(first), digest)
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["status"], "completed")
            self.assertEqual(summary["generated_characters"], 2)

    def test_existing_output_and_changed_resume_settings_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root / "data")
            checkpoint(root / "best.pt")
            self.run_export(root)
            with self.assertRaisesRegex(ValueError, "nonempty"):
                self.run_export(root)
            with self.assertRaisesRegex(ValueError, "settings/checkpoint/data differ"):
                self.run_export(root, "--resume", "--seed", "99")
            state = torch.load(root / "best.pt", weights_only=True)
            state["model"]["network.velocity_head.bias"] += 0.1
            torch.save(state, root / "best.pt")
            with self.assertRaisesRegex(ValueError, "settings/checkpoint/data differ"):
                self.run_export(root, "--resume")

    def test_legacy_checkpoint_cannot_be_mistaken_for_pixel_velocity_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            torch.save({"kind": "twist-all-pair", "version": 2}, root / "best.pt")
            with self.assertRaisesRegex(ValueError, "pixel-resolution"):
                self.run_export(root)


if __name__ == "__main__":
    unittest.main()
