from __future__ import annotations

import base64
from dataclasses import asdict
from http.client import HTTPConnection
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest

import numpy as np
from PIL import Image, ImageFont
import torch

from play import CANVAS_SIZE, Playground, discover_runs, make_server, read_run
from src import Glyph2Hand, ModelConfig, sample_euler
from src.data import normalize_image, render_character


class PlaygroundTests(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.addCleanup(torch.set_num_threads, self.threads)
        self.root = Path(self.temp.name)
        self.run = self.root / "DiT-test"
        self.run.mkdir()
        self.config = ModelConfig(image_size=16, patch_size=4, hidden_size=32,
                                  num_heads=4, local_depth=1, content_depth=1,
                                  generator_depth=1, description_tokens=3,
                                  time_frequency_size=32)
        self.settings = {"model": asdict(self.config),
                         "training": {"sample_steps": 3, "seed": 123}}
        (self.run / "run.json").write_text(json.dumps({"settings": self.settings}), encoding="utf-8")
        self.model = Glyph2Hand(self.config).eval()
        # Exercise a nonzero velocity rather than the zero-initialized output.
        torch.nn.init.normal_(self.model.generator.final.projection.weight, std=0.02)
        self.best = {"model": self.model.state_dict(), "settings": self.settings, "step": 20}
        torch.save(self.best, self.run / "best.pt")
        self.play = Playground(self.root, device="cpu", margin=2)
        font = ImageFont.truetype(str(self.play.font_path), 256)
        self.refs = [render_character(c, font, CANVAS_SIZE, 16) for c in "天地玄黄"]
        self.payload = {"run": self.run.name, "checkpoint": "best.pt", "character": "永",
                        "references": [ref.reshape(-1).tolist() for ref in self.refs]}

    def test_run_discovery_uses_saved_config_and_available_checkpoint_names(self):
        torch.save(self.model.state_dict(), self.run / "last.pt")
        for name, metadata in (("twist", {"model_config": {}}),
                               ("broken", {"settings": {"model": {}, "training": []}})):
            directory = self.root / name
            directory.mkdir()
            (directory / "run.json").write_text(json.dumps(metadata))
            (directory / "best.pt").touch()
        missing = self.root / "missing-config"
        missing.mkdir()
        (missing / "last.pt").touch()
        self.assertEqual(read_run(self.run), (self.config, 3, 123))
        self.assertEqual(discover_runs(self.root), [{"name": self.run.name,
                         "checkpoints": ["best.pt", "last.pt"], "image_size": 16,
                         "steps": 3, "seed": 123}])
        (self.run / "best.pt").unlink()
        self.assertEqual(discover_runs(self.root)[0]["checkpoints"], ["last.pt"])

    def test_generation_matches_existing_sampler_and_accepts_checkpoint_exports(self):
        standard = render_character("永", ImageFont.truetype(str(self.play.font_path), 64), 16, 2)
        references = np.stack([normalize_image(Image.fromarray(ref), 16, 2) for ref in self.refs])
        content = torch.from_numpy(standard.copy()).float()[None, None] / 127.5 - 1
        style = torch.from_numpy(references).float()[None, :, None] / 127.5 - 1
        generator = torch.Generator().manual_seed(123)
        noise = torch.randn(content.shape, generator=generator)
        expected = sample_euler(self.model, content, style, steps=3, noise=noise)
        expected = ((expected[0, 0] + 1) * 127.5).round().byte().numpy()
        first = self.play.generate(self.payload)
        actual = np.asarray(Image.open(io.BytesIO(base64.b64decode(first["image"].split(",")[1]))))
        np.testing.assert_array_equal(actual, expected)
        cached = self.play.model
        self.assertEqual(self.play.generate(self.payload)["image"], first["image"])
        self.assertIs(self.play.model, cached)
        # Both a slim metadata export and a plain state_dict retain inference.
        for checkpoint in ({"model": self.model.state_dict(), "step": 25}, self.model.state_dict()):
            torch.save(checkpoint, self.run / "last.pt")
            result = self.play.generate({**self.payload, "checkpoint": "last.pt"})
            self.assertEqual(result["image"], first["image"])
        incompatible = {**self.settings, "model": {**asdict(self.config), "content_depth": 2}}
        torch.save({**self.best, "settings": incompatible}, self.run / "last.pt")
        with self.assertRaisesRegex(ValueError, "run.json"):
            self.play.generate({**self.payload, "checkpoint": "last.pt"})

    def test_http_info_generation_and_invalid_input(self):
        server = make_server(self.play, 0)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        self.addCleanup(worker.join, 5)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        client = HTTPConnection("127.0.0.1", server.server_port, timeout=10)
        self.addCleanup(client.close)
        client.request("GET", "/")
        response = client.getresponse()
        self.assertEqual(response.status, 200)
        self.assertIn(b"board-4", response.read())
        client.request("GET", "/api/info")
        response = client.getresponse()
        self.assertEqual(json.loads(response.read())["runs"][0]["name"], self.run.name)
        for changes, status in (({}, 200), ({"references": self.payload["references"][:3]}, 400),
                                ({"character": "ab"}, 400), ({"run": "../DiT-test"}, 400),
                                ({"references": [[255] * CANVAS_SIZE**2] * 4}, 400)):
            client.request("POST", "/api/generate", json.dumps({**self.payload, **changes}),
                           {"Content-Type": "application/json"})
            response = client.getresponse()
            body = json.loads(response.read())
            self.assertEqual(response.status, status, body)
            self.assertIn("image" if status == 200 else "error", body)
        client.request("POST", "/api/generate", "{}", {"Content-Type": "application/json",
                       "Origin": "https://example.com"})
        response = client.getresponse()
        self.assertEqual(response.status, 403)
        response.read()


if __name__ == "__main__":
    unittest.main()
