import base64
from dataclasses import asdict
from http.client import HTTPConnection
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from twist import ModelConfig, Twist
from twist.images import NORMALIZATION
from twist.play import Playground, make_server


def drawing(offset=0):
    pixels = np.full((64, 64), 250, dtype=np.uint8)
    pixels[12:52, 28+offset:32+offset] = 50
    pixels[12:52, 27+offset] = 180
    pixels[29:33, 14:50] = 50
    return pixels.ravel().tolist()


def png_pixels(url):
    with Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))) as image:
        return np.asarray(image.convert("L"))


class PlaygroundTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.path = Path(cls.directory.name) / "best.pt"
        config = ModelConfig(token_dim=16, num_heads=4, num_blocks=1, squaring_steps=3)
        torch.save({"kind": "twist-all-pair", "version": 3, "normalization": NORMALIZATION,
                    "model_config": asdict(config), "step": 123, "model": Twist(config).state_dict()}, cls.path)
        cls.playground = Playground(cls.path, "cpu", "fp32")

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_model_runs_once_and_outputs_exact_normalized_endpoints_and_downloadable_gif(self):
        payload = {"source": drawing(), "target": drawing(6), "frames": 5, "fps": 12}
        with patch.object(self.playground.model, "forward", wraps=self.playground.model.forward) as predict:
            result = self.playground.generate(payload)
        self.assertEqual(predict.call_count, 1)
        self.assertEqual(len(result["frames"]), 5)
        self.assertEqual(result["sequence"], [0, 1, 2, 3, 4, 3, 2, 1])
        self.assertEqual(len(result["durations"]), 8)
        self.assertEqual(result["durations"][0], 600)
        self.assertEqual(result["durations"][4], 600)
        for values, frame in ((payload["source"], result["frames"][0]), (payload["target"], result["frames"][-1])):
            pixels = np.array(values).reshape(64, 64)
            expected = np.round((pixels - pixels.min()) / (pixels.max() - pixels.min()) * 255).astype(np.uint8)
            np.testing.assert_array_equal(png_pixels(frame), expected)
        with Image.open(io.BytesIO(base64.b64decode(result["gif"].split(",", 1)[1]))) as gif:
            self.assertEqual(gif.format, "GIF")
            self.assertEqual(gif.size, (64, 64))
            self.assertEqual(gif.info["loop"], 0)
            np.testing.assert_array_equal(np.asarray(gif.convert("L")), png_pixels(result["frames"][0]))
        self.assertTrue(all(parameter.grad is None for parameter in self.playground.model.parameters()))

    def test_empty_and_invalid_canvas_data_are_rejected_before_model_execution(self):
        with patch.object(self.playground.model, "forward", side_effect=AssertionError("invalid input reached model")):
            for values in (None, [], [255] * 4096, [0] * 4096, [256] * 4096,
                           [[255]] * 4096, ["white"] * 4096, [1.2] * 4096):
                with self.assertRaises(ValueError):
                    self.playground.generate({"source": values, "target": drawing()})
            for field, value in (("frames", 100), ("frames", True), ("fps", 0), ("fps", 31)):
                with self.assertRaises(ValueError):
                    self.playground.generate({"source": drawing(), "target": drawing(4), field: value})

    def test_http_page_info_inference_errors_and_origin_check(self):
        server = make_server(self.playground, 0)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        connection = HTTPConnection("127.0.0.1", server.server_port, timeout=10)
        try:
            connection.request("GET", "/")
            response = connection.getresponse()
            self.assertEqual(response.status, 200)
            page = response.read().decode("utf-8")
            self.assertIn('id="left" width="64"', page)
            self.assertIn('id="right" width="64"', page)
            self.assertIn("笔刷柔边", page)
            connection.request("GET", "/api/info")
            response = connection.getresponse()
            self.assertEqual(json.loads(response.read())["step"], 123)
            payload = {"source": drawing(), "target": drawing(4), "frames": 3}
            connection.request("POST", "/api/generate", json.dumps(payload), {"Content-Type": "application/json"})
            response = connection.getresponse()
            self.assertEqual(response.status, 200)
            self.assertTrue(json.loads(response.read())["gif"].startswith("data:image/gif;base64,"))
            connection.request("POST", "/api/generate", "{", {"Content-Type": "application/json"})
            response = connection.getresponse()
            self.assertEqual(response.status, 400)
            self.assertIn("error", json.loads(response.read()))
            connection.request("POST", "/api/generate", json.dumps(payload),
                               {"Content-Type": "application/json", "Origin": "https://example.com"})
            response = connection.getresponse()
            self.assertEqual(response.status, 403)
            response.read()
        finally:
            connection.close()
            server.shutdown()
            server.server_close()
            worker.join(timeout=2)

    def test_unsupported_checkpoints_are_rejected(self):
        path = Path(self.directory.name) / "old.pt"
        torch.save({"kind": "twist-all-pair", "version": 2}, path)
        with self.assertRaisesRegex(ValueError, "version 3"):
            Playground(path, "cpu")
        with self.assertRaisesRegex(ValueError, "not found"):
            Playground(path.with_name("missing.pt"), "cpu")


if __name__ == "__main__":
    unittest.main()
