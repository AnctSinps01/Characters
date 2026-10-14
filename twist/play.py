"""Local two-canvas playground for a trained 64x64 Twist model."""
from __future__ import annotations

import argparse
import base64
from contextlib import nullcontext
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import sys
import threading
import time
import webbrowser

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image
import torch

from twist import ModelConfig, Twist
from twist.images import NORMALIZATION, normalize_minmax


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "runs" / "twist-svf-audited" / "best.pt")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda or cuda:N")
    parser.add_argument("--precision", choices=("auto", "fp32", "bf16"), default="auto")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--no-open", action="store_true", help="Do not automatically open a browser")
    return parser.parse_args(argv)


def png_url(pixels):
    stream = io.BytesIO()
    Image.fromarray(pixels).save(stream, format="PNG")
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode("ascii")


class Playground:
    def __init__(self, checkpoint_path, device="auto", precision="auto", cpu_threads=1):
        if type(cpu_threads) is not int or cpu_threads < 1:
            raise ValueError("cpu-threads must be positive")
        torch.set_num_threads(cpu_threads)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)
        if self.device.type not in ("cpu", "cuda") or (self.device.type == "cuda" and not torch.cuda.is_available()):
            raise ValueError("choose an available cpu or cuda device")
        if self.device.type == "cuda":
            self.device = torch.device("cuda", torch.cuda.current_device() if self.device.index is None else self.device.index)
            torch.cuda.set_device(self.device)
        if precision not in ("auto", "fp32", "bf16"):
            raise ValueError("precision must be auto, fp32 or bf16")
        supports_bf16 = self.device.type == "cuda" and torch.cuda.is_bf16_supported()
        self.precision = ("bf16" if supports_bf16 else "fp32") if precision == "auto" else precision
        if self.precision == "bf16" and not supports_bf16:
            raise ValueError("bf16 is unavailable; use --precision fp32")
        checkpoint_path = Path(checkpoint_path).resolve()
        if not checkpoint_path.is_file():
            raise ValueError(f"Checkpoint not found: {checkpoint_path}; pass --checkpoint /path/to/best.pt")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint.get("kind") != "twist-all-pair" or checkpoint.get("version") != 3:
            raise ValueError("expected pixel-resolution twist-all-pair checkpoint version 3")
        if checkpoint.get("normalization") != NORMALIZATION:
            raise ValueError("checkpoint uses an unsupported normalization")
        config = ModelConfig(**checkpoint["model_config"])
        if config.image_size != 64 or config.in_channels != 1:
            raise ValueError("this drawing playground requires a 64x64 grayscale checkpoint")
        self.model = Twist(config).to(self.device).eval()
        self.model.load_state_dict(checkpoint["model"], strict=True)
        self.step = checkpoint.get("step", 0)
        self.checkpoint_name = checkpoint_path.name
        self.lock = threading.Lock()

    @staticmethod
    def image(values, name):
        if not isinstance(values, list) or len(values) != 64 * 64:
            raise ValueError(f"{name}画板应为 64×64 像素。")
        pixels = np.asarray(values)
        if (pixels.dtype.kind not in "iu" or pixels.ndim != 1
                or pixels.min() < 0 or pixels.max() > 255):
            raise ValueError(f"{name}画板的像素格式不正确。")
        if pixels.min() == pixels.max():
            raise ValueError(f"请先在{name}画板上画几笔。")
        return normalize_minmax(torch.from_numpy(pixels.astype(np.uint8).reshape(1, 1, 64, 64)))

    @torch.inference_mode()
    def generate(self, payload):
        if not isinstance(payload, dict):
            raise ValueError("请求格式不正确。")
        count, fps = payload.get("frames", 21), payload.get("fps", 12)
        if type(count) is not int or not 3 <= count <= 61:
            raise ValueError("帧数应在 3–61 之间。")
        if type(fps) is not int or not 4 <= fps <= 30:
            raise ValueError("播放速度应在 4–30 之间。")
        source = self.image(payload.get("source"), "左侧")
        target = self.image(payload.get("target"), "右侧")
        with self.lock:
            started = time.perf_counter()
            source, target = source.to(self.device), target.to(self.device)
            context = torch.autocast("cuda", dtype=torch.bfloat16) if self.precision == "bf16" else nullcontext()
            with context:
                result = self.model(source, target)
            times = torch.linspace(0, 1, count).tolist()
            frames = torch.stack([result.render(t).image[0, 0] for t in times])
            if not bool(torch.isfinite(frames).all()):
                raise FloatingPointError("模型生成了无效画面，请更换画法后再试。")
            pixels = ((frames.float().clamp(-1, 1) + 1) * 127.5).round().byte().cpu().numpy()
            # Play A->B->A continuously; each direction uses the same rendered frames.
            sequence = list(range(count)) + list(range(count - 2, 0, -1))
            duration = max(10, round(1000 / fps / 10) * 10)
            durations = [600 if index in (0, count - 1) else duration for index in sequence]
            images = [Image.fromarray(pixels[index]) for index in sequence]
            gif = io.BytesIO()
            images[0].save(gif, format="GIF", save_all=True, append_images=images[1:],
                           duration=durations, loop=0, optimize=False, disposal=2)
            return {"gif": "data:image/gif;base64," + base64.b64encode(gif.getvalue()).decode("ascii"),
                    "frames": [png_url(frame) for frame in pixels], "times": times,
                    "sequence": sequence, "durations": durations,
                    "seconds": round(time.perf_counter() - started, 3)}


def make_server(playground, port):
    page_path = Path(__file__).with_name("play.html")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            if str(args[1]) not in ("200", "204"):
                super().log_message(format, *args)

        def send(self, status, body, content_type="application/json; charset=utf-8"):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def json(self, status, data):
            self.send(status, json.dumps(data, ensure_ascii=False, allow_nan=False).encode("utf-8"))

        def local_request(self):
            # The browser and model communicate only on this loopback origin.
            # Reject cross-origin requests instead of exposing a local compute API.
            allowed = (f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}")
            host = self.headers.get("Host", "")
            origin = self.headers.get("Origin")
            return host in allowed and (origin is None or origin in tuple(f"http://{value}" for value in allowed))

        def do_GET(self):
            if not self.local_request():
                self.json(403, {"error": "请通过本地画板页面访问。"})
            elif self.path == "/":
                self.send(200, page_path.read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/api/info":
                self.json(200, {"checkpoint": playground.checkpoint_name, "step": playground.step,
                                "size": 64, "device": str(playground.device)})
            elif self.path == "/favicon.ico":
                self.send(204, b"", "image/x-icon")
            else:
                self.json(404, {"error": "页面不存在。"})

        def do_POST(self):
            if not self.local_request():
                self.json(403, {"error": "请通过本地画板页面访问。"})
                return
            if self.path != "/api/generate":
                self.json(404, {"error": "页面不存在。"})
                return
            try:
                if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                    raise ValueError("请求格式应为 JSON。")
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 100_000:
                    raise ValueError("画板数据大小不正确。")
                payload = json.loads(self.rfile.read(length))
                self.json(200, playground.generate(payload))
            except (ValueError, TypeError) as error:
                self.json(400, {"error": str(error)})
            except Exception as error:
                print(f"Generation failed: {error}", file=sys.stderr, flush=True)
                self.json(500, {"error": "生成失败，请查看启动窗口中的提示。"})

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main():
    args = parse_args()
    if not 1 <= args.port <= 65535:
        raise ValueError("port must be in [1,65535]")
    print(f"Loading {args.checkpoint} ...", flush=True)
    playground = Playground(args.checkpoint, args.device, args.precision, args.cpu_threads)
    server = make_server(playground, args.port)
    url = f"http://127.0.0.1:{server.server_port}"
    print(f"Twist playground: {url} | {playground.device} | step={playground.step}", flush=True)
    print("Keep this window open; Ctrl+C stops the playground.", flush=True)
    if not args.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Playground stopped.", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
