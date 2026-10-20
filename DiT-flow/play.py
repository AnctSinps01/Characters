"""Draw four reference glyphs and generate a character with a DiT-flow run.

From Characters: python DiT-flow/play.py
Models are discovered under runs/; architecture is read from each run.json.
"""
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fontTools.ttLib import TTFont
import numpy as np
from PIL import Image, ImageFont
import torch

from src import Glyph2Hand, ModelConfig, sample_euler
from src.data import normalize_image, render_character


CANVAS_SIZE = 256
REFERENCE_COUNT = 4
CHECKPOINT_NAMES = ("best.pt", "last.pt")
MAX_STEPS = 256


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "runs")
    parser.add_argument("--font", type=Path, default=ROOT / "fonts" / "WenJinMinchoP0-Regular.ttf")
    parser.add_argument("--font-index", type=int, default=0)
    parser.add_argument("--margin", type=int, default=4, help="match the margin used in prepare_data")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda or cuda:N")
    parser.add_argument("--precision", choices=("auto", "fp32", "bf16", "fp16"), default="auto")
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--no-open", action="store_true", help="do not automatically open a browser")
    return parser.parse_args(argv)


def png_url(pixels):
    stream = io.BytesIO()
    Image.fromarray(pixels).save(stream, format="PNG")
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode("ascii")


def read_run(run_dir: Path):
    metadata = json.loads((run_dir / "run.json").read_text(encoding="utf-8-sig"))
    if not isinstance(metadata, dict) or not isinstance(metadata.get("settings"), dict):
        raise ValueError("run.json 缺少 settings。")
    settings = metadata["settings"]
    if not isinstance(settings.get("model"), dict):
        raise ValueError("run.json 缺少 settings.model。")
    config = ModelConfig(**settings["model"])
    if config.in_channels != 1:
        raise ValueError("演示仅支持灰度字图模型。")
    training = settings.get("training", {})
    if not isinstance(training, dict):
        raise ValueError("run.json 中的 settings.training 格式不正确。")
    steps, seed = training.get("sample_steps", 32), training.get("seed", 42)
    if type(steps) is not int or not 1 <= steps <= MAX_STEPS:
        raise ValueError(f"sample_steps 应在 1–{MAX_STEPS} 之间。")
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed 应为 0–4294967295 之间的整数。")
    return config, steps, seed


def discover_runs(runs_dir: Path):
    """List DiT-flow runs with at least one checkpoint and usable run.json."""
    runs = []
    if not runs_dir.is_dir():
        return runs
    for directory in sorted(runs_dir.iterdir(), key=lambda path: path.name):
        if not directory.is_dir():
            continue
        checkpoints = [name for name in CHECKPOINT_NAMES if (directory / name).is_file()]
        if not checkpoints:
            continue
        try:
            config, steps, seed = read_run(directory)
        except (OSError, ValueError, TypeError, KeyError):
            # Other model families (e.g. Twist) may share the runs directory.
            continue
        runs.append({"name": directory.name, "checkpoints": checkpoints,
                     "image_size": config.image_size, "steps": steps, "seed": seed})
    return runs


def load_checkpoint(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        # Compatibility with the older PyTorch releases used by this project.
        return torch.load(path, map_location="cpu")


class Playground:
    def __init__(self, runs_dir=ROOT / "runs", *, font_path=ROOT / "fonts" / "WenJinMinchoP0-Regular.ttf",
                 font_index=0, margin=4, device="auto", precision="auto", cpu_threads=1):
        if type(cpu_threads) is not int or cpu_threads < 1:
            raise ValueError("cpu-threads must be positive")
        if type(margin) is not int or margin < 0:
            raise ValueError("margin must be a nonnegative integer")
        torch.set_num_threads(cpu_threads)
        self.device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
                       if device == "auto" else torch.device(device))
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("choose an available cpu or cuda device")
        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise ValueError("CUDA is unavailable; use --device cpu")
            self.device = torch.device("cuda", torch.cuda.current_device()
                                       if self.device.index is None else self.device.index)
            torch.cuda.set_device(self.device)
        supports_bf16 = self.device.type == "cuda" and torch.cuda.is_bf16_supported()
        self.precision = ("bf16" if supports_bf16 else "fp32") if precision == "auto" else precision
        if self.precision not in ("fp32", "bf16", "fp16"):
            raise ValueError("precision must be auto, fp32, bf16 or fp16")
        if self.precision == "bf16" and not supports_bf16:
            raise ValueError("bf16 is unavailable; use --precision fp32")
        if self.precision == "fp16" and self.device.type != "cuda":
            raise ValueError("fp16 requires CUDA; use --precision fp32 on CPU")
        self.runs_dir = Path(runs_dir).resolve()
        self.font_path = Path(font_path).resolve()
        self.font_index, self.margin = font_index, margin
        with TTFont(str(self.font_path), fontNumber=font_index) as font:
            self.font_characters = set(font.getBestCmap() or {})
        self.lock = threading.Lock()
        self.model = None
        self.model_key = None
        self.step = None

    def info(self):
        return {"runs": discover_runs(self.runs_dir), "device": str(self.device),
                "precision": self.precision, "canvas_size": CANVAS_SIZE,
                "reference_count": REFERENCE_COUNT}

    def _load_model(self, run_name, checkpoint_name, config):
        path = self.runs_dir / run_name / checkpoint_name
        stat = path.stat()
        key = (path, stat.st_mtime_ns, stat.st_size, config)
        if self.model_key == key:
            return self.model
        print(f"Loading {path} ...", flush=True)
        checkpoint = load_checkpoint(path)
        if not isinstance(checkpoint, dict):
            raise ValueError("checkpoint 必须是模型权重字典。")
        saved_settings = checkpoint.get("settings")
        if isinstance(saved_settings, dict) and "model" in saved_settings:
            if ModelConfig(**saved_settings["model"]) != config:
                raise ValueError("checkpoint 的模型配置与 run.json 不一致，请检查所选实验。")
        # Both training exports and plain state_dict files are supported.
        state = checkpoint.get("model", checkpoint)
        model = Glyph2Hand(config)
        model.load_state_dict(state, strict=True)
        step = checkpoint.get("step")
        del state, checkpoint
        # Keep only one run on the inference device at a time.
        self.model = None
        self.model_key = None
        self.model = model.to(self.device).eval()
        self.model_key, self.step = key, step
        return self.model

    @staticmethod
    def reference_image(values, index, size, margin):
        if not isinstance(values, list) or len(values) != CANVAS_SIZE**2:
            raise ValueError(f"参考字 {index} 的画板数据大小不正确。")
        pixels = np.asarray(values)
        if (pixels.ndim != 1 or pixels.dtype.kind not in "iu"
                or pixels.min() < 0 or pixels.max() > 255):
            raise ValueError(f"参考字 {index} 的像素格式不正确。")
        if pixels.min() == pixels.max():
            raise ValueError(f"请先写好参考字 {index}。")
        image = Image.fromarray(pixels.astype(np.uint8).reshape(CANVAS_SIZE, CANVAS_SIZE))
        return normalize_image(image, size, margin)

    @torch.inference_mode()
    def generate(self, payload):
        if not isinstance(payload, dict):
            raise ValueError("请求格式不正确。")
        run_name, checkpoint_name = payload.get("run"), payload.get("checkpoint")
        runs = discover_runs(self.runs_dir)
        selected = next((run for run in runs if run["name"] == run_name), None)
        if selected is None or checkpoint_name not in selected["checkpoints"]:
            raise ValueError("请选择有效的 DiT-flow 实验及 best.pt / last.pt。")
        config, default_steps, default_seed = read_run(self.runs_dir / run_name)
        if config.image_size <= 2 * self.margin:
            raise ValueError("margin 太大，无法在模型图像中保留字形。")
        steps, seed = payload.get("steps", default_steps), payload.get("seed", default_seed)
        if type(steps) is not int or not 1 <= steps <= MAX_STEPS:
            raise ValueError(f"采样步数应为 1–{MAX_STEPS} 之间的整数。")
        if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
            raise ValueError("随机种子应为 0–4294967295 之间的整数。")
        character = payload.get("character")
        if not isinstance(character, str):
            raise ValueError("请输入一个汉字。")
        character = character.strip()
        if len(character) != 1 or not 0x4E00 <= ord(character) <= 0x9FFF:
            raise ValueError("请输入一个基本汉字（与训练预处理字符范围一致）。")
        if ord(character) not in self.font_characters:
            raise ValueError(f"当前标准字体不包含“{character}”，请换字或使用 --font 指定字体。")
        references = payload.get("references")
        if not isinstance(references, list) or len(references) != REFERENCE_COUNT:
            raise ValueError("请写好四个参考字。")
        reference_pixels = np.stack([self.reference_image(values, index + 1,
                                   config.image_size, self.margin)
                                    for index, values in enumerate(references)])
        font = ImageFont.truetype(str(self.font_path), config.image_size * 4, index=self.font_index)
        standard_pixels = render_character(character, font, config.image_size, self.margin)
        with self.lock:
            started = time.perf_counter()
            model = self._load_model(run_name, checkpoint_name, config)
            standard = torch.from_numpy(standard_pixels.copy()).float()[None, None].to(self.device) / 127.5 - 1
            handwriting = torch.from_numpy(reference_pixels).float()[None, :, None].to(self.device) / 127.5 - 1
            generator = torch.Generator(device=self.device).manual_seed(seed)
            noise = torch.randn(standard.shape, generator=generator, device=self.device)
            dtype = torch.bfloat16 if self.precision == "bf16" else torch.float16
            context = (nullcontext() if self.precision == "fp32"
                       else torch.autocast(device_type=self.device.type, dtype=dtype))
            with context:
                result = sample_euler(model, standard, handwriting, steps=steps, noise=noise)
            if not bool(torch.isfinite(result).all()):
                raise FloatingPointError("模型输出存在非有限值，可尝试 --precision fp32。")
            pixels = ((result[0, 0].float() + 1) * 127.5).round().byte().cpu().numpy()
            return {"image": png_url(pixels), "standard": png_url(standard_pixels),
                    "run": run_name, "checkpoint": checkpoint_name, "step": self.step,
                    "character": character, "seed": seed, "steps": steps,
                    "image_size": config.image_size,
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
            allowed = (f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}")
            host, origin = self.headers.get("Host", ""), self.headers.get("Origin")
            return host in allowed and (origin is None or origin in tuple(f"http://{value}" for value in allowed))

        def do_GET(self):
            if not self.local_request():
                self.json(403, {"error": "请通过本地演示页面访问。"})
            elif self.path == "/":
                self.send(200, page_path.read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/api/info":
                self.json(200, playground.info())
            elif self.path == "/favicon.ico":
                self.send(204, b"", "image/x-icon")
            else:
                self.json(404, {"error": "页面不存在。"})

        def do_POST(self):
            if not self.local_request():
                self.json(403, {"error": "请通过本地演示页面访问。"})
                return
            if self.path != "/api/generate":
                self.json(404, {"error": "页面不存在。"})
                return
            try:
                if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                    raise ValueError("请求格式应为 JSON。")
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 2_000_000:
                    raise ValueError("四张画板的数据大小不正确。")
                payload = json.loads(self.rfile.read(length))
                self.json(200, playground.generate(payload))
            except (ValueError, TypeError) as error:
                self.json(400, {"error": str(error)})
            except Exception as error:
                print(f"Generation failed: {error}", file=sys.stderr, flush=True)
                self.json(500, {"error": f"生成失败：{error}"})

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main():
    args = parse_args()
    if not 1 <= args.port <= 65535:
        raise ValueError("port must be in [1,65535]")
    playground = Playground(args.runs_dir, font_path=args.font, font_index=args.font_index,
                            margin=args.margin, device=args.device, precision=args.precision,
                            cpu_threads=args.cpu_threads)
    server = make_server(playground, args.port)
    url = f"http://127.0.0.1:{server.server_port}"
    runs = playground.info()["runs"]
    print(f"DiT-flow playground: {url} | {playground.device} | {playground.precision}", flush=True)
    print(f"Found {len(runs)} DiT-flow run(s) in {playground.runs_dir}; select one in the browser.", flush=True)
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
