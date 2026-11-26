#!/usr/bin/env python3
"""RK3576 RTSP inference worker with a small monitoring/control API."""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import subprocess
import threading
import time
from collections import deque
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import cv2
import numpy as np
import psutil

from infer_rknn import draw_detections, infer_frame, load_labels, preprocess_frame
from runtime_utils import CORE_MODES, create_runtime


NPU_LOAD_PATH = Path("/sys/kernel/debug/rknpu/load")
NPU_FREQ_PATH = Path("/sys/kernel/debug/rknpu/freq")
NPU_DEVFREQ_PATH = Path("/sys/class/devfreq/27700000.npu")


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError):
        return ""


def parse_npu_load() -> list[float]:
    values = re.findall(r"Core\d+:\s*(\d+)%", read_text(NPU_LOAD_PATH))
    return [float(value) for value in values]


def read_npu_frequency() -> int | None:
    value = read_text(NPU_FREQ_PATH) or read_text(NPU_DEVFREQ_PATH / "cur_freq")
    try:
        return int(value)
    except ValueError:
        return None


def fit_stream_frame(frame: np.ndarray, max_side: int) -> np.ndarray:
    """Limit stream resolution while preserving aspect ratio and even dimensions."""
    height, width = frame.shape[:2]
    longest = max(height, width)
    if max_side <= 0 or longest <= max_side:
        return frame
    scale = max_side / longest
    resized_width = max(2, round(width * scale / 2) * 2)
    resized_height = max(2, round(height * scale / 2) * 2)
    return cv2.resize(
        frame, (resized_width, resized_height), interpolation=cv2.INTER_AREA
    )


class LatestFrameReader:
    def __init__(self, source_url: str) -> None:
        self.source_url = source_url
        self.frame = None
        self.frame_id = 0
        self.captured_at = 0.0
        self.source_fps = 0.0
        self.width = 0
        self.height = 0
        self.connected = False
        self.error = ""
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="rtsp-reader", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=3)

    def snapshot(self):
        with self._lock:
            if self.frame is None:
                return None, self.frame_id, self.captured_at
            return self.frame.copy(), self.frame_id, self.captured_at

    def _run(self) -> None:
        os.environ.setdefault(
            "OPENCV_FFMPEG_CAPTURE_OPTIONS",
            "rtsp_transport;tcp|fflags;nobuffer|max_delay;0",
        )
        while not self._stop.is_set():
            capture = cv2.VideoCapture(self.source_url, cv2.CAP_FFMPEG)
            if not capture.isOpened():
                self.error = f"Cannot open source: {self.source_url}"
                self.connected = False
                time.sleep(1)
                continue
            self.connected = True
            self.error = ""
            self.source_fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
            self.width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            while not self._stop.is_set():
                ok, frame = capture.read()
                if not ok:
                    self.connected = False
                    self.error = "Source disconnected; reconnecting"
                    break
                with self._lock:
                    self.frame = frame
                    self.frame_id += 1
                    self.captured_at = time.monotonic()
            capture.release()
            if not self._stop.is_set():
                time.sleep(0.5)


class RawVideoPublisher:
    def __init__(
        self,
        publish_url: str,
        width: int,
        height: int,
        fps: float,
        encoder: str,
    ) -> None:
        self.publish_url = publish_url
        self.width = width
        self.height = height
        self.fps = max(1.0, fps)
        self.encoder = encoder
        self.process: subprocess.Popen[bytes] | None = None
        self.error = ""
        self.stderr_tail = ""
        self._stderr_thread: threading.Thread | None = None
        self._start()

    def _start(self) -> None:
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s:v",
            f"{self.width}x{self.height}",
            "-r",
            f"{self.fps:.3f}",
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            self.encoder,
        ]
        if self.encoder == "libx264":
            command += ["-preset", "ultrafast", "-tune", "zerolatency"]
        command += [
            "-pix_fmt",
            "yuv420p",
            "-g",
            str(max(1, round(self.fps))),
            "-bf",
            "0",
            "-f",
            "rtsp",
            "-rtsp_transport",
            "tcp",
            self.publish_url,
        ]
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr, name="publisher-stderr", daemon=True
        )
        self._stderr_thread.start()

    def write(self, frame) -> bool:
        if self.process is None or self.process.poll() is not None:
            self.error = self.stderr_tail or "FFmpeg publisher stopped"
            return False
        try:
            assert self.process.stdin is not None
            self.process.stdin.write(frame.tobytes())
            return True
        except (BrokenPipeError, OSError) as exc:
            self.error = str(exc)
            return False

    def _drain_stderr(self) -> None:
        process = self.process
        if process is None or process.stderr is None:
            return
        for data in iter(lambda: process.stderr.read(4096), b""):
            if data:
                self.stderr_tail = (self.stderr_tail + data.decode(errors="replace"))[-2000:].strip()

    def stop(self) -> None:
        if self.process is None:
            return
        if self.process.stdin:
            self.process.stdin.close()
        try:
            self.process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self.process.terminate()
        if self._stderr_thread:
            self._stderr_thread.join(timeout=1)
        self._stderr_thread = None
        self.process = None


class MetricsSampler:
    def __init__(self, pipeline: "InferencePipeline") -> None:
        self.pipeline = pipeline
        self.process = psutil.Process()
        self.current: dict[str, Any] = {}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="metrics", daemon=True)
        psutil.cpu_percent(interval=None)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)

    def _run(self) -> None:
        while not self._stop.wait(1):
            memory = psutil.virtual_memory()
            cores = parse_npu_load()
            process_memory = self.process.memory_info().rss
            with self.pipeline.lock:
                live = dict(self.pipeline.live)
            self.current = {
                "timestamp": time.time(),
                "cpu_percent": psutil.cpu_percent(interval=None),
                "memory_percent": memory.percent,
                "memory_used_bytes": memory.used,
                "memory_total_bytes": memory.total,
                "process_rss_bytes": process_memory,
                "npu_cores_percent": cores,
                "npu_percent": max(cores) if cores else None,
                "npu_frequency_hz": read_npu_frequency(),
                "npu_memory_bytes": process_memory,
                "npu_memory_estimated": True,
                "npu_memory_note": "RKNN process RSS; this driver does not expose dedicated NPU memory",
                **live,
            }


class InferencePipeline:
    def __init__(self, model: Path, labels: Path | None = None) -> None:
        self.model_dir = model.parent.resolve()
        self.model = self.resolve_model(model.name)
        self.labels = load_labels(labels)
        self.lock = threading.Lock()
        self.live: dict[str, Any] = self._empty_live()
        self.reader: LatestFrameReader | None = None
        self.publisher: RawVideoPublisher | None = None
        self.runtime = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._frame_times: deque[float] = deque(maxlen=180)
        self.infer_gate = threading.Lock()

    def resolve_model(self, name: str) -> Path:
        model_name = Path(str(name)).name
        if model_name != str(name) or not model_name.lower().endswith(".rknn"):
            raise ValueError(f"Invalid model name: {name}")
        candidate = (self.model_dir / model_name).resolve()
        if candidate.parent != self.model_dir or not candidate.is_file():
            raise ValueError(f"Invalid model name: {name}")
        return candidate

    def available_models(self) -> list[dict[str, Any]]:
        return [
            {
                "name": model.name,
                "bytes": model.stat().st_size,
                "active": model.resolve() == self.model.resolve(),
            }
            for model in sorted(self.model_dir.glob("*.rknn"))
            if model.is_file()
        ]

    def _empty_live(self) -> dict[str, Any]:
        return {
            "running": False,
            "source_connected": False,
            "publisher_connected": False,
            "source_url": "",
            "publish_url": "",
            "fps": 0.0,
            "frames_processed": 0,
            "frames_dropped": 0,
            "detections": 0,
            "preprocess_ms": 0.0,
            "inference_ms": 0.0,
            "postprocess_ms": 0.0,
            "pipeline_ms": 0.0,
            "capture_to_publish_ms": 0.0,
            "last_error": "",
            "model": self.model.name,
        }

    def infer_image(
        self, image_bytes: bytes, cores: str = "both", imgsz: int = 640, conf: float = 0.35, iou: float = 0.6, model_name: str | None = None
    ) -> dict[str, Any]:
        encoded = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("无法解码上传图片，请使用 JPG、PNG 或 WEBP")
        frame = fit_stream_frame(frame, 1280)
        model = self.resolve_model(model_name or self.model.name)
        with self.infer_gate:
            runtime = create_runtime(model, cores)
            try:
                result = infer_frame(runtime, frame, imgsz, conf, iou)
            finally:
                runtime.release()
        rendered = draw_detections(
            frame, result.boxes, result.scores, result.classes, self.labels
        )
        ok, output = cv2.imencode(
            ".jpg", rendered, [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        )
        if not ok:
            raise RuntimeError("检测结果图片编码失败")
        return {
            "image_base64": base64.b64encode(output.tobytes()).decode("ascii"),
            "detections": int(len(result.boxes)),
            "preprocess_ms": result.preprocess_ms,
            "inference_ms": result.inference_ms,
            "postprocess_ms": result.postprocess_ms,
            "width": int(rendered.shape[1]),
            "height": int(rendered.shape[0]),
        }

    def start(self, config: dict[str, Any]) -> None:
        selected_model = self.resolve_model(str(config.get("model", self.model.name)))
        self.stop()
        self.model = selected_model
        source_url = str(config["source_url"])
        publish_url = str(config["publish_url"])
        cores = str(config.get("cores", "both"))
        if cores not in CORE_MODES:
            raise ValueError(f"Invalid core mode: {cores}")
        settings = {
            "source_url": source_url,
            "publish_url": publish_url,
            "cores": cores,
            "warmup": int(config.get("warmup", 20)),
            "imgsz": int(config.get("imgsz", 640)),
            "conf": float(config.get("conf", 0.35)),
            "iou": float(config.get("iou", 0.6)),
            "output_fps": float(config.get("output_fps", 25.0)),
            "output_max_side": int(config.get("output_max_side", 1280)),
            "encoder": str(config.get("encoder", "libx264")),
        }
        self._stop.clear()
        self._frame_times.clear()
        with self.lock:
            self.live = self._empty_live()
            self.live.update(
                running=True,
                source_url=source_url,
                publish_url=publish_url,
                model=selected_model.name,
            )
        self._thread = threading.Thread(
            target=self._run, args=(settings,), name="inference", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        reader = self.reader
        thread = self._thread
        if reader:
            reader.stop()
        if thread and thread.is_alive():
            thread.join(timeout=12)
        if thread and thread.is_alive():
            raise RuntimeError("Inference pipeline did not stop within 12 seconds")
        self._thread = None
        with self.lock:
            self.live["running"] = False

    def _run(self, settings: dict[str, Any]) -> None:
        runtime = None
        reader = None
        publisher = None
        try:
            runtime = create_runtime(self.model, settings["cores"])
            reader = LatestFrameReader(settings["source_url"])
            self.runtime = runtime
            self.reader = reader
            reader.start()
            deadline = time.monotonic() + 20
            frame = None
            captured_at = 0.0
            last_frame_id = 0
            while time.monotonic() < deadline and not self._stop.is_set():
                frame, last_frame_id, captured_at = reader.snapshot()
                if frame is not None:
                    break
                time.sleep(0.05)
            if frame is None:
                reader_error = reader.error or "unknown source error"
                raise RuntimeError(
                    f"No frame received from RTSP source within 20 seconds: {reader_error}"
                )

            frame = fit_stream_frame(frame, settings["output_max_side"])
            warmup_input, _ = preprocess_frame(frame, settings["imgsz"])
            with self.infer_gate:
                for _ in range(settings["warmup"]):
                    runtime.inference(inputs=[warmup_input], data_format=["nhwc"])

            height, width = frame.shape[:2]
            publisher = RawVideoPublisher(
                settings["publish_url"],
                width,
                height,
                settings["output_fps"],
                settings["encoder"],
            )
            self.publisher = publisher
            processed = 0
            while not self._stop.is_set():
                frame, frame_id, captured_at = reader.snapshot()
                if frame is None or frame_id == last_frame_id:
                    time.sleep(0.002)
                    continue
                dropped = max(0, frame_id - last_frame_id - 1)
                last_frame_id = frame_id
                frame = fit_stream_frame(frame, settings["output_max_side"])
                if frame.shape[:2] != (height, width):
                    raise RuntimeError(
                        "Source resolution changed while streaming; restart the pipeline"
                    )
                with self.infer_gate:
                    result = infer_frame(
                        runtime,
                        frame,
                        settings["imgsz"],
                        settings["conf"],
                        settings["iou"],
                    )
                rendered = draw_detections(
                    frame,
                    result.boxes,
                    result.scores,
                    result.classes,
                    self.labels,
                )
                published = publisher.write(rendered)
                now = time.monotonic()
                self._frame_times.append(now)
                while self._frame_times and now - self._frame_times[0] > 3:
                    self._frame_times.popleft()
                fps = (
                    (len(self._frame_times) - 1)
                    / max(0.001, self._frame_times[-1] - self._frame_times[0])
                    if len(self._frame_times) > 1
                    else 0.0
                )
                processed += 1
                pipeline_ms = (
                    result.preprocess_ms + result.inference_ms + result.postprocess_ms
                )
                with self.lock:
                    self.live.update(
                        source_connected=reader.connected,
                        publisher_connected=published,
                        fps=fps,
                        frames_processed=processed,
                        frames_dropped=self.live["frames_dropped"] + dropped,
                        detections=len(result.boxes),
                        preprocess_ms=result.preprocess_ms,
                        inference_ms=result.inference_ms,
                        postprocess_ms=result.postprocess_ms,
                        pipeline_ms=pipeline_ms,
                        capture_to_publish_ms=(now - captured_at) * 1000,
                        last_error=publisher.error or reader.error,
                    )
        except Exception as exc:
            with self.lock:
                self.live["last_error"] = str(exc)
        finally:
            if reader:
                reader.stop()
            if publisher:
                publisher.stop()
            if runtime:
                runtime.release()
            if self.reader is reader:
                self.reader = None
            if self.publisher is publisher:
                self.publisher = None
            if self.runtime is runtime:
                self.runtime = None
            with self.lock:
                self.live["running"] = False


class ApiHandler(BaseHTTPRequestHandler):
    pipeline: InferencePipeline
    sampler: MetricsSampler

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:  # noqa: N802
        self._json(200, {"ok": True})

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in ("/health", "/api/health"):
            self._json(200, {"ok": True, "service": "rk3576-edge-stream"})
        elif path in ("/metrics", "/api/metrics"):
            self._json(200, self.sampler.current or {"running": False})
        elif path == "/api/models":
            self._json(
                200,
                {
                    "models": self.pipeline.available_models(),
                    "active": self.pipeline.model.name,
                },
            )
        elif path == "/api/status":
            with self.pipeline.lock:
                live = dict(self.pipeline.live)
            self._json(200, live)
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        try:
            path = urlparse(self.path).path
            body = self.rfile.read(length)
            if path == "/api/infer/image":
                if length <= 0 or length > 24 * 1024 * 1024:
                    raise ValueError("图片请求大小必须在 1B 到 24MB 之间")
                payload = json.loads(body or b"{}")
                image = base64.b64decode(payload.get("image_base64", ""), validate=True)
                result = self.pipeline.infer_image(
                    image,
                    cores=str(payload.get("cores", "both")),
                    imgsz=int(payload.get("imgsz", 640)),
                    conf=float(payload.get("conf", 0.35)),
                    iou=float(payload.get("iou", 0.6)),
                    model_name=payload.get("model"),
                )
                self._json(200, result)
            elif path == "/api/pipeline/start":
                payload = json.loads(body or b"{}")
                self.pipeline.start(payload)
                self._json(202, {"ok": True, "message": "pipeline starting"})
            elif path == "/api/pipeline/stop":
                self.pipeline.stop()
                self._json(200, {"ok": True})
            else:
                self._json(404, {"error": "not found"})
        except Exception as exc:
            self._json(400, {"error": str(exc)})

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[edge-api] {self.address_string()} {fmt % args}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--model-dir", type=Path, default=Path(__file__).with_name("model"))
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8090)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = args.model_dir.resolve()
    default_model = (
        args.model.resolve()
        if args.model
        else model_dir / "landslide_yolov8n_int8_rk3576_split.rknn"
    )
    if not default_model.is_file():
        candidates = sorted(model_dir.glob("*.rknn"))
        if not candidates:
            raise SystemExit(f"No RKNN model found in: {model_dir}")
        default_model = candidates[0]
    pipeline = InferencePipeline(default_model, args.labels)
    sampler = MetricsSampler(pipeline)
    sampler.start()
    ApiHandler.pipeline = pipeline
    ApiHandler.sampler = sampler
    server = ThreadingHTTPServer((args.host, args.port), ApiHandler)
    print(f"RK3576 edge stream API: http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()
        sampler.stop()
        server.server_close()


if __name__ == "__main__":
    main()
