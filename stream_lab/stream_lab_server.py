#!/usr/bin/env python3
"""Local RTSP simulation server and web console backend."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import shutil
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parent
WEB_ROOT = ROOT / "web"
UPLOAD_ROOT = ROOT / "uploads"
CONFIG_PATH = ROOT / "stream_lab_config.json"


def json_request(
    url: str,
    method: str = "GET",
    payload: dict | None = None,
    timeout: float = 4,
):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = Request(
        url,
        method=method,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


class ProcessController:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.media_process: subprocess.Popen | None = None
        self.source_process: subprocess.Popen | None = None
        self.current_video = ""
        self.last_error = ""
        self.stream_generation = 0
        self.active_payload: dict[str, Any] | None = None
        self.last_start_at = 0.0
        self.lock = threading.Lock()
        self._watchdog_stop = threading.Event()
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop, name="stream-watchdog", daemon=True
        )
        self._watchdog_thread.start()

    @staticmethod
    def _alive(process: subprocess.Popen | None) -> bool:
        return process is not None and process.poll() is None

    def start_media_server(self) -> None:
        if self._alive(self.media_process):
            return
        local_exe = ROOT / "bin" / "mediamtx.exe"
        if local_exe.is_file():
            command = [str(local_exe), str(ROOT / "mediamtx.yml")]
        elif shutil.which("docker"):
            subprocess.run(
                ["docker", "rm", "-f", "landslide-mediamtx"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            command = [
                "docker",
                "run",
                "--rm",
                "--name",
                "landslide-mediamtx",
                "-p",
                "8554:8554",
                "-p",
                "8888:8888",
                "-p",
                "9997:9997",
                "-v",
                f"{ROOT / 'mediamtx.yml'}:/mediamtx.yml:ro",
                "bluenviron/mediamtx:1.20.0",
                "/mediamtx.yml",
            ]
        else:
            raise RuntimeError(
                "MediaMTX is unavailable. Put mediamtx.exe in stream_lab/bin or install Docker."
            )
        self.media_process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        time.sleep(1)
        if self.media_process.poll() is not None:
            raise RuntimeError("MediaMTX failed to start")

    def stop_source(self) -> None:
        if self._alive(self.source_process):
            self.source_process.terminate()
            try:
                self.source_process.wait(timeout=4)
            except subprocess.TimeoutExpired:
                self.source_process.kill()
        self.source_process = None

    def stop_media_server(self) -> None:
        if self._alive(self.media_process):
            self.media_process.terminate()
            try:
                self.media_process.wait(timeout=4)
            except subprocess.TimeoutExpired:
                self.media_process.kill()
        self.media_process = None

    def start_source(self, video_name: str) -> None:
        video_path = (UPLOAD_ROOT / video_name).resolve()
        if UPLOAD_ROOT.resolve() not in video_path.parents or not video_path.is_file():
            raise ValueError("Uploaded video does not exist")
        self.start_media_server()
        self.stop_source()
        command = [
            self.args.ffmpeg,
            "-hide_banner",
            "-loglevel",
            "warning",
            "-re",
            "-stream_loop",
            "-1",
            "-i",
            str(video_path),
            "-an",
            "-vf",
            "scale=w='min(1280,iw)':h='min(1280,ih)':force_original_aspect_ratio=decrease:force_divisible_by=2,fps=25",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-tune",
            "zerolatency",
            "-pix_fmt",
            "yuv420p",
            "-g",
            "25",
            "-keyint_min",
            "25",
            "-sc_threshold",
            "0",
            "-bf",
            "0",
            "-f",
            "rtsp",
            "-rtsp_transport",
            "tcp",
            f"rtsp://{self.args.local_ip}:8554/input",
        ]
        self.source_process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        self.current_video = video_name
        time.sleep(1)
        if self.source_process.poll() is not None:
            raise RuntimeError("FFmpeg source publisher failed; check the RTSP server and video codec")

    def start_pipeline(self, payload: dict[str, Any]) -> dict[str, Any]:
        encoder = str(payload.get("encoder", "libx264"))
        if encoder != "libx264":
            raise ValueError(
                f"编码器 {encoder} 在当前 RK3576 镜像不可用；请使用 libx264。"
            )
        config = {
            "source_url": f"rtsp://{self.args.local_ip}:8554/input",
            "publish_url": f"rtsp://{self.args.local_ip}:8554/detected",
            "cores": payload.get("cores", "both"),
            "warmup": int(payload.get("warmup", 20)),
            "imgsz": int(payload.get("imgsz", 640)),
            "conf": float(payload.get("conf", 0.35)),
            "iou": float(payload.get("iou", 0.6)),
            "output_fps": float(payload.get("output_fps", 25)),
            "output_max_side": int(payload.get("output_max_side", 1280)),
            "encoder": encoder,
        }
        model = str(payload.get("model", "")).strip()
        if model:
            config["model"] = model
        return json_request(
            f"http://{self.args.board_ip}:8090/api/pipeline/start",
            "POST",
            config,
            timeout=25,
        )

    def start_all(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            self.active_payload = None
            self._stop_edge()
            self.stop_source()
            # A fresh media server guarantees new SDP, GOP and HLS state when
            # switching between videos with different resolutions/codecs.
            self.stop_media_server()
            time.sleep(0.3)
            self.start_source(str(payload["video"]))
            try:
                edge = self.start_pipeline(payload)
            except Exception:
                self.stop_source()
                raise
            self.stream_generation += 1
            self.active_payload = dict(payload)
            self.last_start_at = time.monotonic()
            return edge

    def _watchdog_loop(self) -> None:
        while not self._watchdog_stop.wait(2):
            with self.lock:
                payload = dict(self.active_payload or {})
                source_dead = self.active_payload is not None and not self._alive(self.source_process)
                can_probe_edge = self.active_payload is not None and time.monotonic() - self.last_start_at > 20
            edge_dead = False
            if can_probe_edge:
                try:
                    edge = json_request(
                        f"http://{self.args.board_ip}:8090/api/status"
                    )
                    edge_dead = not bool(edge.get("running")) and bool(edge.get("last_error"))
                except Exception:
                    edge_dead = False
            if source_dead:
                try:
                    with self.lock:
                        if self.active_payload is not None:
                            self.start_source(str(payload["video"]))
                            self.last_start_at = time.monotonic()
                            self.last_error = ""
                except Exception as exc:
                    self.last_error = f"自动重启源推流失败: {exc}"
                continue
            if edge_dead:
                try:
                    self.start_pipeline(payload)
                    self.last_start_at = time.monotonic()
                    self.last_error = ""
                except Exception as exc:
                    self.last_error = f"自动重启板端管线失败: {exc}"

    def shutdown(self) -> None:
        self._watchdog_stop.set()
        self._watchdog_thread.join(timeout=3)
        self.stop_all()
        self.stop_media_server()

    def _stop_edge(self) -> None:
        try:
            json_request(
                f"http://{self.args.board_ip}:8090/api/pipeline/stop",
                "POST",
                {},
                timeout=20,
            )
        except (HTTPError, URLError, TimeoutError):
            pass

    def stop_all(self) -> None:
        with self.lock:
            self.active_payload = None
            self.last_start_at = 0.0
            self._stop_edge()
            self.stop_source()

    def status(self) -> dict[str, Any]:
        edge_metrics: dict[str, Any]
        try:
            edge_metrics = json_request(
                f"http://{self.args.board_ip}:8090/api/metrics"
            )
            edge_online = True
        except Exception as exc:
            edge_online = False
            edge_metrics = {"running": False, "last_error": str(exc)}
        return {
            "media_server": self._alive(self.media_process),
            "source_stream": self._alive(self.source_process),
            "current_video": self.current_video,
            "auto_restart_enabled": self.active_payload is not None,
            "watchdog_alive": self._watchdog_thread.is_alive(),
            "controller_error": self.last_error,
            "edge_online": edge_online,
            "edge": edge_metrics,
            "urls": {
                "source_rtsp": f"rtsp://{self.args.local_ip}:8554/input",
                "detected_rtsp": f"rtsp://{self.args.local_ip}:8554/detected",
                "detected_hls": (
                    f"http://{self.args.local_ip}:8888/detected/index.m3u8"
                    f"?v={self.stream_generation}"
                ),
                "edge_api": f"http://{self.args.board_ip}:8090",
            },
        }


class Handler(BaseHTTPRequestHandler):
    controller: ProcessController

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        return json.loads(self.rfile.read(length) or b"{}")

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/api/status":
            self._json(200, self.controller.status())
            return
        if path == "/api/videos":
            videos = [
                {"name": item.name, "bytes": item.stat().st_size}
                for item in sorted(UPLOAD_ROOT.iterdir())
                if item.is_file()
            ]
            self._json(200, {"videos": videos})
            return
        if path == "/api/models":
            models = json_request(
                f"http://{self.controller.args.board_ip}:8090/api/models"
            )
            self._json(200, models)
            return
        if path == "/api/image/status":
            self._json(200, {"ok": True, "max_bytes": 16 * 1024 * 1024})
            return
        relative = "index.html" if path == "/" else unquote(path.lstrip("/"))
        target = (WEB_ROOT / relative).resolve()
        if WEB_ROOT.resolve() not in target.parents and target != WEB_ROOT.resolve():
            self.send_error(403)
            return
        if not target.is_file():
            self.send_error(404)
            return
        body = target.read_bytes()
        self.send_response(200)
        self.send_header(
            "Content-Type", mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        )
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        try:
            if path == "/api/upload":
                filename = Path(unquote(self.headers.get("X-Filename", "video.mp4"))).name
                if not filename.lower().endswith((".mp4", ".mov", ".mkv", ".avi", ".webm")):
                    raise ValueError("Unsupported video extension")
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0:
                    raise ValueError("Empty upload")
                UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
                target = UPLOAD_ROOT / filename
                remaining = length
                with target.open("wb") as output:
                    while remaining:
                        chunk = self.rfile.read(min(1024 * 1024, remaining))
                        if not chunk:
                            raise RuntimeError("Upload ended early")
                        output.write(chunk)
                        remaining -= len(chunk)
                self._json(201, {"ok": True, "name": filename, "bytes": length})
            elif path == "/api/image":
                filename = Path(unquote(self.headers.get("X-Filename", "image.jpg"))).name
                if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                    raise ValueError("Unsupported image extension")
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 16 * 1024 * 1024:
                    raise ValueError("图片大小必须在 1B 到 16MB 之间")
                image_bytes = self.rfile.read(length)
                if len(image_bytes) != length:
                    raise RuntimeError("图片上传未完成")
                result = json_request(
                    f"http://{self.controller.args.board_ip}:8090/api/infer/image",
                    "POST",
                    {
                        "image_base64": base64.b64encode(image_bytes).decode("ascii"),
                        "cores": "both",
                        "imgsz": 640,
                        "conf": float(unquote(self.headers.get("X-Conf", "0.35"))),
                        "iou": float(unquote(self.headers.get("X-Iou", "0.6"))),
                        "model": unquote(self.headers.get("X-Model", "")),
                    },
                    timeout=60,
                )
                self._json(200, {"ok": True, **result})
            elif path == "/api/start":
                payload = self._read_json()
                edge = self.controller.start_all(payload)
                self._json(202, {"ok": True, "edge": edge})
            elif path == "/api/stop":
                self.controller.stop_all()
                self._json(200, {"ok": True})
            else:
                self._json(404, {"error": "not found"})
        except Exception as exc:
            self.controller.last_error = str(exc)
            self._json(400, {"error": str(exc)})

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[stream-lab] {self.address_string()} {fmt % args}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--local-ip", default="10.8.30.29")
    parser.add_argument("--board-ip", default="10.8.30.54")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument(
        "--auto-start",
        action="store_true",
        help="启动服务后自动循环推送上传目录中最新的视频",
    )
    parser.add_argument("--auto-start-video", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    controller = ProcessController(args)
    Handler.controller = controller
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Stream lab: http://{args.local_ip}:{args.port}")
    if args.auto_start:
        candidates = [
            item for item in UPLOAD_ROOT.iterdir() if item.is_file()
            and item.suffix.lower() in (".mp4", ".mov", ".mkv", ".avi", ".webm")
        ]
        selected = args.auto_start_video or (
            max(candidates, key=lambda item: item.stat().st_mtime).name if candidates else ""
        )
        if selected:
            try:
                controller.start_all({"video": selected, "conf": 0.35, "output_fps": 25, "cores": "both", "encoder": "libx264"})
            except Exception as exc:
                controller.last_error = f"自动启动推流失败: {exc}"
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        controller.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
