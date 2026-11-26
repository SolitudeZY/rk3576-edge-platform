from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from stream_lab import stream_lab_server


class ProcessControllerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.args = argparse.Namespace(
            local_ip="10.8.30.29",
            board_ip="10.8.30.54",
            ffmpeg="ffmpeg",
        )
        self.controller = stream_lab_server.ProcessController(self.args)

    def test_start_all_stops_source_when_edge_start_fails(self) -> None:
        with (
            mock.patch.object(self.controller, "_stop_edge"),
            mock.patch.object(self.controller, "stop_source"),
            mock.patch.object(self.controller, "stop_media_server"),
            mock.patch.object(stream_lab_server.time, "sleep"),
            mock.patch.object(self.controller, "start_source") as start_source,
            mock.patch.object(
                self.controller, "start_pipeline", side_effect=RuntimeError("offline")
            ),
            mock.patch.object(self.controller, "stop_source") as stop_source,
        ):
            with self.assertRaisesRegex(RuntimeError, "offline"):
                self.controller.start_all({"video": "test.mp4"})
        start_source.assert_called_once_with("test.mp4")
        self.assertEqual(stop_source.call_count, 2)

    def test_source_is_normalized_for_realtime_streaming(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            upload_root = Path(directory)
            (upload_root / "4k.mp4").write_bytes(b"video")
            process = mock.Mock()
            process.poll.return_value = None
            with (
                mock.patch.object(stream_lab_server, "UPLOAD_ROOT", upload_root),
                mock.patch.object(self.controller, "start_media_server"),
                mock.patch.object(self.controller, "stop_source"),
                mock.patch.object(stream_lab_server.subprocess, "Popen", return_value=process) as popen,
                mock.patch.object(stream_lab_server.time, "sleep"),
            ):
                self.controller.start_source("4k.mp4")
        command = popen.call_args.args[0]
        video_filter = command[command.index("-vf") + 1]
        self.assertIn("min(1280,iw)", video_filter)
        self.assertIn("fps=25", video_filter)
        self.assertEqual(command[command.index("-g") + 1], "25")

    def test_rejects_unavailable_hardware_encoder(self) -> None:
        with self.assertRaisesRegex(ValueError, "不可用"):
            self.controller.start_pipeline({"encoder": "h264_v4l2m2m"})

    def test_forwards_selected_model_to_edge(self) -> None:
        with mock.patch.object(
            stream_lab_server, "json_request", return_value={"ok": True}
        ) as request:
            self.controller.start_pipeline(
                {"encoder": "libx264", "model": "model_v2.rknn"}
            )
        payload = request.call_args.args[2]
        self.assertEqual(payload["model"], "model_v2.rknn")

    def test_start_source_rejects_path_outside_upload_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            outside = Path(directory) / "outside.mp4"
            outside.write_bytes(b"video")
            with mock.patch.object(stream_lab_server, "UPLOAD_ROOT", Path(directory) / "uploads"):
                with self.assertRaisesRegex(ValueError, "does not exist"):
                    self.controller.start_source(str(outside))


if __name__ == "__main__":
    unittest.main()
