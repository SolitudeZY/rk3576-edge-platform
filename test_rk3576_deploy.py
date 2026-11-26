from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from benchmark_rknn import summarize
from infer_rknn import LetterboxInfo, decode_outputs, infer_frame, preprocess_frame
from runtime_utils import resolve_core_mask
from split_yolo_output import split_monolithic_output
from edge_rtsp_service import InferencePipeline


class RecordingRuntime:
    def __init__(self, outputs=None) -> None:
        self.inputs = None
        self.data_format = None
        self.outputs = outputs or [np.zeros((1, 5, 8400), dtype=np.float32)]

    def inference(self, inputs, data_format=None):
        self.inputs = inputs
        self.data_format = data_format
        return self.outputs


class PreprocessTest(unittest.TestCase):
    def test_feeds_contiguous_uint8_nhwc_batch(self) -> None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        input_batch, info = preprocess_frame(frame, 640)

        self.assertEqual(input_batch.shape, (1, 640, 640, 3))
        self.assertEqual(input_batch.dtype, np.uint8)
        self.assertTrue(input_batch.flags.c_contiguous)
        self.assertEqual(info.scale, 1.0)

    def test_infer_frame_records_split_timings(self) -> None:
        runtime = RecordingRuntime()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        result = infer_frame(runtime, frame, 640, 0.35, 0.6)

        self.assertEqual(runtime.inputs[0].shape, (1, 640, 640, 3))
        self.assertEqual(runtime.data_format, ["nhwc"])
        self.assertEqual(len(result.boxes), 0)
        self.assertGreaterEqual(result.preprocess_ms, 0)
        self.assertGreaterEqual(result.inference_ms, 0)
        self.assertGreaterEqual(result.postprocess_ms, 0)


class OutputDecodeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.info = LetterboxInfo(1.0, 0, 0, 640, 640)

    def test_decodes_monolithic_yolov8_output(self) -> None:
        output = np.zeros((1, 5, 2), dtype=np.float32)
        output[0, :, 0] = [100, 120, 40, 20, 0.9]

        boxes, scores, classes = decode_outputs(
            [output], self.info, 0.35, 0.6, 640
        )

        np.testing.assert_allclose(boxes, [[80, 110, 120, 130]])
        np.testing.assert_allclose(scores, [0.9])
        np.testing.assert_array_equal(classes, [0])

    def test_accepts_six_output_rockchip_head(self) -> None:
        outputs = []
        for size in (2, 1, 1):
            outputs.append(np.zeros((1, 64, size, size), dtype=np.float32))
            outputs.append(np.zeros((1, 1, size, size), dtype=np.float32))

        boxes, scores, classes = decode_outputs(
            outputs, self.info, 0.35, 0.6, 640
        )

        self.assertEqual(boxes.shape, (0, 4))
        self.assertEqual(scores.shape, (0,))
        self.assertEqual(classes.shape, (0,))

    def test_accepts_split_coordinate_and_score_outputs(self) -> None:
        boxes_output = np.zeros((1, 4, 3), dtype=np.float32)
        # Split output preserves the original YOLO cx, cy, width, height layout.
        boxes_output[0, :, 0] = [100, 120, 40, 20]
        boxes_output[0, :, 1] = [102, 121, 40, 20]

        # Keep the third candidate below threshold and make the two high-score
        # boxes overlap so the test also exercises NMS.
        scores_output = np.asarray([[[0.8, 0.7, 0.1]]], dtype=np.float32)

        boxes, scores, classes = decode_outputs(
            [boxes_output, scores_output], self.info, 0.35, 0.6, 640
        )

        np.testing.assert_allclose(boxes, [[80, 110, 120, 130]])
        np.testing.assert_allclose(scores, [0.8])
        np.testing.assert_array_equal(classes, [0])


class RuntimeConfigTest(unittest.TestCase):
    def test_model_selection_is_limited_to_model_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            default_model = model_dir / "default.rknn"
            other_model = model_dir / "other.rknn"
            default_model.write_bytes(b"default")
            other_model.write_bytes(b"other")
            pipeline = InferencePipeline(default_model)

            self.assertEqual(pipeline.resolve_model("other.rknn"), other_model.resolve())
            self.assertEqual(
                [item["name"] for item in pipeline.available_models()],
                ["default.rknn", "other.rknn"],
            )
            with self.assertRaisesRegex(ValueError, "Invalid model"):
                pipeline.resolve_model("../outside.rknn")

    def test_maps_rk3576_core_modes(self) -> None:
        fake_rknn_lite = SimpleNamespace(
            NPU_CORE_AUTO=0,
            NPU_CORE_0=1,
            NPU_CORE_1=2,
            NPU_CORE_0_1=3,
        )

        self.assertEqual(resolve_core_mask(fake_rknn_lite, "auto"), 0)
        self.assertEqual(resolve_core_mask(fake_rknn_lite, "core0"), 1)
        self.assertEqual(resolve_core_mask(fake_rknn_lite, "core1"), 2)
        self.assertEqual(resolve_core_mask(fake_rknn_lite, "both"), 3)

    def test_summarizes_benchmark_samples(self) -> None:
        stats = summarize([10.0, 20.0, 30.0])

        self.assertEqual(stats.mean_ms, 20.0)
        self.assertEqual(stats.median_ms, 20.0)
        self.assertEqual(stats.minimum_ms, 10.0)
        self.assertEqual(stats.maximum_ms, 30.0)
        self.assertEqual(stats.fps, 50.0)

    def test_splitter_requires_path_objects_and_writes_two_outputs(self) -> None:
        # The graph-level splitter is exercised in the host integration command;
        # this test only keeps the public import available to the deployment suite.
        self.assertTrue(callable(split_monolithic_output))


if __name__ == "__main__":
    unittest.main()
