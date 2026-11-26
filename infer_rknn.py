#!/usr/bin/env python3
"""Run a YOLOv8 RKNN model on RK3576 with split performance timings."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from runtime_utils import CORE_MODES, create_runtime


@dataclass(frozen=True)
class LetterboxInfo:
    scale: float
    left: int
    top: int
    original_width: int
    original_height: int


@dataclass(frozen=True)
class FrameResult:
    boxes: np.ndarray
    scores: np.ndarray
    classes: np.ndarray
    preprocess_ms: float
    inference_ms: float
    postprocess_ms: float


def letterbox(image: np.ndarray, size: int = 640) -> tuple[np.ndarray, LetterboxInfo]:
    height, width = image.shape[:2]
    scale = min(size / width, size / height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    left = (size - resized_width) // 2
    top = (size - resized_height) // 2
    right = size - resized_width - left
    bottom = size - resized_height - top
    padded = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    return padded, LetterboxInfo(scale, left, top, width, height)


def preprocess_frame(
    frame: np.ndarray, size: int
) -> tuple[np.ndarray, LetterboxInfo]:
    padded, info = letterbox(frame, size)
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(rgb[None], dtype=np.uint8), info


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    if len(boxes) == 0:
        return np.empty(0, dtype=np.int64)
    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size:
        index = int(order[0])
        keep.append(index)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[index], x1[rest])
        yy1 = np.maximum(y1[index], y1[rest])
        xx2 = np.minimum(x2[index], x2[rest])
        yy2 = np.minimum(y2[index], y2[rest])
        intersection = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[index] + areas[rest] - intersection
        iou = intersection / np.maximum(union, 1e-7)
        order = rest[iou <= iou_threshold]
    return np.asarray(keep, dtype=np.int64)


def dfl(position: np.ndarray) -> np.ndarray:
    if position.ndim != 4 or position.shape[1] % 4:
        raise ValueError(f"Invalid DFL tensor shape: {position.shape}")
    batch, channels, height, width = position.shape
    bins = channels // 4
    logits = position.reshape(batch, 4, bins, height, width).astype(np.float32)
    logits -= logits.max(axis=2, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=2, keepdims=True)
    values = np.arange(bins, dtype=np.float32).reshape(1, 1, bins, 1, 1)
    return (probabilities * values).sum(axis=2)


def decode_rockchip_heads(
    outputs: Sequence[np.ndarray], input_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(outputs) not in (6, 9):
        raise ValueError(f"Expected 6 or 9 Rockchip YOLO outputs, got {len(outputs)}")

    boxes_per_head: list[np.ndarray] = []
    scores_per_head: list[np.ndarray] = []
    pair_per_head = len(outputs) // 3
    for head_index in range(3):
        position = np.asarray(outputs[pair_per_head * head_index])
        class_scores = np.asarray(outputs[pair_per_head * head_index + 1])
        if position.ndim != 4 or class_scores.ndim != 4:
            raise ValueError(
                f"Expected NCHW head tensors, got {position.shape} and {class_scores.shape}"
            )

        height, width = position.shape[2:]
        grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
        grid = np.stack((grid_x, grid_y), axis=0).reshape(1, 2, height, width)
        stride = np.array(
            [input_size / width, input_size / height], dtype=np.float32
        ).reshape(1, 2, 1, 1)
        distances = dfl(position)
        top_left = (grid + 0.5 - distances[:, :2]) * stride
        bottom_right = (grid + 0.5 + distances[:, 2:]) * stride
        boxes = np.concatenate((top_left, bottom_right), axis=1)

        boxes_per_head.append(boxes.transpose(0, 2, 3, 1).reshape(-1, 4))
        scores_per_head.append(
            class_scores.transpose(0, 2, 3, 1).reshape(-1, class_scores.shape[1])
        )

    boxes = np.concatenate(boxes_per_head).astype(np.float32)
    class_scores = np.concatenate(scores_per_head).astype(np.float32)
    classes = class_scores.argmax(axis=1)
    scores = class_scores[np.arange(len(class_scores)), classes]
    return boxes, scores, classes


def decode_monolithic_output(
    output: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    predictions = np.asarray(output).squeeze()
    if predictions.ndim != 2:
        raise ValueError(f"Expected a 2-D YOLO output after squeeze, got {predictions.shape}")
    if predictions.shape[1] < 5 <= predictions.shape[0] or (
        predictions.shape[0] < predictions.shape[1] and predictions.shape[0] >= 5
    ):
        predictions = predictions.T
    if predictions.shape[1] < 5:
        raise ValueError(f"Expected xywh plus class scores, got {predictions.shape}")

    class_scores = predictions[:, 4:]
    classes = class_scores.argmax(axis=1)
    scores = class_scores[np.arange(len(class_scores)), classes]
    xywh = predictions[:, :4].astype(np.float32)
    boxes = np.empty_like(xywh)
    boxes[:, 0] = xywh[:, 0] - xywh[:, 2] / 2
    boxes[:, 1] = xywh[:, 1] - xywh[:, 3] / 2
    boxes[:, 2] = xywh[:, 0] + xywh[:, 2] / 2
    boxes[:, 3] = xywh[:, 1] + xywh[:, 3] / 2
    return boxes, scores, classes


def decode_split_output(
    boxes_output: np.ndarray, scores_output: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    boxes_tensor = np.asarray(boxes_output).squeeze()
    scores_tensor = np.asarray(scores_output).squeeze()
    if boxes_tensor.ndim != 2 or scores_tensor.ndim != 1:
        raise ValueError(
            f"Expected split outputs [1,4,N] and [1,1,N], got "
            f"{np.asarray(boxes_output).shape} and {np.asarray(scores_output).shape}"
        )
    if boxes_tensor.shape[0] == 4:
        xywh = boxes_tensor.T
    elif boxes_tensor.shape[1] == 4:
        xywh = boxes_tensor
    else:
        raise ValueError(f"Expected four box values per candidate, got {boxes_tensor.shape}")
    if scores_tensor.shape[0] != xywh.shape[0]:
        raise ValueError(
            f"Split box/score candidate counts differ: {xywh.shape} and {scores_tensor.shape}"
        )
    xywh = xywh.astype(np.float32)
    boxes = np.empty_like(xywh)
    boxes[:, 0] = xywh[:, 0] - xywh[:, 2] / 2
    boxes[:, 1] = xywh[:, 1] - xywh[:, 3] / 2
    boxes[:, 2] = xywh[:, 0] + xywh[:, 2] / 2
    boxes[:, 3] = xywh[:, 1] + xywh[:, 3] / 2
    scores = scores_tensor.astype(np.float32)
    classes = np.zeros(scores.shape[0], dtype=np.int64)
    return boxes, scores, classes


def decode_outputs(
    outputs: Sequence[np.ndarray],
    info: LetterboxInfo,
    confidence_threshold: float,
    iou_threshold: float,
    input_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(outputs) == 1:
        boxes, scores, classes = decode_monolithic_output(outputs[0])
    elif len(outputs) == 2:
        boxes, scores, classes = decode_split_output(outputs[0], outputs[1])
    elif len(outputs) in (6, 9):
        boxes, scores, classes = decode_rockchip_heads(outputs, input_size)
    else:
        shapes = [tuple(np.asarray(output).shape) for output in outputs]
        raise ValueError(f"Unsupported RKNN YOLO output layout: {shapes}")

    selected = scores >= confidence_threshold
    boxes, scores, classes = boxes[selected], scores[selected], classes[selected]
    if len(boxes) == 0:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.int64),
        )

    kept: list[int] = []
    for class_id in np.unique(classes):
        indices = np.flatnonzero(classes == class_id)
        kept.extend(indices[nms(boxes[indices], scores[indices], iou_threshold)].tolist())
    keep = np.asarray(kept, dtype=np.int64)
    keep = keep[np.argsort(scores[keep])[::-1]]
    boxes, scores, classes = boxes[keep], scores[keep], classes[keep]

    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - info.left) / info.scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - info.top) / info.scale
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, info.original_width - 1)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, info.original_height - 1)
    return boxes, scores.astype(np.float32), classes.astype(np.int64)


def infer_frame(
    runtime,
    frame: np.ndarray,
    input_size: int,
    confidence_threshold: float,
    iou_threshold: float,
) -> FrameResult:
    started = time.perf_counter()
    input_batch, info = preprocess_frame(frame, input_size)
    preprocess_ms = (time.perf_counter() - started) * 1000

    started = time.perf_counter()
    outputs = runtime.inference(inputs=[input_batch], data_format=["nhwc"])
    inference_ms = (time.perf_counter() - started) * 1000
    if not outputs:
        raise RuntimeError("RKNN runtime returned no output")

    started = time.perf_counter()
    boxes, scores, classes = decode_outputs(
        outputs, info, confidence_threshold, iou_threshold, input_size
    )
    postprocess_ms = (time.perf_counter() - started) * 1000
    return FrameResult(
        boxes,
        scores,
        classes,
        preprocess_ms,
        inference_ms,
        postprocess_ms,
    )


def load_labels(path: Path | None) -> list[str]:
    if path is None:
        return ["landslide"]
    labels = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return [label for label in labels if label]


def class_label(class_id: int, labels: Sequence[str]) -> str:
    return labels[class_id] if 0 <= class_id < len(labels) else f"class_{class_id}"


def draw_detections(
    image: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    labels: Sequence[str],
) -> np.ndarray:
    result = image.copy()
    for box, score, class_id in zip(boxes, scores, classes):
        x1, y1, x2, y2 = np.rint(box).astype(int)
        label = f"{class_label(int(class_id), labels)} {score:.2f}"
        cv2.rectangle(result, (x1, y1), (x2, y2), (0, 0, 255), 2)
        (label_width, label_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        label_x = max(0, x1)
        label_y = y1 - 6
        if label_y - label_height - baseline < 0:
            label_y = min(result.shape[0] - baseline, y1 + label_height + baseline + 6)
        cv2.rectangle(
            result,
            (label_x, label_y - label_height - baseline),
            (label_x + label_width + 6, label_y + 2),
            (0, 0, 255),
            -1,
        )
        cv2.putText(
            result,
            label,
            (label_x + 3, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        print(f"class={label} box={[x1, y1, x2, y2]}")
    return result


def is_image(path: str) -> bool:
    return Path(path).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_source(value: str) -> str | int:
    return int(value) if value.isdigit() else value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source", required=True, help="Image, video, RTSP URL, or camera index")
    parser.add_argument("--output", type=Path, default=Path("result.mp4"))
    parser.add_argument("--labels", type=Path, help="Optional class label file")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.6)
    parser.add_argument("--cores", choices=CORE_MODES, default="auto")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if not args.model.is_file():
        parser.error(f"RKNN model does not exist: {args.model}")
    if args.labels is not None and not args.labels.is_file():
        parser.error(f"Label file does not exist: {args.labels}")
    if args.warmup < 0 or args.max_frames < 0:
        parser.error("--warmup and --max-frames must be non-negative")
    return args


def warmup(runtime, input_batch: np.ndarray, runs: int) -> None:
    for _ in range(runs):
        outputs = runtime.inference(inputs=[input_batch], data_format=["nhwc"])
        if not outputs:
            raise RuntimeError("RKNN runtime returned no output during warmup")


def print_timings(result: FrameResult) -> None:
    total_ms = result.preprocess_ms + result.inference_ms + result.postprocess_ms
    print(
        f"preprocess_ms={result.preprocess_ms:.2f} "
        f"inference_ms={result.inference_ms:.2f} "
        f"postprocess_ms={result.postprocess_ms:.2f} "
        f"pipeline_ms={total_ms:.2f}"
    )


def main() -> None:
    args = parse_args()
    labels = load_labels(args.labels)
    runtime = create_runtime(args.model, args.cores, args.verbose)
    try:
        if is_image(args.source):
            frame = cv2.imread(args.source)
            if frame is None:
                raise RuntimeError(f"Failed to read image: {args.source}")
            warmup_input, _ = preprocess_frame(frame, args.imgsz)
            warmup(runtime, warmup_input, args.warmup)
            result = infer_frame(runtime, frame, args.imgsz, args.conf, args.iou)
            rendered = draw_detections(
                frame, result.boxes, result.scores, result.classes, labels
            )
            output = args.output.with_suffix(Path(args.source).suffix or ".jpg")
            output.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(output), rendered):
                raise RuntimeError(f"Failed to write image: {output}")
            print_timings(result)
            print(f"detections={len(result.boxes)} cores={args.cores} output={output}")
            return

        capture = cv2.VideoCapture(parse_source(args.source))
        if not capture.isOpened():
            raise RuntimeError(f"Failed to open video source: {args.source}")
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = capture.get(cv2.CAP_PROP_FPS)
        fps = fps if fps and fps > 0 else 25.0
        args.output.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if not writer.isOpened():
            capture.release()
            raise RuntimeError(f"Failed to create output video: {args.output}")

        frame_count = 0
        preprocess_ms = inference_ms = postprocess_ms = 0.0
        warmed_up = False
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if not warmed_up:
                    warmup_input, _ = preprocess_frame(frame, args.imgsz)
                    warmup(runtime, warmup_input, args.warmup)
                    warmed_up = True
                result = infer_frame(runtime, frame, args.imgsz, args.conf, args.iou)
                preprocess_ms += result.preprocess_ms
                inference_ms += result.inference_ms
                postprocess_ms += result.postprocess_ms
                writer.write(
                    draw_detections(
                        frame, result.boxes, result.scores, result.classes, labels
                    )
                )
                frame_count += 1
                if args.max_frames and frame_count >= args.max_frames:
                    break
        finally:
            capture.release()
            writer.release()

        if frame_count:
            pipeline_ms = preprocess_ms + inference_ms + postprocess_ms
            print(
                f"frames={frame_count} cores={args.cores} "
                f"avg_preprocess_ms={preprocess_ms / frame_count:.2f} "
                f"avg_inference_ms={inference_ms / frame_count:.2f} "
                f"avg_postprocess_ms={postprocess_ms / frame_count:.2f} "
                f"pipeline_fps={frame_count * 1000 / pipeline_ms:.2f} "
                f"output={args.output}"
            )
    finally:
        runtime.release()


if __name__ == "__main__":
    main()
