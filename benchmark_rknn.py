#!/usr/bin/env python3
"""Benchmark warmed-up RKNN inference on one or more RK3576 core modes."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from infer_rknn import preprocess_frame
from runtime_utils import CORE_MODES, create_runtime


@dataclass(frozen=True)
class BenchmarkStats:
    mean_ms: float
    median_ms: float
    p95_ms: float
    minimum_ms: float
    maximum_ms: float

    @property
    def fps(self) -> float:
        return 1000.0 / self.mean_ms


def summarize(samples_ms: list[float]) -> BenchmarkStats:
    if not samples_ms:
        raise ValueError("At least one benchmark sample is required")
    samples = np.asarray(samples_ms, dtype=np.float64)
    return BenchmarkStats(
        mean_ms=float(samples.mean()),
        median_ms=float(np.median(samples)),
        p95_ms=float(np.percentile(samples, 95)),
        minimum_ms=float(samples.min()),
        maximum_ms=float(samples.max()),
    )


def benchmark_runtime(
    runtime, input_batch: np.ndarray, warmup_runs: int, measured_runs: int
) -> tuple[BenchmarkStats, list[tuple[int, ...]]]:
    outputs = None
    for _ in range(warmup_runs):
        outputs = runtime.inference(inputs=[input_batch], data_format=["nhwc"])
        if not outputs:
            raise RuntimeError("RKNN runtime returned no output during warmup")

    samples_ms: list[float] = []
    for _ in range(measured_runs):
        started = time.perf_counter()
        outputs = runtime.inference(inputs=[input_batch], data_format=["nhwc"])
        samples_ms.append((time.perf_counter() - started) * 1000)
        if not outputs:
            raise RuntimeError("RKNN runtime returned no output during benchmark")

    output_shapes = [tuple(np.asarray(output).shape) for output in outputs or []]
    return summarize(samples_ms), output_shapes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image", type=Path, help="Optional image; zeros are used when omitted")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument(
        "--cores",
        default="auto",
        help="Comma-separated modes: auto,core0,core1,both",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if not args.model.is_file():
        parser.error(f"RKNN model does not exist: {args.model}")
    if args.image is not None and not args.image.is_file():
        parser.error(f"Image does not exist: {args.image}")
    if args.warmup < 0 or args.runs <= 0:
        parser.error("--warmup must be non-negative and --runs must be positive")
    core_modes = [mode.strip() for mode in args.cores.split(",") if mode.strip()]
    invalid_modes = [mode for mode in core_modes if mode not in CORE_MODES]
    if not core_modes or invalid_modes:
        parser.error(f"Invalid core modes: {invalid_modes or args.cores}")
    args.core_modes = core_modes
    return args


def make_input(image_path: Path | None, image_size: int) -> np.ndarray:
    if image_path is None:
        return np.zeros((1, image_size, image_size, 3), dtype=np.uint8)
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise RuntimeError(f"Failed to read image: {image_path}")
    input_batch, _ = preprocess_frame(frame, image_size)
    return input_batch


def main() -> None:
    args = parse_args()
    input_batch = make_input(args.image, args.imgsz)
    print(
        f"model={args.model.resolve()} input_shape={input_batch.shape} "
        f"warmup={args.warmup} runs={args.runs}"
    )
    for core_mode in args.core_modes:
        runtime = create_runtime(args.model, core_mode, args.verbose)
        try:
            sdk_version = runtime.get_sdk_version()
            stats, output_shapes = benchmark_runtime(
                runtime, input_batch, args.warmup, args.runs
            )
        finally:
            runtime.release()
        print(
            f"cores={core_mode:<5} mean_ms={stats.mean_ms:.2f} "
            f"median_ms={stats.median_ms:.2f} p95_ms={stats.p95_ms:.2f} "
            f"min_ms={stats.minimum_ms:.2f} max_ms={stats.maximum_ms:.2f} "
            f"fps={stats.fps:.2f} outputs={output_shapes} sdk={sdk_version}"
        )


if __name__ == "__main__":
    main()
