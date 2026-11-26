#!/usr/bin/env python3
"""Convert a YOLOv8 ONNX model to RKNN for RK3576."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True, help="Input ONNX model")
    parser.add_argument("--output", type=Path, required=True, help="Output .rknn model")
    parser.add_argument(
        "--precision",
        choices=("int8", "fp16"),
        default="int8",
        help="INT8 is recommended for deployment; FP16 is for accuracy debugging",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        help="Calibration image list required for INT8, one image path per line",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    if not args.onnx.is_file():
        parser.error(f"ONNX model does not exist: {args.onnx}")
    if args.precision == "int8" and (
        args.dataset is None or not args.dataset.is_file()
    ):
        parser.error("--precision int8 requires an existing --dataset list")
    return args


def check(ret: int, operation: str) -> None:
    if ret != 0:
        raise RuntimeError(f"RKNN {operation} failed with code {ret}")


def describe_onnx_outputs(model_path: Path) -> bool:
    try:
        import onnx
    except ImportError:
        return False

    model = onnx.load(str(model_path.resolve()), load_external_data=False)
    descriptions: list[str] = []
    for output in model.graph.output:
        dims = []
        for dim in output.type.tensor_type.shape.dim:
            dims.append(dim.dim_value if dim.dim_value else dim.dim_param or "?")
        descriptions.append(f"{output.name}: {dims}")
    print("ONNX outputs: " + ", ".join(descriptions))
    if len(model.graph.output) == 1:
        print(
            "WARNING: This is a monolithic Ultralytics output model. Do not use "
            "ordinary INT8 on [1,5,8400]: coordinate and score ranges share one "
            "quantization scale and scores can collapse to zero."
        )
        return True
    return False


def main() -> None:
    args = parse_args()
    try:
        from rknn.api import RKNN
    except ImportError as exc:
        raise SystemExit(
            "rknn-toolkit2 is not installed. Run conversion on the x86 host "
            "with RKNN-Toolkit2 v2.3.2."
        ) from exc

    monolithic_output = describe_onnx_outputs(args.onnx)
    if args.precision == "int8" and monolithic_output:
        raise SystemExit(
            "Refusing INT8 conversion for a monolithic YOLO output. Run "
            "split_yolo_output.py first, then convert the generated two-output ONNX."
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    do_quantization = args.precision == "int8"
    dataset = str(args.dataset.resolve()) if args.dataset else None

    rknn = RKNN(verbose=args.verbose)
    try:
        check(
            rknn.config(
                mean_values=[[0, 0, 0]],
                std_values=[[255, 255, 255]],
                target_platform="rk3576",
            ),
            "config",
        )
        check(rknn.load_onnx(model=str(args.onnx.resolve())), "load_onnx")
        check(
            rknn.build(do_quantization=do_quantization, dataset=dataset),
            "build",
        )
        check(rknn.export_rknn(str(args.output.resolve())), "export_rknn")
    finally:
        rknn.release()

    print(f"Created {args.precision.upper()} RK3576 model: {args.output.resolve()}")


if __name__ == "__main__":
    main()
