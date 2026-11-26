#!/usr/bin/env python3
"""Split a monolithic YOLOv8 [xywh + score] ONNX output for INT8 export."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence


def split_monolithic_output(source: Path, destination: Path) -> None:
    try:
        import onnx
        from onnx import TensorProto, helper
    except ImportError as exc:
        raise RuntimeError("onnx is required to split the output graph") from exc

    model = onnx.load(str(source.resolve()), load_external_data=False)
    if len(model.graph.output) != 1:
        raise ValueError(
            f"Expected one monolithic output, got {len(model.graph.output)} outputs"
        )

    output = model.graph.output[0]
    dims = [dim.dim_value for dim in output.type.tensor_type.shape.dim]
    if dims != [1, 5, 8400]:
        raise ValueError(
            f"This splitter expects output shape [1, 5, 8400], got {dims}"
        )

    source_name = output.name
    starts_name = "rk3576_split_starts"
    starts_scores_name = "rk3576_split_starts_scores"
    ends_boxes_name = "rk3576_split_ends_boxes"
    ends_scores_name = "rk3576_split_ends_scores"
    axes_name = "rk3576_split_axes"
    steps_name = "rk3576_split_steps"
    boxes_name = "rk3576_boxes"
    scores_name = "rk3576_scores"

    model.graph.initializer.extend(
        [
            helper.make_tensor(starts_name, TensorProto.INT64, [1], [0]),
            helper.make_tensor(starts_scores_name, TensorProto.INT64, [1], [4]),
            helper.make_tensor(ends_boxes_name, TensorProto.INT64, [1], [4]),
            helper.make_tensor(ends_scores_name, TensorProto.INT64, [1], [5]),
            helper.make_tensor(axes_name, TensorProto.INT64, [1], [1]),
            helper.make_tensor(steps_name, TensorProto.INT64, [1], [1]),
        ]
    )
    model.graph.node.extend(
        [
            helper.make_node(
                "Slice",
                [source_name, starts_name, ends_boxes_name, axes_name, steps_name],
                [boxes_name],
                name="RK3576SplitBoxes",
            ),
            helper.make_node(
                "Slice",
                [
                    source_name,
                    starts_scores_name,
                    ends_scores_name,
                    axes_name,
                    steps_name,
                ],
                [scores_name],
                name="RK3576SplitScores",
            ),
        ]
    )

    del model.graph.output[:]
    model.graph.output.extend(
        [
            helper.make_tensor_value_info(
                boxes_name, TensorProto.FLOAT, [1, 4, 8400]
            ),
            helper.make_tensor_value_info(
                scores_name, TensorProto.FLOAT, [1, 1, 8400]
            ),
        ]
    )
    onnx.checker.check_model(model)
    destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(destination.resolve()))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.source.is_file():
        parser.error(f"ONNX model does not exist: {args.source}")
    return args


def main() -> None:
    args = parse_args()
    split_monolithic_output(args.source, args.output)
    print(f"Created split-output ONNX model: {args.output.resolve()}")


if __name__ == "__main__":
    main()
