#!/usr/bin/env python3
"""Shared RK3576 RKNNLite runtime helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any


CORE_MODES = ("auto", "core0", "core1", "both")


def resolve_core_mask(rknn_lite_type: Any, core_mode: str) -> int:
    masks = {
        "auto": rknn_lite_type.NPU_CORE_AUTO,
        "core0": rknn_lite_type.NPU_CORE_0,
        "core1": rknn_lite_type.NPU_CORE_1,
        "both": rknn_lite_type.NPU_CORE_0_1,
    }
    try:
        return masks[core_mode]
    except KeyError as exc:
        raise ValueError(f"Unsupported core mode: {core_mode}") from exc


def create_runtime(model: Path, core_mode: str = "auto", verbose: bool = False):
    try:
        from rknnlite.api import RKNNLite
    except ImportError as exc:
        raise RuntimeError(
            "rknn-toolkit-lite2 is not installed. Install the v2.3.2 "
            "AArch64 wheel that matches the board Python version."
        ) from exc

    runtime = RKNNLite(verbose=verbose)
    if runtime.load_rknn(str(model.resolve())) != 0:
        runtime.release()
        raise RuntimeError(f"Failed to load RKNN model: {model}")

    core_mask = resolve_core_mask(RKNNLite, core_mode)
    if runtime.init_runtime(core_mask=core_mask) != 0:
        runtime.release()
        raise RuntimeError(
            "Failed to initialize RK3576 runtime. Check librknnrt, Lite2, "
            "model target, and RKNPU driver versions."
        )
    return runtime
