"""Post-processing helpers for predictions and funscript output."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from .postprocessor import PostProcessConfig, apply_postprocessing
from .models import PostprocessOptionsModel
from .version_info import APP_NAME, APP_VERSION

GENERATOR_NAME = APP_NAME
GENERATOR_VERSION = APP_VERSION


def build_postprocess_config(options: PostprocessOptionsModel, frame_rate: float) -> PostProcessConfig:
    return PostProcessConfig(
        frame_rate=frame_rate,
        smooth_window_frames=options.smooth_window_frames,
        prominence_ratio=options.prominence_ratio,
        min_prominence=options.min_prominence,
        max_slope=options.max_slope,
        boost_slope=options.boost_slope,
        min_slope=options.min_slope,
        merge_threshold_ms=options.merge_threshold_ms,
        fft_denoise=options.fft_denoise,
        fft_frames_per_component=options.fft_frames_per_component,
        fft_window_frames=options.fft_window_frames,
    )


def run_postprocess(
    predictions: pd.DataFrame,
    options: PostprocessOptionsModel,
    frame_rate: float,
) -> pd.DataFrame:
    config = build_postprocess_config(options, frame_rate)
    processed = apply_postprocessing(predictions, config)
    processed.attrs.setdefault("options", options.dict())
    processed.attrs.setdefault("frame_rate", frame_rate)
    return processed


def write_funscript(
    processed: pd.DataFrame,
    output_path: Path,
    model_name: str,
    options: PostprocessOptionsModel,
) -> None:
    frame_rate = processed.attrs.get("frame_rate", 30.0)
    actions = _build_actions(processed, frame_rate)
    payload = {
        "version": "1.0",
        "inverted": False,
        "range": 100,
        "actions": actions,
        "generator": {
            "name": GENERATOR_NAME,
            "version": GENERATOR_VERSION,
            "model": model_name,
            "options": options.dict(),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    actions_json = json.dumps(actions, ensure_ascii=False, separators=(",", ":"))
    formatted = json.dumps(payload, ensure_ascii=False, indent=2)
    flattened = _inline_actions_array(formatted, actions_json)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(flattened)
        if not flattened.endswith("\n"):
            handle.write("\n")


def _inline_actions_array(payload_json: str, actions_json: str) -> str:
    lines = payload_json.splitlines()
    result: List[str] = []
    skip_until = -1
    closing_comma = False
    indent = ""
    for idx, line in enumerate(lines):
        if idx <= skip_until:
            continue
        stripped = line.lstrip()
        if not indent and stripped.startswith('"actions"'):
            indent = line[: len(line) - len(stripped)]
            # Locate closing bracket line
            closing_index = idx
            closing_comma = False
            for probe in range(idx, len(lines)):
                probe_stripped = lines[probe].lstrip()
                if probe_stripped.startswith("]"):
                    closing_index = probe
                    closing_comma = probe_stripped.endswith(",")
                    break
            skip_until = closing_index
            suffix = "," if closing_comma else ""
            result.append(f"{indent}\"actions\": {actions_json}{suffix}")
        else:
            result.append(line)
    return "\n".join(result)


def _build_actions(processed: pd.DataFrame, frame_rate: float) -> List[Dict[str, int]]:
    meta = processed.attrs.get("postprocess", {}) if hasattr(processed, "attrs") else {}
    graph_points = meta.get("graph_points", []) if isinstance(meta, dict) else []
    actions: List[Dict[str, int]] = []
    if graph_points:
        last_time = -1
        step_ms = 1000.0 / frame_rate if frame_rate > 1e-6 else 33.3
        for point in graph_points:
            position = float(point.get("position", 0.0))
            value = float(point.get("value", 50.0))
            time_ms = int(round(position * step_ms))
            if last_time >= 0 and time_ms <= last_time:
                time_ms = last_time + 1
            amplitude = int(round(float(np.clip(value, 0.0, 100.0))))
            actions.append({"at": max(0, time_ms), "pos": amplitude})
            last_time = time_ms
    if not actions:
        series = processed.get("processed_value") if isinstance(processed, pd.DataFrame) else None
        if series is not None:
            step_ms = 1000.0 / frame_rate if frame_rate > 1e-6 else 33.3
            for idx, value in enumerate(np.asarray(series, dtype=np.float64)):
                time_ms = int(round(idx * step_ms))
                amplitude = int(round(float(np.clip(value, 0.0, 100.0))))
                actions.append({"at": max(0, time_ms), "pos": amplitude})
    if not actions:
        actions.append({"at": 0, "pos": 50})
    deduped: Dict[int, int] = {}
    for action in sorted(actions, key=lambda item: item["at"]):
        deduped[action["at"]] = action["pos"]
    final_actions = []
    last_time = -1
    for time_ms, amplitude in sorted(deduped.items()):
        if last_time >= 0 and time_ms <= last_time:
            time_ms = last_time + 1
        final_actions.append({"at": time_ms, "pos": amplitude})
        last_time = time_ms
    return final_actions


__all__ = [
    "run_postprocess",
    "write_funscript",
    "build_postprocess_config",
]
