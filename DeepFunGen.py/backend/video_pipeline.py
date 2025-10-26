"""Video decoding and ONNX inference pipeline."""
from __future__ import annotations

import csv
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import pandas as pd

from .onnx_runner import OnnxSequenceModel


@dataclass
class PipelineResult:
    frame_count: int
    fps: float
    timestamps: np.ndarray
    predicted_change: np.ndarray
    predictions_df: pd.DataFrame
    prediction_path: Path
    model_name: str


class ProcessingCancelled(Exception):
    """Raised when a job cancellation is detected mid-processing."""


def resolve_prediction_path(video_path: Path, model_path: Path) -> Path:
    video_dir = video_path.parent
    csv_name = f"{video_path.stem}.{model_path.stem}.csv"
    return video_dir / csv_name


def resolve_script_path(video_path: Path) -> Path:
    return video_path.parent / f"{video_path.stem}.funscript"


def process_video(
    video_path: Path,
    model: OnnxSequenceModel,
    *,
    progress_cb: Callable[[float, str], None],
    should_cancel: Callable[[], bool],
    log_cb: Callable[[str], None],
) -> PipelineResult:
    """Run the core video -> predictions pipeline."""

    if should_cancel():
        raise ProcessingCancelled()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 1e-3:
        fps = 30.0

    total_est = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_est <= 0:
        total_est = 0

    per_frame_shape = (model.HEIGHT, model.WIDTH, model.CHANNELS)
    window = deque(maxlen=model.SEQUENCE_LENGTH)
    predicted_changes: list[float] = []

    frame_index = 0
    frame_ms = 1000.0 / fps
    log_cb(f"Decoding video at ~{fps:.2f} fps")

    try:
        while True:
            if should_cancel():
                raise ProcessingCancelled()

            ret, frame = cap.read()
            if not ret or frame is None:
                break
            if frame.size == 0:
                continue

            resized = cv2.resize(frame, (model.WIDTH, model.HEIGHT), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            normalized = rgb.astype(np.float32) / 255.0

            if normalized.shape != per_frame_shape:
                normalized = normalized.reshape(per_frame_shape)

            window.append(normalized)

            if len(predicted_changes) <= frame_index:
                predicted_changes.extend([0.0] * (frame_index - len(predicted_changes) + 1))

            if len(window) == model.SEQUENCE_LENGTH:
                sequence = np.stack(window, axis=0)
                value = model.infer(sequence)
                predicted_changes[frame_index] = value
            else:
                predicted_changes[frame_index] = 0.0

            frame_index += 1

            if total_est > 0:
                progress = min(1.0, frame_index / total_est)
                if frame_index % max(1, total_est // 20 or 1) == 0:
                    progress_cb(progress, f"Processing {frame_index}/{total_est} frames")
            else:
                if frame_index % 30 == 0:
                    progress_cb(float("nan"), f"Processing {frame_index} frames")

        progress_cb(0.95, "Finalising predictions")
    finally:
        cap.release()

    frame_count = frame_index
    if frame_count == 0:
        raise RuntimeError("No frames decoded from video")

    predicted = np.array(predicted_changes[:frame_count], dtype=np.float32)
    if predicted.size < frame_count:
        extra = frame_count - predicted.size
        predicted = np.pad(predicted, (0, extra), constant_values=0.0)

    cutoff = min(model.SEQUENCE_LENGTH - 1, predicted.size)
    if cutoff > 0:
        predicted[:cutoff] = 0.0

    timestamps = np.arange(frame_count, dtype=np.float64) * frame_ms

    df = pd.DataFrame(
        {
            "frame_index": np.arange(frame_count, dtype=np.int32),
            "timestamp_ms": timestamps,
            "predicted_change": predicted,
        }
    )

    model_name = model.model_path.stem
    prediction_path = resolve_prediction_path(video_path, model.model_path)
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    with prediction_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame_index", "timestamp_ms", "predicted_change"])
        for frame_idx in range(frame_count):
            writer.writerow([
                frame_idx,
                f"{timestamps[frame_idx]:.6f}",
                f"{predicted[frame_idx]:.9f}",
            ])

    return PipelineResult(
        frame_count=frame_count,
        fps=fps,
        timestamps=timestamps,
        predicted_change=predicted,
        predictions_df=df,
        prediction_path=prediction_path,
        model_name=model_name,
    )


__all__ = [
    "PipelineResult",
    "ProcessingCancelled",
    "process_video",
    "resolve_prediction_path",
    "resolve_script_path",
]
