"""Video decoding and ONNX inference pipeline."""
from __future__ import annotations

import csv
import os
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

from queue import Empty, Full, Queue
from threading import Event, Thread

import cv2
import numpy as np
import pandas as pd

try:  # Optional fast decoder
    import av  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    av = None  # type: ignore

_AVError = getattr(av, "AVError", Exception)

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
    stats: Dict[str, int] = field(default_factory=dict)


class ProcessingCancelled(Exception):
    """Raised when a job cancellation is detected mid-processing."""


def _parse_env_flag(*keys: str) -> Optional[bool]:
    """Return first parsed boolean flag from the provided environment keys."""
    truthy = {"1", "true", "yes", "on"}
    falsy = {"0", "false", "no", "off"}
    for key in keys:
        raw = os.environ.get(key)
        if raw is None:
            continue
        value = raw.strip().lower()
        if value in truthy:
            return True
        if value in falsy:
            return False
    return None


def _to_float(value: Optional[object]) -> float:
    """Best-effort conversion of PyAV numeric types to float."""
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        numerator = getattr(value, "numerator", None)
        denominator = getattr(value, "denominator", None)
        if numerator is not None and denominator not in (None, 0):
            try:
                return float(numerator) / float(denominator)
            except (TypeError, ValueError, ZeroDivisionError):
                return 0.0
    return 0.0


_VR_FOCUS_TOP = 0.20
_VR_FOCUS_BOTTOM = 0.0
_VR_FOCUS_LEFT = 0.10
_VR_FOCUS_RIGHT = 0.10


def apply_vr_focus_crop(frame: np.ndarray) -> np.ndarray:
    """Crop VR footage: keep left-eye view then focus on lower centre."""
    if frame.ndim < 2:
        return frame
    height, width = frame.shape[:2]
    if height == 0 or width == 0:
        return frame
    if width > 1:
        width = width // 2
        frame = frame[:, :width, :]
    top = min(height - 1, max(0, int(round(height * _VR_FOCUS_TOP))))
    bottom = min(
        height,
        max(top + 1, height - int(round(height * _VR_FOCUS_BOTTOM))),
    )
    left = min(width - 1, max(0, int(round(width * _VR_FOCUS_LEFT))))
    right = min(
        width,
        max(left + 1, width - int(round(width * _VR_FOCUS_RIGHT))),
    )
    if bottom <= top or right <= left:
        return frame
    return frame[top:bottom, left:right]


class FramePreprocessor:
    """Resize, colour convert, and normalise frames with optional GPU assist."""

    def __init__(
        self,
        *,
        width: int,
        height: int,
        channels: int,
        log_cb: Callable[[str], None],
        prefer_gpu: bool,
    ) -> None:
        self.width = width
        self.height = height
        self.channels = channels
        self._prefer_gpu = prefer_gpu
        self._use_umat = False
        self._log_cb = log_cb
        self._scale = 1.0 / 255.0
        self._color_code = None
        if channels == 1:
            self._color_code = cv2.COLOR_BGR2GRAY
        elif channels == 3:
            self._color_code = cv2.COLOR_BGR2RGB
        else:
            raise ValueError(f"Unsupported channel count for preprocessing: {channels}")
        self._initialise_backend()

    # ------------------------------------------------------------------
    def _initialise_backend(self) -> None:
        if not self._prefer_gpu:
            return
        if not hasattr(cv2, "UMat"):
            self._log_cb("GPU preprocessing requested but cv2.UMat is unavailable; using CPU.")
            return
        have_opencl = False
        try:
            have_opencl = cv2.ocl.haveOpenCL() if hasattr(cv2, "ocl") else False
        except cv2.error:
            have_opencl = False
        if not have_opencl:
            self._log_cb("GPU preprocessing requested but OpenCL is not available; using CPU.")
            return
        try:
            cv2.ocl.setUseOpenCL(True)
        except cv2.error:
            self._log_cb("Failed to enable OpenCL for preprocessing; using CPU.")
            return
        self._use_umat = True
        self._log_cb("Using OpenCL-accelerated preprocessing via cv2.UMat.")

    # ------------------------------------------------------------------
    def prepare(self, frame: np.ndarray) -> np.ndarray:
        if self._use_umat:
            try:
                return self._prepare_with_umat(frame)
            except cv2.error as exc:
                self._log_cb(f"OpenCL preprocessing failed ({exc}); falling back to CPU.")
                self._use_umat = False
        return self._prepare_cpu(frame)

    # ------------------------------------------------------------------
    def _prepare_with_umat(self, frame: np.ndarray) -> np.ndarray:
        umat = cv2.UMat(frame)
        resized = cv2.resize(umat, (self.width, self.height), interpolation=cv2.INTER_AREA)
        converted = cv2.cvtColor(resized, self._color_code) if self._color_code is not None else resized
        array = converted.get().astype(np.float32, copy=False)
        array *= self._scale
        if array.ndim == 2:
            array = array[:, :, np.newaxis]
        return array

    # ------------------------------------------------------------------
    def _prepare_cpu(self, frame: np.ndarray) -> np.ndarray:
        resized = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        converted = cv2.cvtColor(resized, self._color_code) if self._color_code is not None else resized
        array = converted.astype(np.float32, copy=False)
        array *= self._scale
        if array.ndim == 2:
            array = array[:, :, np.newaxis]
        return array


def resolve_prediction_path(video_path: Path, model_path: Path) -> Path:
    video_dir = video_path.parent
    csv_name = f"{video_path.stem}.{model_path.stem}.csv"
    return video_dir / csv_name


def resolve_script_path(video_path: Path) -> Path:
    return video_path.parent / f"{video_path.stem}.funscript"


def _try_enable_hw_decode(cap: cv2.VideoCapture, log_cb: Callable[[str], None]) -> None:
    """Attempt to enable hardware-accelerated decoding when supported."""
    hw_prop = getattr(cv2, "CAP_PROP_HW_ACCELERATION", None)
    accel_any = getattr(cv2, "VIDEO_ACCELERATION_ANY", None)
    if hw_prop is None or accel_any is None:
        return
    try:
        if cap.set(hw_prop, accel_any):
            log_cb("Requested hardware-accelerated video decoding.")
    except cv2.error:
        return


def process_video(
    video_path: Path,
    model: OnnxSequenceModel,
    *,
    progress_cb: Callable[[float, str], None],
    should_cancel: Callable[[], bool],
    log_cb: Callable[[str], None],
    use_vr_left_half: bool = False,
    target_size: Optional[Tuple[int, int]] = None,
    prefer_gpu_preprocess: Optional[bool] = None,
    stats_cb: Optional[Callable[[int, int, Optional[int]], None]] = None,
) -> PipelineResult:
    """Run the core video -> predictions pipeline."""

    if should_cancel():
        raise ProcessingCancelled()

    cap: Optional[cv2.VideoCapture] = None
    container = None
    stream = None
    use_pyav = False

    fps = 0.0
    total_est = 0

    if av is not None:
        try:
            container = av.open(str(video_path))
            video_streams = [candidate for candidate in container.streams if getattr(candidate, "type", None) == "video"]
            if not video_streams:
                raise RuntimeError("No video stream present")
            stream = video_streams[0]
            try:
                stream.thread_type = "AUTO"
            except (AttributeError, _AVError):  # pragma: no cover - optional optimisation
                pass
            fps = _to_float(getattr(stream, "average_rate", None))
            if fps <= 1e-3:
                fps = _to_float(getattr(stream, "guessed_rate", None))
            total_est = int(getattr(stream, "frames", 0) or 0)
            use_pyav = True
            log_cb("Using PyAV decoder for preprocessing pipeline.")
        except Exception as exc:
            if container is not None:
                try:
                    container.close()
                except Exception:
                    pass
            container = None
            stream = None
            log_cb(f"PyAV unavailable ({exc}); falling back to OpenCV decoding.")

    if not use_pyav:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video: {video_path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        total_est = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 1e-3:
        fps = 30.0
    if total_est <= 0:
        total_est = 0

    model_height = int(getattr(model, "height", getattr(model, "HEIGHT", 224)))
    model_width = int(getattr(model, "width", getattr(model, "WIDTH", 224)))
    channels = int(getattr(model, "channels", getattr(model, "CHANNELS", 3)))
    sequence_length = int(getattr(model, "sequence_length", getattr(model, "SEQUENCE_LENGTH", 10)))
    if target_size is not None:
        override_h, override_w = target_size
        model_height = int(override_h)
        model_width = int(override_w)
        log_cb(f"Overriding frame size to {model_width}x{model_height} for inference.")
    per_frame_shape = (model_height, model_width, channels)
    if prefer_gpu_preprocess is None:
        env_flag = _parse_env_flag("DEEPFUNGEN_GPU_PREPROCESS", "DEEPFUNGEN_USE_GPU_PREPROCESS")
        prefer_gpu_preprocess = bool(env_flag) if env_flag is not None else False
    else:
        prefer_gpu_preprocess = bool(prefer_gpu_preprocess)
    if prefer_gpu_preprocess and cap is not None:
        _try_enable_hw_decode(cap, log_cb)
    preprocessor = FramePreprocessor(
        width=model_width,
        height=model_height,
        channels=channels,
        log_cb=log_cb,
        prefer_gpu=bool(prefer_gpu_preprocess),
    )
    window = deque(maxlen=sequence_length)
    predicted_sums: list[float] = []
    predicted_counts: list[int] = []
    buffer_target = 100
    buffer_size = max(sequence_length, min(buffer_target, total_est if total_est > 0 else buffer_target))
    frame_queue: "Queue[Optional[tuple[int, np.ndarray]]]" = Queue(maxsize=buffer_size)
    producer_done = Event()
    producer_exc: Optional[Exception] = None
    counts = {"preprocessed": 0, "inferred": 0}
    frames_total_actual: Optional[int] = None

    def emit_stats() -> None:
        if stats_cb is None:
            return
        total_hint = frames_total_actual or (total_est if total_est > 0 else None)
        try:
            stats_cb(counts["preprocessed"], counts["inferred"], total_hint)
        except Exception:
            pass

    frame_ms = 1000.0 / fps
    log_cb(f"Decoding video at ~{fps:.2f} fps")
    if use_vr_left_half:
        log_cb("VR layout detected; applying focus crop (top 20%, left/right 10%).")

    def ensure_capacity(target_idx: int) -> None:
        if target_idx < 0:
            return
        missing = target_idx + 1 - len(predicted_sums)
        if missing > 0:
            predicted_sums.extend([0.0] * missing)
            predicted_counts.extend([0] * missing)

    def producer() -> None:
        nonlocal producer_exc, frames_total_actual
        frame_idx = 0
        try:
            if use_pyav and container is not None and stream is not None:
                for frame in container.decode(stream):
                    if should_cancel():
                        break
                    try:
                        array = frame.to_ndarray(format="bgr24")
                    except Exception:
                        continue
                    if array.size == 0:
                        continue
                    if use_vr_left_half and array.shape[1] > 1:
                        array = apply_vr_focus_crop(array)
                    normalized = preprocessor.prepare(array)
                    while True:
                        if should_cancel():
                            return
                        try:
                            frame_queue.put((frame_idx, normalized), timeout=0.05)
                            break
                        except Full:
                            if should_cancel():
                                return
                            continue
                    frame_idx += 1
                    counts["preprocessed"] = frame_idx
                    emit_stats()
            else:
                while True:
                    if should_cancel():
                        break
                    ret, frame = cap.read() if cap is not None else (False, None)
                    if not ret or frame is None:
                        break
                    if frame.size == 0:
                        continue
                    if use_vr_left_half and frame.shape[1] > 1:
                        frame = apply_vr_focus_crop(frame)
                    normalized = preprocessor.prepare(frame)
                    while True:
                        if should_cancel():
                            return
                        try:
                            frame_queue.put((frame_idx, normalized), timeout=0.05)
                            break
                        except Full:
                            if should_cancel():
                                return
                            continue
                    frame_idx += 1
                    counts["preprocessed"] = frame_idx
                    emit_stats()
            frames_total_actual = frame_idx
            emit_stats()
        except Exception as exc:
            producer_exc = exc
        finally:
            producer_done.set()
            pushed = False
            while not pushed:
                try:
                    frame_queue.put(None, timeout=0.05)
                    pushed = True
                except Full:
                    if should_cancel():
                        break
                    continue

    producer_thread = Thread(target=producer, name="video-preprocessor", daemon=True)
    producer_thread.start()

    latest_frame_index = -1
    latest_prediction_index = -1
    try:
        while True:
            if should_cancel():
                raise ProcessingCancelled()
            try:
                item = frame_queue.get(timeout=0.05)
            except Empty:
                if producer_done.is_set():
                    break
                continue
            if item is None:
                break
            idx, normalized = item
            latest_frame_index = max(latest_frame_index, idx)

            if normalized.shape != per_frame_shape:
                normalized = normalized.reshape(per_frame_shape)

            window.append(normalized)

            if len(window) == sequence_length:
                sequence = np.stack(window, axis=0)
                values = np.asarray(model.infer(sequence), dtype=np.float32).reshape(-1)
                if values.size == 0:
                    continue
                start_idx = idx - values.size + 1
                for offset, value in enumerate(values):
                    target_idx = start_idx + offset
                    if target_idx < 0:
                        continue
                    ensure_capacity(target_idx)
                    predicted_sums[target_idx] += float(value)
                    predicted_counts[target_idx] += 1
                    if target_idx > latest_prediction_index:
                        latest_prediction_index = target_idx
                if latest_prediction_index >= 0:
                    counts["inferred"] = max(counts["inferred"], latest_prediction_index + 1)
                emit_stats()

                total_hint = frames_total_actual or (total_est if total_est > 0 else None)
                if total_hint:
                    progress = min(1.0, counts["inferred"] / max(1.0, float(total_hint)))
                else:
                    progress = float("nan")
                message_total = total_hint if total_hint else "?"
                message = (
                    f"Processing {counts['preprocessed']}/{message_total} frames · "
                    f"Inferred {counts['inferred']}"
                )
                progress_cb(progress, message)

        progress_cb(0.95, "Finalising predictions")
    finally:
        producer_thread.join()
        if cap is not None:
            cap.release()
        if container is not None:
            try:
                container.close()
            except Exception:
                pass
        if producer_exc:
            raise producer_exc

    frame_count = (
        frames_total_actual
        if frames_total_actual is not None
        else (latest_frame_index + 1 if latest_frame_index >= 0 else counts["preprocessed"])
    )
    if frame_count == 0:
        raise RuntimeError("No frames decoded from video")

    ensure_capacity(frame_count - 1)
    sums_arr = np.asarray(predicted_sums[:frame_count], dtype=np.float32)
    counts_arr = np.asarray(predicted_counts[:frame_count], dtype=np.float32)
    predicted = np.zeros(frame_count, dtype=np.float32)
    valid = counts_arr > 0
    predicted[valid] = sums_arr[valid] / counts_arr[valid]

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

    if frames_total_actual is None:
        frames_total_actual = frame_count
    counts["preprocessed"] = max(counts["preprocessed"], frame_count)
    emit_stats()

    stats = {
        "frames_total": frame_count,
        "frames_preprocessed": counts["preprocessed"],
        "frames_inferred": counts["inferred"],
    }

    return PipelineResult(
        frame_count=frame_count,
        fps=fps,
        timestamps=timestamps,
        predicted_change=predicted,
        predictions_df=df,
        prediction_path=prediction_path,
        model_name=model_name,
        stats=stats,
    )


__all__ = [
    "PipelineResult",
    "ProcessingCancelled",
    "process_video",
    "resolve_prediction_path",
    "resolve_script_path",
]
