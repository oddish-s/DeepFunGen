"""Optimised frame extraction for viewer previews."""
from __future__ import annotations

import logging
import os
from collections import OrderedDict
from pathlib import Path
from threading import Lock
from typing import Optional

import cv2
import numpy as np

from .video_pipeline import apply_vr_focus_crop

logger = logging.getLogger("deepfungen.preview")

try:  # PyAV for fast seeking/decoding
    import av  # type: ignore
except ImportError:  # pragma: no cover - optional acceleration
    av = None  # type: ignore

_AVError = getattr(av, "AVError", Exception)

try:  # TurboJPEG for faster JPEG encoding
    from turbojpeg import TJPF_BGR, TurboJPEG  # type: ignore
except ImportError:  # pragma: no cover - optional acceleration
    TurboJPEG = None  # type: ignore
    TJPF_BGR = None  # type: ignore


_MAX_CACHE_ENTRIES = 4
_JPEG_QUALITY = None
_PREVIEW_CACHE: Optional["_PreviewCache"] = None
_TURBO_JPEG = None

if TurboJPEG is not None:
    lib_path = os.environ.get("DEEPFUNGEN_TURBOJPEG_PATH")
    try:
        _TURBO_JPEG = TurboJPEG(lib_path=lib_path) if lib_path else TurboJPEG()
    except Exception as exc:  # pragma: no cover - optional dependency failure
        logger.warning(
            "TurboJPEG unavailable; preview encoding falls back to cv2.imencode (%s)",
            exc,
        )
        _TURBO_JPEG = None


def _read_jpeg_quality() -> int:
    raw = os.environ.get("DEEPFUNGEN_PREVIEW_JPEG_QUALITY")
    if raw is None:
        return 60
    try:
        value = int(raw)
    except ValueError:
        return 60
    return max(10, min(95, value))


_JPEG_QUALITY = _read_jpeg_quality()


def _parse_env_flag(*keys: str) -> Optional[bool]:
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


def _prefer_gpu_preprocess(pref_override: Optional[bool]) -> bool:
    if pref_override is not None:
        return bool(pref_override)
    env_flag = _parse_env_flag("DEEPFUNGEN_GPU_PREPROCESS", "DEEPFUNGEN_USE_GPU_PREPROCESS")
    if env_flag is not None:
        return bool(env_flag)
    return False


def _try_enable_hw_decode(capture: cv2.VideoCapture) -> None:
    hw_prop = getattr(cv2, "CAP_PROP_HW_ACCELERATION", None)
    accel_any = getattr(cv2, "VIDEO_ACCELERATION_ANY", None)
    if hw_prop is None or accel_any is None:
        return
    try:
        capture.set(hw_prop, accel_any)
    except cv2.error:
        return


def _encode_jpeg(image_bgr: np.ndarray) -> bytes:
    if _TURBO_JPEG is not None and TJPF_BGR is not None:
        try:
            return _TURBO_JPEG.encode(image_bgr, quality=_JPEG_QUALITY, pixel_format=TJPF_BGR)
        except Exception:  # pragma: no cover - fallback to OpenCV
            logger.debug("TurboJPEG encode failed; falling back to cv2", exc_info=True)
    success, buffer = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), _JPEG_QUALITY])
    if not success:
        raise ValueError("Failed to encode frame")
    return buffer.tobytes()


def _prepare_frame(image_bgr: np.ndarray, max_width: int, use_vr_left_half: bool) -> np.ndarray:
    frame = image_bgr
    if use_vr_left_half and frame.shape[1] > 1:
        frame = apply_vr_focus_crop(frame)
    if max_width > 0 and frame.shape[1] > max_width:
        ratio = max_width / frame.shape[1]
        height = max(1, int(frame.shape[0] * ratio))
        frame = cv2.resize(frame, (max_width, height), interpolation=cv2.INTER_AREA)
    return frame


def _decode_with_opencv(
    video_path: Path,
    timestamp_ms: float,
    max_width: int,
    use_vr_left_half: bool,
    prefer_gpu_preprocess: Optional[bool],
) -> bytes:
    capture = cv2.VideoCapture(str(video_path))
    if _prefer_gpu_preprocess(prefer_gpu_preprocess):
        _try_enable_hw_decode(capture)
    if not capture.isOpened():
        capture.release()
        raise FileNotFoundError("Video file could not be opened")
    if timestamp_ms is not None:
        capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, float(timestamp_ms)))
    success, frame = capture.read()
    capture.release()
    if not success or frame is None:
        raise ValueError("Frame unavailable")
    prepared = _prepare_frame(frame, max_width, use_vr_left_half)
    return _encode_jpeg(prepared)


class _PreviewDecoder:
    """Holds an open PyAV container for fast random access decoding."""

    def __init__(self, video_path: Path) -> None:
        if av is None:  # pragma: no cover - guard
            raise RuntimeError("PyAV is not available")
        self.path = video_path
        self._container = av.open(str(video_path))
        streams = [stream for stream in self._container.streams if stream.type == "video"]
        if not streams:
            raise ValueError("Video stream not found")
        self._stream = streams[0]
        try:
            self._stream.thread_type = "AUTO"
        except (AttributeError, _AVError):  # pragma: no cover - optional setting
            pass
        self._lock = Lock()

    def close(self) -> None:
        with self._lock:
            try:
                self._container.close()
            except Exception:
                pass

    def _frame_step_ms(self) -> float:
        rate = _to_float(getattr(self._stream, "average_rate", None))
        if rate > 0.0:
            return 1000.0 / rate
        guessed = _to_float(getattr(self._stream, "guessed_rate", None))
        if guessed > 0.0:
            return 1000.0 / guessed
        return 0.0

    def decode(self, timestamp_ms: float, max_width: int, use_vr_left_half: bool) -> bytes:
        target_ms = max(0.0, float(timestamp_ms))
        with self._lock:
            try:
                self._container.flush()
            except _AVError:  # pragma: no cover - defensive
                pass
            stream = self._stream
            time_base = _to_float(stream.time_base)
            if time_base <= 0.0:
                rate = _to_float(stream.average_rate)
                if rate > 0.0:
                    time_base = 1.0 / rate
                else:
                    guessed = _to_float(getattr(stream, "guessed_rate", None))
                    time_base = 1.0 / guessed if guessed > 0.0 else 1.0 / 30.0
            target_seconds = target_ms / 1000.0
            target_pts = int(target_seconds / time_base) if time_base > 0 else int(target_seconds * 1e6)
            try:
                self._container.seek(target_pts, stream=stream, any_frame=False, backward=True)
            except _AVError:
                self._container.seek(0, stream=stream, any_frame=True)

            chosen = None
            frame_step_ms = self._frame_step_ms()
            for frame in self._container.decode(stream):
                frame_time = frame.time
                if frame_time is None and frame.pts is not None:
                    frame_time = frame.pts * _to_float(stream.time_base)
                frame_ms = frame_time * 1000.0 if frame_time is not None else None
                chosen = frame
                if frame_ms is None:
                    break
                if frame_ms + frame_step_ms >= target_ms:
                    break
            if chosen is None:
                raise ValueError("Frame unavailable")
            image = chosen.to_ndarray(format="bgr24")
        prepared = _prepare_frame(image, max_width, use_vr_left_half)
        return _encode_jpeg(prepared)


class _PreviewCache:
    """LRU cache of preview decoders keyed by absolute video path."""

    def __init__(self, max_entries: int = _MAX_CACHE_ENTRIES) -> None:
        self._max_entries = max(1, max_entries)
        self._cache: "OrderedDict[Path, _PreviewDecoder]" = OrderedDict()
        self._lock = Lock()

    def decoder_for(self, video_path: Path) -> _PreviewDecoder:
        key = video_path.resolve()
        with self._lock:
            decoder = self._cache.get(key)
            if decoder is not None:
                self._cache.move_to_end(key, last=True)
                return decoder
            decoder = _PreviewDecoder(key)
            self._cache[key] = decoder
            if len(self._cache) > self._max_entries:
                old_key, old_decoder = self._cache.popitem(last=False)
                try:
                    old_decoder.close()
                except Exception:
                    pass
            return decoder


def _ensure_cache() -> Optional[_PreviewCache]:
    global _PREVIEW_CACHE  # noqa: PLW0603 - module-level cache
    if av is None:
        return None
    if _PREVIEW_CACHE is None:
        _PREVIEW_CACHE = _PreviewCache()
    return _PREVIEW_CACHE


def _to_float(value: Optional[object]) -> float:
    if value is None:
        return 0.0
    try:
        return float(value)
    except Exception:
        return 0.0


def decode_preview_frame(
    video_path: Path,
    timestamp_ms: float,
    max_width: int,
    *,
    use_vr_left_half: bool = False,
    prefer_gpu_preprocess: Optional[bool] = None,
) -> bytes:
    cache = _ensure_cache()
    if cache is not None:
        try:
            decoder = cache.decoder_for(video_path)
            return decoder.decode(timestamp_ms, max_width, use_vr_left_half)
        except Exception:
            logger.debug("PyAV preview decode failed; falling back to OpenCV", exc_info=True)
    return _decode_with_opencv(video_path, timestamp_ms, max_width, use_vr_left_half, prefer_gpu_preprocess)


__all__ = ["decode_preview_frame"]
