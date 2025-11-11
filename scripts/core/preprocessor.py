"""Video preprocessing utilities for FunTorch5."""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Callable

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


SUPPORTED_VIDEO_EXTENSIONS: Tuple[str, ...] = (".mp4", ".mov", ".avi", ".mkv")
DEFAULT_TARGET_SIZE: Tuple[int, int] = (224, 224)
DEFAULT_FRAME_TEMPLATE = "{frame:07d}.jpg"
FRAME_MODE_FILE = ".frame_mode.json"
FRAME_MODE_VR = "vr_focus_crop"
FRAME_MODE_FULL = "full"
_VR_FOCUS_TOP = 0.20
_VR_FOCUS_BOTTOM = 0.0
_VR_FOCUS_LEFT = 0.10
_VR_FOCUS_RIGHT = 0.10

try:  # Optional acceleration
    import av  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    av = None  # type: ignore

_AVError = getattr(av, "AVError", Exception)


@dataclass
class PreprocessResult:
    """Metadata describing the outcome of a preprocessing pass."""

    video_folder: Path
    frames_dir: Path
    meta_path: Optional[Path]
    frame_count: int
    reused_frames: bool
    reused_meta: bool


class VideoPreprocessor:
    """Handles frame extraction, metadata generation, and caching."""

    def __init__(
        self,
        target_size: Tuple[int, int] = DEFAULT_TARGET_SIZE,
        frame_template: str = DEFAULT_FRAME_TEMPLATE,
        allowed_video_exts: Sequence[str] = SUPPORTED_VIDEO_EXTENSIONS,
        image_quality: int = 95,
        aspect_ratio_bounds: Optional[Tuple[float, float]] = (0.75, 1.3333333333),
        use_vr_left_half: bool = False,
        prefer_pyav: Optional[bool] = None,
        log_fn: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.target_size = tuple(int(x) for x in target_size)
        self.frame_template = frame_template
        self.allowed_video_exts = tuple(e.lower() for e in allowed_video_exts)
        self.image_quality = int(image_quality)
        self.use_vr_left_half = bool(use_vr_left_half)
        if prefer_pyav is None:
            self.prefer_pyav = av is not None
        else:
            self.prefer_pyav = bool(prefer_pyav) and av is not None
        self._log_fn = log_fn
        if aspect_ratio_bounds is not None:
            lo, hi = aspect_ratio_bounds
            if hi <= 0 or lo <= 0 or hi < lo:
                raise ValueError("aspect_ratio_bounds must be positive and ordered (min, max)")
            self.aspect_ratio_bounds = (float(lo), float(hi))
        else:
            self.aspect_ratio_bounds = None

    def process_video_folder(self, video_folder: str | Path, force: bool = False) -> PreprocessResult:
        folder = Path(video_folder)
        if not folder.exists():
            raise FileNotFoundError(f"Video folder not found: {folder}")

        frames_dir = folder / "frames"
        meta_path = folder / "meta.csv"
        script_path = folder / "video.funscript"

        frames_available = frames_dir.exists() and any(frames_dir.glob("*.jpg"))
        meta_available = meta_path.exists()
        expected_mode = FRAME_MODE_VR if self.use_vr_left_half else FRAME_MODE_FULL
        current_mode = self._read_frame_mode(frames_dir) if frames_dir.exists() else "unknown"
        video_file = self._find_video_file(folder)

        need_rebuild = force or not frames_available or current_mode != expected_mode
        if (
            need_rebuild
            and current_mode != expected_mode
            and frames_available
            and video_file is None
            and not force
        ):
            need_rebuild = False

        if need_rebuild and frames_dir.exists():
            shutil.rmtree(frames_dir)
        frames_dir.mkdir(parents=True, exist_ok=True)

        reused_frames = frames_available and not need_rebuild
        reused_meta = meta_available and not force

        total_frames = 0
        fps = None

        if not reused_frames:
            if video_file is None:
                raise FileNotFoundError(
                    f"No video file found in {folder}. Supported extensions: {self.allowed_video_exts}"
                )
            fps, total_frames = self.extract_frames(video_file, frames_dir)
            self._write_frame_mode(frames_dir, expected_mode)
        else:
            total_frames = sum(1 for _ in frames_dir.glob("*.jpg"))
            if video_file is not None:
                fps = self._probe_fps(video_file)
            if current_mode != expected_mode:
                self._write_frame_mode(frames_dir, expected_mode)

        if script_path.exists() and (not reused_meta or not meta_available):
            if fps is None:
                if video_file is None:
                    raise RuntimeError(
                        f"Cannot infer FPS for {folder}: video file missing and no cached metadata"
                    )
                fps = self._probe_fps(video_file)
            meta_df = self.parse_script_json(script_path, fps=fps, total_frames=total_frames)
            self.create_meta_csv(frames_dir, meta_df, meta_path)
            reused_meta = False
        elif not script_path.exists():
            meta_path = None

        return PreprocessResult(
            video_folder=folder,
            frames_dir=frames_dir,
            meta_path=meta_path,
            frame_count=total_frames,
            reused_frames=reused_frames,
            reused_meta=reused_meta,
        )

    def extract_frames(self, video_path: Path, output_dir: Path) -> Tuple[float, int]:
        if self.prefer_pyav and av is not None:
            try:
                return self._extract_with_pyav(video_path, output_dir)
            except Exception as exc:
                self._log(f"PyAV preprocessing failed ({exc}); falling back to OpenCV.")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        frame_idx = 0
        writer_params = [int(cv2.IMWRITE_JPEG_QUALITY), self.image_quality]

        progress = tqdm(total=total_frames or None, desc=f"Extracting {video_path.name}", unit="frame")
        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                resized = self._prepare_frame(frame)
                output_path = output_dir / self.frame_template.format(frame=frame_idx)
                success = cv2.imwrite(str(output_path), resized, writer_params)
                if not success:
                    raise RuntimeError(f"Failed to write frame {output_path}")
                frame_idx += 1
                progress.update(1)
        finally:
            progress.close()
            cap.release()

        if frame_idx == 0:
            raise RuntimeError(f"No frames extracted from {video_path}")
        return float(fps), frame_idx

    def parse_script_json(self, script_path: Path, fps: float, total_frames: int) -> pd.DataFrame:
        with script_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        actions = data.get("actions") or []
        if not actions:
            raise ValueError(f"video.funscript has no 'actions' entries: {script_path}")

        actions = sorted(actions, key=lambda a: a["at"])
        timestamps_ms = np.array([float(a["at"]) for a in actions], dtype=np.float32)
        values = np.array([float(a["pos"]) for a in actions], dtype=np.float32)

        frame_times = np.arange(total_frames) / float(fps) * 1000.0
        interpolated = np.interp(frame_times, timestamps_ms, values, left=values[0], right=values[-1])
        diffs = np.zeros_like(interpolated)
        diffs[1:] = np.diff(interpolated)

        df = pd.DataFrame(
            {
                "frame_index": np.arange(total_frames, dtype=np.int32),
                "timestamp_ms": frame_times,
                "interference_value": interpolated,
                "interference_change": diffs,
            }
        )
        return df

    def create_meta_csv(self, frames_dir: Path, interpolated_data: pd.DataFrame, meta_path: Path) -> None:
        if not interpolated_data.empty:
            expected_frames = len(list(frames_dir.glob("*.jpg")))
            actual_frames = int(interpolated_data["frame_index"].max() + 1)
            if expected_frames != actual_frames:
                raise ValueError(
                    f"Frame count mismatch for {frames_dir.parent}: frames={expected_frames}, meta={actual_frames}"
                )
        interpolated_data.to_csv(meta_path, index=False)

    def _prepare_frame(self, frame: np.ndarray) -> np.ndarray:
        if self.use_vr_left_half and frame.shape[1] > 1:
            frame = self._apply_vr_focus_crop(frame)
        return self._resize_frame(frame)

    def _apply_vr_focus_crop(self, frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        if height == 0 or width == 0:
            return frame
        if width > 1:
            # Use left-eye view first
            width = width // 2
            frame = frame[:, :width, :]
        top_crop = int(round(height * _VR_FOCUS_TOP))
        bottom_crop = int(round(height * _VR_FOCUS_BOTTOM))
        left_crop = int(round(width * _VR_FOCUS_LEFT))
        right_crop = int(round(width * _VR_FOCUS_RIGHT))
        top = min(height - 1, max(0, top_crop))
        bottom = min(height, max(top + 1, height - bottom_crop))
        left = min(width - 1, max(0, left_crop))
        right = min(width, max(left + 1, width - right_crop))
        if bottom <= top or right <= left:
            return frame
        return frame[top:bottom, left:right]

    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        target_w, target_h = self.target_size
        src_h, src_w = frame.shape[:2]
        if src_h == 0 or src_w == 0:
            raise ValueError("Encountered empty frame during resize")

        if self.aspect_ratio_bounds is not None:
            min_ratio, max_ratio = self.aspect_ratio_bounds
            src_ratio = src_w / src_h if src_h else 0.0
            if src_ratio < min_ratio:
                desired_h = max(1, int(round(src_w / min_ratio)))
                crop_h = min(src_h, desired_h)
                top = max(0, (src_h - crop_h) // 2)
                frame = frame[top : top + crop_h, :]
            elif src_ratio > max_ratio:
                desired_w = max(1, int(round(src_h * max_ratio)))
                crop_w = min(src_w, desired_w)
                left = max(0, (src_w - crop_w) // 2)
                frame = frame[:, left : left + crop_w]

        return cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)

    def _read_frame_mode(self, frames_dir: Path) -> str:
        mode_path = frames_dir / FRAME_MODE_FILE
        if not mode_path.exists():
            return "unknown"
        try:
            with mode_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception:
            return "unknown"
        if isinstance(payload, dict):
            mode = payload.get("mode")
        elif isinstance(payload, str):
            mode = payload
        else:
            mode = None
        if mode == FRAME_MODE_VR:
            return FRAME_MODE_VR
        if mode == FRAME_MODE_FULL:
            return FRAME_MODE_FULL
        return "unknown"

    def _write_frame_mode(self, frames_dir: Path, mode: str) -> None:
        try:
            with (frames_dir / FRAME_MODE_FILE).open("w", encoding="utf-8") as handle:
                json.dump({"mode": mode}, handle)
        except Exception:
            pass

    def _find_video_file(self, folder: Path) -> Optional[Path]:
        for ext in self.allowed_video_exts:
            matches = sorted(folder.glob(f"*{ext}"))
            if matches:
                return matches[0]
        return None

    def _probe_fps(self, video_path: Path) -> float:
        if self.prefer_pyav and av is not None:
            try:
                container = av.open(str(video_path))
                try:
                    stream = next((s for s in container.streams if getattr(s, "type", None) == "video"), None)
                    if stream is not None:
                        fps = self._stream_fps(stream)
                        if fps > 0:
                            return fps
                finally:
                    container.close()
            except Exception:
                pass

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video for FPS probe: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()
        return float(fps)

    def _extract_with_pyav(self, video_path: Path, output_dir: Path) -> Tuple[float, int]:
        if av is None:
            raise RuntimeError("PyAV is not available")

        container = av.open(str(video_path))
        try:
            stream = next((s for s in container.streams if getattr(s, "type", None) == "video"), None)
            if stream is None:
                raise RuntimeError("No video stream found in container")

            fps = self._stream_fps(stream)
            if fps <= 0:
                fps = 30.0
            total_frames = int(getattr(stream, "frames", 0) or 0)
            frame_idx = 0
            writer_params = [int(cv2.IMWRITE_JPEG_QUALITY), self.image_quality]

            progress = tqdm(total=total_frames or None, desc=f"Extracting {video_path.name}", unit="frame")
            try:
                for frame in container.decode(stream):
                    try:
                        array = frame.to_ndarray(format="bgr24")
                    except _AVError:
                        continue
                    if array.size == 0:
                        continue
                    prepared = self._prepare_frame(array)
                    output_path = output_dir / self.frame_template.format(frame=frame_idx)
                    success = cv2.imwrite(str(output_path), prepared, writer_params)
                    if not success:
                        raise RuntimeError(f"Failed to write frame {output_path}")
                    frame_idx += 1
                    progress.update(1)
            finally:
                progress.close()

            if frame_idx == 0:
                raise RuntimeError(f"No frames extracted from {video_path}")
            return float(fps), frame_idx
        finally:
            container.close()

    def _stream_fps(self, stream) -> float:
        rate = getattr(stream, "average_rate", None)
        fps = self._to_float(rate)
        if fps <= 0:
            guessed = getattr(stream, "guessed_rate", None)
            fps = self._to_float(guessed)
        if fps <= 0:
            time_base = self._to_float(getattr(stream, "time_base", None))
            if time_base > 0:
                fps = 1.0 / time_base
        return float(fps) if fps > 0 else 0.0

    def _log(self, message: str) -> None:
        if self._log_fn is not None:
            try:
                self._log_fn(message)
            except Exception:
                pass
        else:
            try:
                tqdm.write(message)
            except Exception:
                pass

    @staticmethod
    def _to_float(value: Optional[object]) -> float:
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


__all__ = ["VideoPreprocessor", "PreprocessResult"]
