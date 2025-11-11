"""Background worker that processes pending jobs."""
from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import pandas as pd

from .config import MODEL_SEARCH_PATHS
from .jobs import JobRecord, JobStore
from .logstream import LogBroker
from .models import JobStatus, LogEvent, PostprocessOptionsModel
from .onnx_runner import OnnxSequenceModel
from .postprocess import run_postprocess, write_funscript
from .storage import SettingsManager, StateRepository
from .video_pipeline import (
    PipelineResult,
    ProcessingCancelled,
    process_video,
    resolve_prediction_path,
    resolve_script_path,
)


class ProcessingWorker:
    """Serial worker that executes ONNX inference and post-processing."""

    def __init__(
        self,
        store: JobStore,
        logs: LogBroker,
        settings_manager: SettingsManager,
        repository: StateRepository,
    ) -> None:
        self.store = store
        self.logs = logs
        self.settings_manager = settings_manager
        self.repository = repository
        self._task: Optional[asyncio.Task] = None
        self._shutdown = asyncio.Event()
        self._model_cache: Dict[Path, OnnxSequenceModel] = {}
        self._model_change_cb: Optional[Callable[[Path, str], None]] = None

    # ------------------------------------------------------------------
    def set_model_change_callback(self, callback: Callable[[Path, str], None]) -> None:
        self._model_change_cb = callback

    # ------------------------------------------------------------------
    async def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self._shutdown.clear()
        loop = asyncio.get_running_loop()
        self._task = loop.create_task(self._run_loop(), name="processing-worker")

    async def stop(self, *, cancel_running: bool = False, reason: str = "Cancelled") -> None:
        if cancel_running:
            self.cancel_active_jobs(reason)
        self._shutdown.set()
        if self._task is not None:
            await self._task
            self._task = None

    # ------------------------------------------------------------------
    def run_job_sync(self, job: JobRecord) -> JobRecord:
        """Process a single job synchronously.

        This mirrors the async worker loop, allowing CLI use without spinning up
        the background task.
        """

        existing = self.store.get(job.id)
        if existing is None:
            self.store.add_job(job)
        elif existing is not job:
            raise ValueError(f"Job id conflict for {job.id}")

        job.apply_status(JobStatus.PROCESSING, message="Preparing inference")
        job.update_progress(0.02)
        self._log(job, "Processing started")

        try:
            self._execute_job(job)
        except ProcessingCancelled:
            job.apply_status(JobStatus.CANCELLED, message="Cancelled")
            job.update_progress(0.0)
            self._log(job, "Processing cancelled", level="WARNING")
        except Exception as exc:  # pragma: no cover - safety net
            job.error = str(exc)
            job.apply_status(JobStatus.FAILED, message=f"Failed: {exc}")
            job.update_progress(0.0)
            self._log(job, f"Processing failed: {exc}", level="ERROR")
        else:
            if job.status == JobStatus.PROCESSING:
                job.apply_status(JobStatus.COMPLETED, message="Completed")
                job.update_progress(1.0)
                self._log(job, "Processing completed")
        finally:
            self._persist()

        return job

    async def _run_loop(self) -> None:
        while not self._shutdown.is_set():
            job = self._next_pending_job()
            if job is None:
                try:
                    await asyncio.wait_for(self._shutdown.wait(), timeout=0.5)
                except asyncio.TimeoutError:
                    continue
                else:
                    break
            await self._process_job(job)

    def _next_pending_job(self) -> Optional[JobRecord]:
        pending = self.store.pending()
        return pending[0] if pending else None

    async def _process_job(self, job: JobRecord) -> None:
        job.apply_status(JobStatus.PROCESSING, message="Preparing inference")
        job.update_progress(0.02)
        self._log(job, "Processing started")

        try:
            await asyncio.to_thread(self._execute_job, job)
        except ProcessingCancelled:
            job.apply_status(JobStatus.CANCELLED, message="Cancelled")
            job.update_progress(0.0)
            self._log(job, "Processing cancelled", level="WARNING")
        except Exception as exc:  # pragma: no cover - safety net
            job.error = str(exc)
            job.apply_status(JobStatus.FAILED, message=f"Failed: {exc}")
            job.update_progress(0.0)
            self._log(job, f"Processing failed: {exc}", level="ERROR")
        else:
            if job.status == JobStatus.PROCESSING:
                job.apply_status(JobStatus.COMPLETED, message="Completed")
                job.update_progress(1.0)
                self._log(job, "Processing completed")
        finally:
            self._persist()

    # ------------------------------------------------------------------
    def _load_cached_predictions(self, job: JobRecord, model: OnnxSequenceModel) -> Optional[PipelineResult]:
        prediction_path = resolve_prediction_path(Path(job.video_path), model.model_path)
        if not prediction_path.exists():
            return None
        try:
            df = pd.read_csv(prediction_path)
        except Exception as exc:
            self._log(job, f"Failed to read cached predictions ({exc}); recalculating.", level="WARNING")
            return None
        required_columns = {"frame_index", "timestamp_ms", "predicted_change"}
        if not required_columns.issubset(df.columns):
            self._log(job, "Cached predictions missing expected columns; recalculating.", level="WARNING")
            return None
        frame_index = pd.to_numeric(df["frame_index"], errors="coerce").to_numpy(dtype=np.int32, copy=False)
        timestamps = pd.to_numeric(df["timestamp_ms"], errors="coerce").to_numpy(dtype=np.float64, copy=False)
        predicted_change = pd.to_numeric(df["predicted_change"], errors="coerce").to_numpy(dtype=np.float32, copy=False)
        if np.isnan(timestamps).any() or np.isnan(predicted_change).any():
            self._log(job, "Cached predictions contain NaN values; recalculating.", level="WARNING")
            return None
        frame_count = int(timestamps.size)
        if frame_count == 0:
            return None
        fps = self._infer_fps_from_timestamps(timestamps)
        cleaned = pd.DataFrame(
            {
                "frame_index": frame_index,
                "timestamp_ms": timestamps,
                "predicted_change": predicted_change,
            }
        )
        return PipelineResult(
            frame_count=frame_count,
            fps=fps,
            timestamps=timestamps,
            predicted_change=predicted_change,
            predictions_df=cleaned,
            prediction_path=prediction_path,
            model_name=model.model_path.stem,
        )

    @staticmethod
    def _infer_fps_from_timestamps(timestamps: np.ndarray) -> float:
        if timestamps.size <= 1:
            return 30.0
        diffs = np.diff(timestamps)
        diffs = diffs[np.isfinite(diffs) & (diffs > 0.01)]
        if diffs.size == 0:
            return 30.0
        avg_ms = float(np.mean(diffs))
        if avg_ms <= 1e-6:
            return 30.0
        fps = 1000.0 / avg_ms
        return float(max(1.0, min(240.0, fps)))

    def cancel_active_jobs(self, reason: str = "Cancelled") -> None:
        for job in self.store.all():
            if job.status in {JobStatus.PENDING, JobStatus.PROCESSING}:
                job.apply_status(JobStatus.CANCELLED, message=reason)
                job.update_progress(0.0)

    # ------------------------------------------------------------------
    def _execute_job(self, job: JobRecord) -> None:
        if job.status == JobStatus.CANCELLED:
            raise ProcessingCancelled()

        job.frames_preprocessed = 0
        job.frames_inferred = 0
        job.frames_total = None

        model_path = self._resolve_model_path(job)
        job.model_path = str(model_path)
        self._log(job, f"Using model: {model_path.name}")

        model = self._ensure_model(model_path)
        input_height = int(getattr(model, "height", getattr(model, "HEIGHT", 224)))
        input_width = int(getattr(model, "width", getattr(model, "WIDTH", 224)))
        job.model_input_height = input_height
        job.model_input_width = input_width

        use_vr_left_half = self._should_apply_vr_focus(model_path)
        job.use_vr_focus_crop = use_vr_left_half
        if use_vr_left_half:
            self._log(
                job,
                f"Model input {input_width}x{input_height} triggers VR focus crop.",
            )

        override_size: Optional[Tuple[int, int]] = None
        if input_height > 0 and input_width > 0:
            override_size = (input_height, input_width)
            default_size = (getattr(OnnxSequenceModel, "HEIGHT", 224), getattr(OnnxSequenceModel, "WIDTH", 224))
            if override_size == default_size:
                override_size = None

        post_opts = job.postprocess_options
        if post_opts is None:
            default_pp = self.settings_manager.current().default_postprocess
            post_opts = default_pp or PostprocessOptionsModel()
            job.postprocess_options = post_opts

        if self._model_change_cb is not None:
            self._model_change_cb(model_path, model.execution_provider)

        pipeline_result = self._load_cached_predictions(job, model)
        if pipeline_result is not None:
            job.frame_count = pipeline_result.frame_count
            job.frame_rate = pipeline_result.fps
            job.prediction_path = str(pipeline_result.prediction_path)
            job.frames_total = pipeline_result.frame_count
            job.frames_preprocessed = pipeline_result.frame_count
            job.frames_inferred = pipeline_result.frame_count
            job.update_progress(0.75, message="Using cached predictions")
            self._log(job, "Reusing cached predictions")
        else:

            def should_cancel() -> bool:
                return job.status == JobStatus.CANCELLED

            def stats_cb(preprocessed: int, inferred: int, total_hint: Optional[int]) -> None:
                job.frames_preprocessed = int(preprocessed)
                job.frames_inferred = int(inferred)
                if total_hint is not None and total_hint > 0:
                    job.frames_total = int(total_hint)

            def progress_cb(value: float, message: str) -> None:
                if job.status == JobStatus.CANCELLED:
                    return
                if np.isnan(value):
                    job.message = message
                else:
                    job.update_progress(value, message=message)

            pipeline_result = process_video(
                Path(job.video_path),
                model,
                progress_cb=progress_cb,
                should_cancel=should_cancel,
                log_cb=lambda msg: self._log(job, msg),
                use_vr_left_half=use_vr_left_half,
                target_size=override_size,
                stats_cb=stats_cb,
            )

            if should_cancel():
                raise ProcessingCancelled()

            job.frame_count = pipeline_result.frame_count
            job.frame_rate = pipeline_result.fps
            job.prediction_path = str(pipeline_result.prediction_path)

        stats = getattr(pipeline_result, "stats", {}) or {}
        if "frames_total" in stats:
            try:
                job.frames_total = int(stats["frames_total"])
            except (TypeError, ValueError):
                job.frames_total = job.frames_total or job.frame_count
        elif job.frames_total is None:
            job.frames_total = job.frame_count
        if "frames_preprocessed" in stats:
            try:
                job.frames_preprocessed = int(stats["frames_preprocessed"])
            except (TypeError, ValueError):
                pass
        if "frames_inferred" in stats:
            try:
                job.frames_inferred = int(stats["frames_inferred"])
            except (TypeError, ValueError):
                pass

        if job.status == JobStatus.CANCELLED:
            raise ProcessingCancelled()

        job.update_progress(0.97, message="Post-processing predictions")
        processed = run_postprocess(
            pipeline_result.predictions_df,
            post_opts,
            pipeline_result.fps,
        )

        job.update_progress(0.99, message="Writing outputs")
        script_path = resolve_script_path(Path(job.video_path))
        write_funscript(processed, script_path, pipeline_result.model_name, post_opts)
        job.script_path = str(script_path)
        job.message = "Outputs generated"

    # ------------------------------------------------------------------
    def _ensure_model(self, model_path: Path) -> OnnxSequenceModel:
        cached = self._model_cache.get(model_path)
        if cached is not None:
            return cached
        model = OnnxSequenceModel(model_path)
        self._model_cache[model_path] = model
        return model

    @staticmethod
    def _should_apply_vr_focus(model_path: Path) -> bool:
        return "vr" in model_path.stem.lower()

    def _resolve_model_path(self, job: JobRecord) -> Path:
        candidates = []
        if job.model_path:
            candidates.append(Path(job.model_path))
        settings_model = self.settings_manager.current().default_model_path
        if settings_model:
            candidates.append(Path(settings_model))
        for root in MODEL_SEARCH_PATHS:
            path = Path(root)
            if path.exists() and path.is_dir():
                for candidate in path.rglob("*.onnx"):
                    candidates.append(candidate)
        seen = set()
        for candidate in candidates:
            resolved = candidate.expanduser().resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            if resolved.exists():
                return resolved
        raise RuntimeError("No ONNX model available. Please configure a model path in settings.")

    def _persist(self) -> None:
        self.repository.dump_from(self.store, self.settings_manager.current())

    def _log(self, job: Optional[JobRecord], message: str, level: str = "INFO") -> None:
        event = LogEvent(
            timestamp=datetime.now().astimezone(),
            job_id=job.id if job else None,
            job_name=Path(job.video_path).name if job else None,
            level=level,
            message=message,
        )
        self.logs.publish(event)


__all__ = ["ProcessingWorker"]

