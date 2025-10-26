"""API route definitions for the DeepFunGen Python app."""
from __future__ import annotations

import asyncio
import io
import logging
import math
from pathlib import Path
from typing import List

import cv2
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, StreamingResponse

from .config import MODEL_SEARCH_PATHS
from .jobs import JobRecord, JobStore
from .logstream import LogBroker
from .models import (
    AddJobsRequest,
    JobDetail,
    JobStatus,
    JobSummary,
    SettingsModel,
    PostprocessOptionsModel,
)
from .storage import SettingsManager, StateRepository
from .telemetry import TelemetryCollector
from .postprocess import run_postprocess
from .worker import ProcessingWorker

SUPPORTED_VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".m4v",
    ".avi",
    ".mkv",
    ".mpg",
    ".mpeg",
    ".wmv",
}

router = APIRouter(prefix="/api")
logger = logging.getLogger("deepfungen.api")


# ---------------------------------------------------------------------------
# Dependency helpers
# ---------------------------------------------------------------------------

def _get_store(request: Request) -> JobStore:
    return request.app.state.job_store


def _get_repo(request: Request) -> StateRepository:
    return request.app.state.state_repo


def _get_settings_manager(request: Request) -> SettingsManager:
    return request.app.state.settings_manager


def _get_worker(request: Request) -> ProcessingWorker:
    return request.app.state.worker


def _get_logs(request: Request) -> LogBroker:
    return request.app.state.log_broker


def _get_telemetry(request: Request) -> TelemetryCollector:
    return request.app.state.telemetry


def _persist(request: Request) -> None:
    store: JobStore = _get_store(request)
    repo: StateRepository = _get_repo(request)
    settings_manager: SettingsManager = _get_settings_manager(request)
    repo.dump_from(store, settings_manager.current())


def _series_to_list(series):
    return [None if pd.isna(value) else float(value) for value in series]


def _sequence_to_list(values, expected_length: int | None = None) -> list[float | None]:
    if values is None:
        return []
    if hasattr(values, "tolist"):
        try:
            raw = values.tolist()
        except Exception:
            raw = None
    else:
        try:
            raw = list(values)
        except Exception:
            raw = None
    if raw is None:
        return []
    if expected_length is not None and len(raw) != expected_length:
        return []
    result: list[float | None] = []
    for item in raw:
        if item is None:
            result.append(None)
            continue
        try:
            numeric = float(item)
        except (TypeError, ValueError):
            result.append(None)
            continue
        if math.isnan(numeric):
            result.append(None)
        else:
            result.append(numeric)
    return result


def _extract_video_frame(video_path: Path, timestamp_ms: float,  max_width: int = 480) -> bytes:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise FileNotFoundError("Video file could not be opened")
    if timestamp_ms is not None:
        capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, float(timestamp_ms)))
    success, frame = capture.read()
    capture.release()
    if not success or frame is None:
        raise ValueError("Frame unavailable")
    success, buffer = cv2.imencode(".jpg", frame)
    if not success:
        raise ValueError("Failed to encode frame")

    # --- 리사이즈 단계 ---
    h, w = frame.shape[:2]
    if w > max_width:
        ratio = max_width / w
        frame = cv2.resize(frame, (max_width, int(h * ratio)), interpolation=cv2.INTER_AREA)

    # --- JPEG 압축 인코딩 ---
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 60]  # 0~100 (기본 95), 80은 용량 확 줄어듦
    success, buffer = cv2.imencode(".jpg", frame, encode_param)
    if not success:
        raise ValueError("Failed to encode frame")

    return buffer.tobytes()



# ---------------------------------------------------------------------------
# Queue management
# ---------------------------------------------------------------------------

@router.get("/queue", response_model=List[JobSummary])
async def list_queue(store: JobStore = Depends(_get_store)) -> List[JobSummary]:
    return store.summaries()


@router.post("/queue/add")
async def add_jobs(
    payload: AddJobsRequest,
    request: Request,
    store: JobStore = Depends(_get_store),
    settings_manager: SettingsManager = Depends(_get_settings_manager),
    worker: ProcessingWorker = Depends(_get_worker),
) -> JSONResponse:
    added = 0
    skipped: List[str] = []
    default_model = settings_manager.current().default_model_path

    paths = list(payload.video_paths)
    expanded: set[str] = set()
    for raw_path in paths:
        path_obj = Path(raw_path).expanduser()
        if path_obj.is_dir():
            iterator = path_obj.rglob('*') if payload.recursive else path_obj.glob('*')
            for candidate in iterator:
                if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS:
                    expanded.add(str(candidate.resolve()))
            continue
        if path_obj.is_file():
            expanded.add(str(path_obj.resolve()))

    for raw_path in expanded:
        normalized = str(Path(raw_path).resolve())
        if not Path(normalized).exists() or not Path(normalized).is_file():
            skipped.append(raw_path)
            continue
        if Path(normalized).suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
            skipped.append(raw_path)
            continue
        target_model = payload.model_path or default_model
        if store.exists_for(normalized, target_model, payload.postprocess_options):
            skipped.append(raw_path)
            continue

        record = JobRecord(
            video_path=normalized,
            model_path=payload.model_path or default_model,
            output_directory=payload.output_directory,
            postprocess_options=payload.postprocess_options,
        )
        store.add_job(record)
        added += 1

    if added:
        await worker.start()
        _persist(request)

    return JSONResponse({
        "success": added > 0,
        "added_count": added,
        "skipped": skipped,
    })


@router.get("/jobs/{job_id}", response_model=JobDetail)
async def get_job(job_id: str, store: JobStore = Depends(_get_store)) -> JobDetail:
    record = store.get(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return record.to_detail()


@router.post("/queue/{job_id}/cancel")
async def cancel_job(job_id: str, request: Request, store: JobStore = Depends(_get_store)) -> JSONResponse:
    record = store.get(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if record.status not in {JobStatus.PENDING, JobStatus.PROCESSING}:
        raise HTTPException(status_code=400, detail="Job cannot be cancelled")
    record.apply_status(JobStatus.CANCELLED, message="Cancelled by user")
    record.update_progress(0.0)
    _persist(request)
    return JSONResponse({"success": True})


@router.post("/queue/clear_finished")
async def clear_finished(request: Request, store: JobStore = Depends(_get_store)) -> JSONResponse:
    removed = store.clear_finished()
    if removed:
        _persist(request)
    return JSONResponse({"success": True, "removed_count": len(removed)})


# ---------------------------------------------------------------------------
# Results endpoints (completed jobs reuse queue summary for now)
# ---------------------------------------------------------------------------

@router.get("/results", response_model=List[JobSummary])
async def list_results(store: JobStore = Depends(_get_store)) -> List[JobSummary]:
    return [summary for summary in store.summaries() if summary.status == JobStatus.COMPLETED]


@router.get("/results/{job_id}/frame")
async def get_result_frame(
    job_id: str,
    timestamp_ms: float = 0.0,
    store: JobStore = Depends(_get_store),
) -> StreamingResponse:
    record = store.get(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Job not found")
    video_path = Path(record.video_path)
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video file missing")
    try:
        data = await run_in_threadpool(_extract_video_frame, video_path, timestamp_ms)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Video file missing")
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return StreamingResponse(io.BytesIO(data), media_type="image/jpeg")


@router.get("/results/{job_id}")
async def get_result_detail(job_id: str, store: JobStore = Depends(_get_store)):
    record = store.get(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if record.status != JobStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Job not completed")
    if not record.prediction_path:
        raise HTTPException(status_code=404, detail="Prediction file missing")
    csv_path = Path(record.prediction_path)
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail="Prediction file missing")
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read prediction file: {exc}") from exc
    options = record.postprocess_options or PostprocessOptionsModel()
    processed = run_postprocess(df.copy(), options, record.frame_rate or 30.0)
    postprocess_meta = processed.attrs.get("postprocess", {}) if hasattr(processed, "attrs") else {}
    fft_values = []
    if isinstance(postprocess_meta, dict):
        fft_values = _sequence_to_list(postprocess_meta.get("fft_denoised_signal"), len(df))
    result = {
        "job": record.to_detail(),
        "predictions": {
            "timestamps": df.get('timestamp_ms', pd.Series(dtype=float)).tolist(),
            "predicted_change": df.get('predicted_change', pd.Series(dtype=float)).tolist(),
            "processed_value": processed.get('processed_value', pd.Series(dtype=float)).tolist(),
            "processed_change": processed.get('processed_change', pd.Series(dtype=float)).tolist(),
            "phase_marker": _series_to_list(processed.get('phase_marker', pd.Series(dtype=float))),
            "phase_source": processed.get('phase_source', pd.Series(dtype=str)).tolist(),
            "fft_denoised": fft_values,
        },
        "script_path": record.script_path,
    }
    return result



# ---------------------------------------------------------------------------
# Models & settings
# ---------------------------------------------------------------------------

@router.get("/models")
async def list_models(request: Request) -> JSONResponse:
    models = []
    seen_paths: set[str] = set()
    for root in MODEL_SEARCH_PATHS:
        path = Path(root)
        if not path.exists() or not path.is_dir():
            continue
        for candidate in path.rglob("*.onnx"):
            resolved = str(candidate.resolve())
            if resolved in seen_paths:
                continue
            seen_paths.add(resolved)
            models.append({
                "path": resolved,
                "name": candidate.name,
                "display_name": candidate.stem,
            })
    models.sort(key=lambda item: item["display_name"].lower(), reverse=True)
    return JSONResponse({
        "models": models,
        "current_model": getattr(request.app.state, "current_model_path", None),
        "execution_provider": getattr(request.app.state, "execution_provider", None),
    })


@router.get("/settings", response_model=SettingsModel)
async def get_settings(settings_manager: SettingsManager = Depends(_get_settings_manager)) -> SettingsModel:
    return settings_manager.current()


@router.post("/settings", response_model=SettingsModel)
async def update_settings(
    payload: SettingsModel,
    request: Request,
    settings_manager: SettingsManager = Depends(_get_settings_manager),
) -> SettingsModel:
    settings_manager.replace(payload)
    _persist(request)
    return settings_manager.current()


# ---------------------------------------------------------------------------
# Telemetry & logs
# ---------------------------------------------------------------------------

@router.get("/system/status")
async def system_status(
    request: Request,
    store: JobStore = Depends(_get_store),
    telemetry: TelemetryCollector = Depends(_get_telemetry),
) -> JSONResponse:
    counts = store.counts()
    status = telemetry.snapshot(
        queue_counts=counts,
        execution_provider=getattr(request.app.state, "execution_provider", None),
    )
    return JSONResponse(status.dict())


@router.post("/system/shutdown")
async def system_shutdown(
    request: Request,
    worker: ProcessingWorker = Depends(_get_worker),
) -> JSONResponse:
    await worker.stop(cancel_running=True, reason="Window closed")
    _persist(request)
    server = getattr(request.app.state, "uvicorn_server", None)
    if server is not None:
        server.should_exit = True
    return JSONResponse({"success": True})


@router.get("/logs/stream")
async def stream_logs(logs: LogBroker = Depends(_get_logs)) -> StreamingResponse:
    async def event_generator():
        try:
            async for event in logs.iter_events():
                yield f"data: {event.json()}\n\n"
        except (asyncio.CancelledError, ConnectionResetError, BrokenPipeError):
            return
        except Exception:  # pragma: no cover - defensive logging
            logger.debug("Log stream ended due to transport error", exc_info=True)
            return

    return StreamingResponse(event_generator(), media_type="text/event-stream")


__all__ = ["router", "SUPPORTED_VIDEO_EXTENSIONS"]
