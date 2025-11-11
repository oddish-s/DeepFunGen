"""API route definitions for the DeepFunGen Python app."""
from __future__ import annotations

import asyncio
import io
import json
import logging
import math
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from subprocess import CalledProcessError
from typing import List, Optional, Tuple

import pandas as pd

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.encoders import jsonable_encoder
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
    LogEvent,
)
from .storage import SettingsManager, StateRepository
from .telemetry import TelemetryCollector
from .postprocess import run_postprocess, write_funscript
from .worker import ProcessingWorker
from .video_preview import decode_preview_frame
from .video_pipeline import resolve_script_path
from .version_info import get_version_info

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

_GITHUB_RELEASE_URL = "https://api.github.com/repos/oddish-s/DeepFunGen/releases/latest"
_UPDATE_CACHE_TTL = 3600.0
_UPDATE_ERROR_TTL = 60.0


class _UpdateCheckError(Exception):
    """Raised when the update check could not be completed."""


def _version_tuple(value: Optional[str]) -> Tuple[int, ...]:
    if not value:
        return ()
    text = value.strip()
    if not text:
        return ()
    if text[0] in ("v", "V"):
        text = text[1:]
    numbers: list[int] = []
    for chunk in text.replace("-", ".").split("."):
        if chunk.isdigit():
            numbers.append(int(chunk))
            continue
        stripped = "".join(ch for ch in chunk if ch.isdigit())
        if stripped:
            numbers.append(int(stripped))
            break
    return tuple(numbers)


def _log_job_event(logs: Optional[LogBroker], job: Optional[JobRecord], message: str, level: str = "INFO") -> None:
    if logs is None or job is None:
        return
    try:
        event = LogEvent(
            timestamp=datetime.now().astimezone(),
            job_id=job.id,
            job_name=Path(job.video_path).name if job.video_path else None,
            level=level,
            message=message,
        )
        logs.publish(event)
    except Exception:  # pragma: no cover - logging should never break request handling
        logging.getLogger("deepfungen.api").debug("Failed to publish log event", exc_info=True)


def _fetch_latest_release(timeout: float = 5.0) -> dict[str, Optional[str]]:
    max_time = max(1, int(timeout))
    command = [
        "curl",
        "-fsSL",
        "-H",
        "Accept: application/vnd.github+json",
        "-H",
        "User-Agent: DeepFunGen-App",
        "--max-time",
        str(max_time),
        _GITHUB_RELEASE_URL,
    ]
    logger.debug(
        "Invoking curl for latest release",
        extra={"command": " ".join(command), "timeout": timeout},
    )
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:  # pragma: no cover - depends on system
        raise _UpdateCheckError("curl executable is not available on this system") from exc
    except CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        raise _UpdateCheckError(f"curl failed with exit code {exc.returncode}: {stderr}") from exc
    except Exception as exc:  # pragma: no cover - safety net
        raise _UpdateCheckError(f"curl invocation failed: {exc}") from exc

    payload_text = (result.stdout or "").strip()
    if not payload_text:
        raise _UpdateCheckError("curl returned an empty response from GitHub")
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError as exc:
        raise _UpdateCheckError("Failed to decode GitHub release response") from exc

    tag = payload.get("tag_name") or payload.get("name")
    html_url = payload.get("html_url")
    published_at = payload.get("published_at")
    logger.debug(
        "Received release payload",
        extra={
            "latest_version": tag,
            "published_at": published_at,
        },
    )
    return {
        "latest_version": tag or None,
        "latest_url": html_url or None,
        "published_at": published_at,
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


def _is_vr_model(record: JobRecord) -> bool:
    if getattr(record, "use_vr_focus_crop", False):
        return True
    model_path = getattr(record, "model_path", None)
    if not model_path:
        return False
    name = Path(model_path).name.lower()
    return "vr" in name


def _extract_video_frame(
    video_path: Path,
    timestamp_ms: float,
    max_width: int = 480,
    *,
    use_vr_left_half: bool = False,
) -> bytes:
    return decode_preview_frame(
        video_path,
        timestamp_ms,
        max_width,
        use_vr_left_half=use_vr_left_half,
    )



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
    use_vr_left = _is_vr_model(record)
    preview_width = 360 if use_vr_left else 480
    try:
        data = await run_in_threadpool(
            _extract_video_frame,
            video_path,
            timestamp_ms,
            preview_width,
            use_vr_left_half=use_vr_left,
        )
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


@router.post("/results/{job_id}/postprocess", response_model=JobDetail)
async def reprocess_job(
    job_id: str,
    payload: PostprocessOptionsModel,
    request: Request,
    store: JobStore = Depends(_get_store),
    logs: LogBroker = Depends(_get_logs),
) -> JobDetail:
    record = store.get(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if record.status == JobStatus.PROCESSING:
        raise HTTPException(status_code=409, detail="Job is currently processing")
    if not record.prediction_path:
        raise HTTPException(status_code=404, detail="Prediction file missing")

    csv_path = Path(record.prediction_path)
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail="Prediction file missing")

    previous_status = record.status
    previous_message = record.message
    previous_options = record.postprocess_options.copy(deep=True) if record.postprocess_options else None
    previous_completed_at = record.completed_at
    previous_started_at = record.started_at
    previous_progress = record.progress

    target_options = payload.copy(deep=True)
    script_path = Path(record.script_path) if record.script_path else resolve_script_path(Path(record.video_path))
    model_name = Path(record.model_path).stem if record.model_path else "model"
    frame_rate = record.frame_rate or 30.0

    record.apply_status(JobStatus.PROCESSING, message="Re-processing outputs")
    record.update_progress(0.98, message="Re-processing outputs")
    record.error = None
    _log_job_event(logs, record, "Re-processing predictions with updated options")
    _persist(request)

    def _run_postprocess() -> None:
        df = pd.read_csv(csv_path)
        processed = run_postprocess(df.copy(), target_options, frame_rate)
        write_funscript(processed, script_path, model_name, target_options)

    try:
        await run_in_threadpool(_run_postprocess)
    except Exception as exc:  # pragma: no cover - defensive
        record.postprocess_options = previous_options
        record.apply_status(previous_status, message=f"Re-process failed: {exc}")
        record.completed_at = previous_completed_at
        record.started_at = previous_started_at
        record.update_progress(previous_progress)
        record.error = str(exc)
        _log_job_event(logs, record, f"Re-process failed: {exc}", level="ERROR")
        _persist(request)
        raise HTTPException(status_code=500, detail=f"Failed to re-process results: {exc}") from exc
    else:
        record.postprocess_options = target_options
        record.apply_status(JobStatus.COMPLETED, message="Outputs updated")
        record.update_progress(1.0)
        record.error = None
        record.script_path = str(script_path)
        _log_job_event(logs, record, "Outputs regenerated with new options")
    finally:
        _persist(request)

    return record.to_detail()



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
    models.sort(key=lambda item: "vr" in item["display_name"].lower())
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


@router.get("/system/info")
async def system_info(request: Request) -> JSONResponse:
    info = getattr(request.app.state, "version_info", None)
    if not isinstance(info, dict):
        info = get_version_info()
    return JSONResponse(info)


# ---------------------------------------------------------------------------
# Update checks
# ---------------------------------------------------------------------------


@router.get("/system/update")
async def system_update(request: Request) -> JSONResponse:
    app = request.app
    current_version = getattr(app, "version", None)
    cache = getattr(app.state, "update_cache", None)
    now = time.time()
    data: dict[str, Optional[str]]
    ttl = None
    if isinstance(cache, dict):
        ttl = cache.get("ttl")
    if ttl is None:
        ttl = _UPDATE_CACHE_TTL
    cache_hit = (
        isinstance(cache, dict)
        and isinstance(cache.get("timestamp"), (int, float))
        and (now - float(cache["timestamp"])) < float(ttl)
    )
    logger.info(
        "Update check requested",
        extra={
            "current_version": current_version,
            "cache_hit": cache_hit,
            "cached_error": (cache or {}).get("data", {}).get("error") if isinstance(cache, dict) else None,
        },
    )
    if cache_hit:
        data = cache.get("data", {}) or {}
    else:
        logger.info("Update cache stale; fetching latest release from GitHub")
        try:
            data = await run_in_threadpool(_fetch_latest_release)
            app.state.update_cache = {"timestamp": now, "ttl": _UPDATE_CACHE_TTL, "data": data}
            logger.info(
                "Latest release fetched",
                extra={
                    "latest_version": data.get("latest_version") if isinstance(data, dict) else None,
                    "published_at": data.get("published_at") if isinstance(data, dict) else None,
                },
            )
        except _UpdateCheckError as exc:
            data = {"error": str(exc)}
            app.state.update_cache = {"timestamp": now, "ttl": _UPDATE_ERROR_TTL, "data": data}
            logger.error(
                "Update check failed; cached error for retry",
                exc_info=True,
            )

    latest_version = data.get("latest_version") if isinstance(data, dict) else None
    latest_url = data.get("latest_url") if isinstance(data, dict) else None
    published_at = data.get("published_at") if isinstance(data, dict) else None
    error = data.get("error") if isinstance(data, dict) else None

    current_tuple = _version_tuple(current_version)
    latest_tuple = _version_tuple(latest_version)

    has_update = False
    up_to_date = False
    if latest_version and latest_tuple:
        if current_tuple and latest_tuple > current_tuple:
            has_update = True
        elif current_tuple and latest_tuple <= current_tuple:
            up_to_date = True

    response = {
        "current_version": current_version,
        "latest_version": latest_version,
        "latest_url": latest_url,
        "published_at": published_at,
        "has_update": has_update,
        "up_to_date": up_to_date if latest_version else False,
        "error": error,
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }
    logger.info(
        "Update check completed",
        extra={
            "has_update": has_update,
            "up_to_date": response["up_to_date"],
            "error": error,
            "latest_version": latest_version,
        },
    )
    return JSONResponse(response)


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


@router.get("/logs/history")
async def logs_history(logs: LogBroker = Depends(_get_logs)) -> JSONResponse:
    events = jsonable_encoder(logs.history())
    return JSONResponse(events)


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
