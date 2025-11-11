"""Simple JSON persistence for job metadata and app settings."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from .config import STATE_FILE
from .jobs import JobRecord, JobStore
from .models import JobStatus, PostprocessOptionsModel, SettingsModel


class StateRepository:
    """Handles serialising job state to disk between runs."""

    def __init__(self, path: Path = STATE_FILE) -> None:
        self.path = path

    # ------------------------------------------------------------------
    def load_into(self, store: JobStore, settings_manager: "SettingsManager" | None = None) -> None:
        if not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return
        jobs = payload.get("jobs", []) if isinstance(payload, dict) else []
        for raw in jobs:
            record = self._from_dict(raw)
            store.add_job(record)
        if settings_manager is not None:
            raw_settings = payload.get("settings") if isinstance(payload, dict) else None
            if isinstance(raw_settings, dict):
                try:
                    settings_manager.replace(SettingsModel(**raw_settings))
                except Exception:
                    pass

    def dump_from(self, store: JobStore, settings: SettingsModel | None = None) -> None:
        serialised = {
            "jobs": [self._to_dict(job) for job in store.all()],
            "settings": settings.dict() if settings else None,
        }
        self.path.write_text(json.dumps(serialised, indent=2, default=str), encoding="utf-8")

    # ------------------------------------------------------------------
    @staticmethod
    def _to_dict(job: JobRecord) -> Dict[str, Any]:
        return {
            "id": job.id,
            "video_path": job.video_path,
            "model_path": job.model_path,
            "output_directory": job.output_directory,
            "status": job.status.value,
            "message": job.message,
            "progress": job.progress,
            "frame_count": job.frame_count,
            "frame_rate": job.frame_rate,
            "prediction_path": job.prediction_path,
            "script_path": job.script_path,
            "error": job.error,
            "created_at": job.created_at.isoformat(),
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            "postprocess_options": job.postprocess_options.dict() if job.postprocess_options else None,
            "frames_total": job.frames_total,
            "frames_preprocessed": job.frames_preprocessed,
            "frames_inferred": job.frames_inferred,
            "model_input_height": job.model_input_height,
            "model_input_width": job.model_input_width,
            "use_vr_focus_crop": job.use_vr_focus_crop,
        }

    @staticmethod
    def _from_dict(raw: Dict[str, Any]) -> JobRecord:
        options = raw.get("postprocess_options")
        postprocess = None
        if isinstance(options, dict):
            try:
                postprocess = PostprocessOptionsModel(**options)
            except Exception:
                postprocess = None
        record = JobRecord(
            video_path=raw.get("video_path", ""),
            model_path=raw.get("model_path"),
            output_directory=raw.get("output_directory"),
            postprocess_options=postprocess,
            id=raw.get("id", ""),
        )
        status_value = raw.get("status")
        if isinstance(status_value, str):
            try:
                record.status = JobStatus(status_value)
            except ValueError:
                record.status = JobStatus.PENDING
        record.message = raw.get("message", record.message)
        record.progress = float(raw.get("progress", record.progress))
        record.frame_count = raw.get("frame_count")
        record.frame_rate = raw.get("frame_rate")
        record.prediction_path = raw.get("prediction_path")
        record.script_path = raw.get("script_path")
        record.error = raw.get("error")
        for key, attr in (
            ("model_input_height", "model_input_height"),
            ("model_input_width", "model_input_width"),
        ):
            value = raw.get(key, getattr(record, attr))
            try:
                setattr(record, attr, int(value) if value is not None else None)
            except (TypeError, ValueError):
                setattr(record, attr, getattr(record, attr))
        crop_flag = raw.get("use_vr_focus_crop", record.use_vr_focus_crop)
        if isinstance(crop_flag, bool):
            record.use_vr_focus_crop = crop_flag
        elif isinstance(crop_flag, (int, float)):
            record.use_vr_focus_crop = bool(int(crop_flag))
        elif isinstance(crop_flag, str):
            record.use_vr_focus_crop = crop_flag.strip().lower() in {"1", "true", "yes", "on"}
        else:
            record.use_vr_focus_crop = bool(crop_flag)
        total = raw.get("frames_total", record.frames_total)
        try:
            record.frames_total = int(total) if total is not None else record.frames_total
        except (TypeError, ValueError):
            pass
        for key, attr in (("frames_preprocessed", "frames_preprocessed"), ("frames_inferred", "frames_inferred")):
            value = raw.get(key, getattr(record, attr))
            try:
                setattr(record, attr, int(value))
            except (TypeError, ValueError):
                pass
        record.created_at = _parse_datetime(raw.get("created_at")) or record.created_at
        record.started_at = _parse_datetime(raw.get("started_at"))
        record.completed_at = _parse_datetime(raw.get("completed_at"))
        return record


def _parse_datetime(value: Any) -> Any:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except Exception:
        return None


class SettingsManager:
    """Thread-safe holder for application settings."""

    def __init__(self) -> None:
        from threading import RLock

        self._lock = RLock()
        self._settings = SettingsModel()

    def current(self) -> SettingsModel:
        with self._lock:
            return SettingsModel(**self._settings.dict())

    def replace(self, settings: SettingsModel) -> None:
        with self._lock:
            self._settings = settings


__all__ = ["StateRepository", "SettingsManager"]
