"""In-memory job bookkeeping and persistence glue."""
from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .models import JobDetail, JobStatus, JobSummary, PostprocessOptionsModel


@dataclass
class JobRecord:
    """Mutable representation of a queued or processed video job."""

    video_path: str
    model_path: Optional[str] = None
    output_directory: Optional[str] = None
    postprocess_options: Optional[PostprocessOptionsModel] = None
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: JobStatus = JobStatus.PENDING
    message: str = "Queued"
    progress: float = 0.0
    frame_count: Optional[int] = None
    frame_rate: Optional[float] = None
    prediction_path: Optional[str] = None
    script_path: Optional[str] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    def apply_status(self, status: JobStatus, *, message: Optional[str] = None) -> None:
        if status == self.status and message is None:
            return
        self.status = status
        if message is not None:
            self.message = message
        if status == JobStatus.PROCESSING and self.started_at is None:
            self.started_at = datetime.utcnow()
        if status in {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}:
            self.completed_at = datetime.utcnow()

    def update_progress(self, value: float, *, message: Optional[str] = None) -> None:
        self.progress = float(max(0.0, min(1.0, value)))
        if message is not None:
            self.message = message

    def to_summary(self) -> JobSummary:
        return JobSummary(
            id=self.id,
            video_path=self.video_path,
            video_name=Path(self.video_path).name,
            status=self.status,
            message=self.message,
            progress=self.progress,
            progress_percent=int(round(self.progress * 100)),
            created_at=self.created_at,
            started_at=self.started_at,
            completed_at=self.completed_at,
            model_path=self.model_path,
            prediction_path=self.prediction_path,
            script_path=self.script_path,
            postprocess_options=self.postprocess_options,
        )

    def to_detail(self) -> JobDetail:
        summary = self.to_summary()
        data = summary.dict()
        data.update(
            {
                "frame_count": self.frame_count,
                "frame_rate": self.frame_rate,
                "output_directory": self.output_directory,
                "error": self.error,
            }
        )
        return JobDetail(**data)


class JobStore:
    """Thread-safe registry of jobs plus queue ordering."""

    def __init__(self) -> None:
        self._jobs: Dict[str, JobRecord] = {}
        self._queue: List[str] = []
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Basic CRUD
    # ------------------------------------------------------------------
    def add_job(self, record: JobRecord) -> JobRecord:
        with self._lock:
            if record.id in self._jobs:
                raise ValueError(f"Duplicate job id: {record.id}")
            self._jobs[record.id] = record
            self._queue.append(record.id)
            return record

    def get(self, job_id: str) -> Optional[JobRecord]:
        with self._lock:
            return self._jobs.get(job_id)

    def all(self) -> List[JobRecord]:
        with self._lock:
            return [self._jobs[jid] for jid in self._queue if jid in self._jobs]

    def pending(self) -> List[JobRecord]:
        with self._lock:
            return [self._jobs[jid] for jid in self._queue if self._jobs[jid].status == JobStatus.PENDING]

    def remove(self, job_id: str) -> Optional[JobRecord]:
        with self._lock:
            record = self._jobs.pop(job_id, None)
            if record is not None and job_id in self._queue:
                self._queue.remove(job_id)
            return record

    def clear_finished(self) -> List[JobRecord]:
        with self._lock:
            finished = [jid for jid, job in self._jobs.items() if job.status in {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}]
            removed: List[JobRecord] = []
            for jid in finished:
                removed.append(self._jobs.pop(jid))
                if jid in self._queue:
                    self._queue.remove(jid)
            return removed

    def reorder(self, job_ids: Iterable[str]) -> None:
        with self._lock:
            ordered = [jid for jid in job_ids if jid in self._jobs]
            remaining = [jid for jid in self._queue if jid not in ordered]
            self._queue = ordered + remaining

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    def exists_for(
        self,
        video_path: str,
        model_path: Optional[str] = None,
        options: Optional[PostprocessOptionsModel] = None,
    ) -> bool:
        normalized_video = str(Path(video_path).resolve())
        normalized_model = str(Path(model_path).resolve()) if model_path else None

        def _options_signature(source: Optional[PostprocessOptionsModel]) -> Optional[tuple]:
            if source is None:
                return None
            return tuple(sorted(source.dict().items()))

        candidate_signature = (
            normalized_video,
            normalized_model,
            _options_signature(options),
        )
        with self._lock:
            for job in self._jobs.values():
                if job.status in {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}:
                    continue
                job_video = str(Path(job.video_path).resolve())
                if job_video != normalized_video:
                    continue
                job_model = str(Path(job.model_path).resolve()) if job.model_path else None
                job_signature = (
                    job_video,
                    job_model,
                    _options_signature(job.postprocess_options),
                )
                if job_signature == candidate_signature:
                    return True
        return False

    def summaries(self) -> List[JobSummary]:
        return [job.to_summary() for job in self.all()]

    def details(self) -> List[JobDetail]:
        return [job.to_detail() for job in self.all()]

    def counts(self) -> Dict[str, int]:
        with self._lock:
            total = len(self._jobs)
            pending = sum(1 for job in self._jobs.values() if job.status == JobStatus.PENDING)
            processing = sum(1 for job in self._jobs.values() if job.status == JobStatus.PROCESSING)
            completed = sum(1 for job in self._jobs.values() if job.status == JobStatus.COMPLETED)
        return {
            "total": total,
            "pending": pending,
            "processing": processing,
            "completed": completed,
        }


__all__ = ["JobRecord", "JobStore"]
