"""Shared Pydantic models for the FastAPI layer."""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    """Enumeration of processing states for a queue item."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class PostprocessOptionsModel(BaseModel):
    """User-configurable post-processing knobs."""

    smooth_window_frames: int = Field(3, ge=1, le=512)
    prominence_ratio: float = Field(0.1, ge=0.0, le=1.0)
    min_prominence: float = Field(0.0, ge=0.0)
    max_slope: float = Field(10.0, ge=0.0)
    boost_slope: float = Field(7.0, ge=0.0)
    min_slope: float = Field(2.0, ge=0.0)
    merge_threshold_ms: float = Field(120.0, ge=0.0)
    central_deviation_threshold: float = Field(0.03, ge=0.0)
    fft_denoise: bool = True
    fft_frames_per_component: int = Field(10, ge=1, le=10_000)
    fft_window_frames: Optional[int] = Field(None, ge=1, le=10_000)


class AddJobsRequest(BaseModel):
    """Payload for enqueuing one or more videos."""

    video_paths: List[str] = Field(default_factory=list)
    model_path: Optional[str] = None
    output_directory: Optional[str] = None
    postprocess_options: Optional[PostprocessOptionsModel] = None
    recursive: bool = False



class JobSummary(BaseModel):
    """Minimal projection used by queue and results lists."""

    id: str
    video_path: str
    video_name: str
    status: JobStatus
    message: str = ""
    progress: float = 0.0
    progress_percent: int = 0
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    model_path: Optional[str] = None
    prediction_path: Optional[str] = None
    script_path: Optional[str] = None
    postprocess_options: Optional[PostprocessOptionsModel] = None


class JobDetail(JobSummary):
    """Extended representation returned when inspecting a job."""

    frame_count: Optional[int] = None
    frame_rate: Optional[float] = None
    output_directory: Optional[str] = None
    error: Optional[str] = None


class SettingsModel(BaseModel):
    """Application-level preferences persisted across sessions."""

    theme: str = Field("dark", pattern=r"^(dark|light)$")
    language: str = Field("ko", pattern=r"^(ko|en)$")
    default_model_path: Optional[str] = None
    default_postprocess: Optional[PostprocessOptionsModel] = None


class SystemStatusModel(BaseModel):
    """Snapshot for the footer status bar."""

    gpu_usage: Optional[float] = None
    cpu_usage: float = 0.0
    queue_total: int = 0
    queue_pending: int = 0
    queue_processing: int = 0
    queue_completed: int = 0
    execution_provider: Optional[str] = None


class LogEvent(BaseModel):
    """Structure streamed to the log viewer via SSE."""

    timestamp: datetime
    job_id: Optional[str]
    job_name: Optional[str]
    level: str = "INFO"
    message: str


__all__ = [
    "JobStatus",
    "PostprocessOptionsModel",
    "AddJobsRequest",
    "JobSummary",
    "JobDetail",
    "SettingsModel",
    "SystemStatusModel",
    "LogEvent",
]
