"""Reusable backend bootstrap helpers for GUI and CLI entry-points."""
from __future__ import annotations

from dataclasses import dataclass

from .jobs import JobStore
from .logstream import LogBroker
from .storage import SettingsManager, StateRepository
from .telemetry import TelemetryCollector
from .worker import ProcessingWorker


@dataclass
class BackendServices:
    """Container bundling the core runtime services."""

    job_store: JobStore
    log_broker: LogBroker
    settings_manager: SettingsManager
    state_repository: StateRepository
    telemetry: TelemetryCollector
    worker: ProcessingWorker


def bootstrap_backend(*, load_state: bool = True) -> BackendServices:
    """Initialise shared backend services.

    Parameters
    ----------
    load_state:
        When true, hydrate the job store and settings from the persisted state file.
    """

    job_store = JobStore()
    log_broker = LogBroker()
    settings_manager = SettingsManager()
    state_repository = StateRepository()
    telemetry = TelemetryCollector()
    worker = ProcessingWorker(job_store, log_broker, settings_manager, state_repository)

    if load_state:
        state_repository.load_into(job_store, settings_manager)

    return BackendServices(
        job_store=job_store,
        log_broker=log_broker,
        settings_manager=settings_manager,
        state_repository=state_repository,
        telemetry=telemetry,
        worker=worker,
    )


__all__ = ["BackendServices", "bootstrap_backend"]

