"""FastAPI application factory."""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .config import FRONTEND_STATIC_DIR, FRONTEND_TEMPLATES_DIR
from .jobs import JobStore
from .logstream import LogBroker, LogRelayHandler
from .routes import router
from .storage import SettingsManager, StateRepository
from .telemetry import TelemetryCollector
from .worker import ProcessingWorker


def create_app() -> FastAPI:
    """Instantiate the FastAPI application and wire dependencies."""

    app = FastAPI(title="DeepFunGen", version="1.0.0")

    templates = Jinja2Templates(directory=str(FRONTEND_TEMPLATES_DIR))
    app.mount("/static", StaticFiles(directory=str(FRONTEND_STATIC_DIR)), name="static")

    job_store = JobStore()
    log_broker = LogBroker()
    settings_manager = SettingsManager()
    repository = StateRepository()
    repository.load_into(job_store, settings_manager)
    telemetry = TelemetryCollector()
    worker = ProcessingWorker(job_store, log_broker, settings_manager, repository)

    def _on_model_change(path: Path, provider: str) -> None:
        app.state.current_model_path = str(path)
        app.state.execution_provider = provider

    worker.set_model_change_callback(_on_model_change)

    relay_handler = LogRelayHandler(log_broker)
    relay_handler.setLevel(logging.DEBUG)
    relay_handler.setFormatter(logging.Formatter("%(message)s"))
    attached_loggers = []
    for logger_name in (None, "uvicorn", "uvicorn.error", "uvicorn.access"):
        logger_obj = logging.getLogger(logger_name) if logger_name else logging.getLogger()
        if logger_obj.level > logging.DEBUG or logger_obj.level == logging.NOTSET:
            logger_obj.setLevel(logging.DEBUG)
        if not any(isinstance(handler, LogRelayHandler) for handler in logger_obj.handlers):
            logger_obj.addHandler(relay_handler)
            attached_loggers.append(logger_obj)

    app.state.job_store = job_store
    app.state.log_broker = log_broker
    app.state.settings_manager = settings_manager
    app.state.state_repo = repository
    app.state.telemetry = telemetry
    app.state.worker = worker
    app.state.log_handler = relay_handler
    app.state.log_handler_loggers = attached_loggers
    app.state.current_model_path = None
    app.state.execution_provider = None
    app.state.templates = templates

    @app.on_event("startup")
    async def _startup() -> None:
        log_broker.bind_loop(asyncio.get_running_loop())
        await worker.start()

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        await worker.stop(cancel_running=True, reason="Shutting down")
        telemetry.shutdown()
        repository.dump_from(job_store, settings_manager.current())
        handler = getattr(app.state, "log_handler", None)
        attached = getattr(app.state, "log_handler_loggers", []) or []
        for logger_obj in attached:
            if handler in logger_obj.handlers:
                logger_obj.removeHandler(handler)

    @app.get("/", response_class=HTMLResponse)
    async def root(request: Request):
        return templates.TemplateResponse("base.html", {"request": request})

    app.include_router(router)

    return app


__all__ = ["create_app"]

