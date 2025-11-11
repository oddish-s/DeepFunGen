"""FastAPI application factory."""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .bootstrap import bootstrap_backend
from .config import FRONTEND_STATIC_DIR, FRONTEND_TEMPLATES_DIR
from .jobs import JobStore
from .logstream import LogBroker, LogRelayHandler
from .routes import router
from .storage import SettingsManager, StateRepository
from .telemetry import TelemetryCollector
from .worker import ProcessingWorker
from .version_info import APP_NAME, APP_VERSION, get_version_info


def create_app() -> FastAPI:
    """Instantiate the FastAPI application and wire dependencies."""

    metadata = get_version_info()
    app = FastAPI(title=APP_NAME, version=APP_VERSION)

    templates = Jinja2Templates(directory=str(FRONTEND_TEMPLATES_DIR))
    app.mount("/static", StaticFiles(directory=str(FRONTEND_STATIC_DIR)), name="static")

    services = bootstrap_backend(load_state=True)
    job_store = services.job_store
    log_broker = services.log_broker
    settings_manager = services.settings_manager
    repository = services.state_repository
    telemetry = services.telemetry
    worker = services.worker

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
    app.state.version_info = metadata

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
