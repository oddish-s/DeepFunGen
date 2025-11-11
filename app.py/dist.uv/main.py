"""Desktop entry-point launching FastAPI + pywebview shell."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
import subprocess
import sys
import threading
from pathlib import Path
from typing import List, Optional
from urllib import request as urlrequest

import uvicorn
import webview
try:
    from webview.dom import DOMEventHandler
except Exception:  # pragma: no cover - older pywebview builds
    DOMEventHandler = None  # type: ignore

# Allow `backend` package imports and repository modules when running directly
APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent

sys.path.insert(0, str(APP_DIR))
sys.path.insert(0, str(REPO_ROOT))

from backend.app import create_app

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("deepfungen")
logger.setLevel(logging.DEBUG)


class DesktopBridge:
    """Exposes native dialogs to the frontend via pywebview."""

    def __init__(self) -> None:
        self._window: Optional[webview.Window] = None
        self._dom_handlers: List[tuple[object, str, DOMEventHandler]] = []

    @staticmethod
    def _dialog_mode(candidates: tuple[str, ...], legacy: str) -> Optional[object]:
        dialog_cls = getattr(webview, "FileDialog", None)
        if dialog_cls is not None:
            for name in candidates:
                if hasattr(dialog_cls, name):
                    return getattr(dialog_cls, name)
        return getattr(webview, legacy, None)

    def bind(self, window: webview.Window) -> None:
        self._window = window
        self._register_drag_drop()

    def select_files(self) -> List[str]:
        if self._window is None:
            return []
        mode = self._dialog_mode(("OPEN", "OPEN_FILE"), "OPEN_DIALOG")
        if mode is None:
            return []
        selection = self._window.create_file_dialog(
            mode,
            allow_multiple=True,
            file_types=(
                "Video Files (*.mp4;*.mov;*.m4v;*.avi;*.mkv;*.mpg;*.mpeg;*.wmv)",
                "All Files (*.*)",
            ),
        )
        return selection or []

    def select_folder(self) -> Optional[str]:
        if self._window is None:
            return None
        mode = self._dialog_mode(("FOLDER", "DIRECTORY", "OPEN_DIRECTORY"), "FOLDER_DIALOG")
        if mode is None:
            return None
        selection = self._window.create_file_dialog(mode)
        if selection:
            return selection[0]
        return None

    def _register_drag_drop(self) -> None:
        if self._window is None:
            return
        if DOMEventHandler is None:
            logger.debug("pywebview DOM support not available; drag/drop hook skipped")
            return
        try:
            document = self._window.dom.document  # type: ignore[attr-defined]
        except Exception:
            logger.debug("pywebview DOM API unavailable; drag/drop hook skipped", exc_info=True)
            return
        try:
            #document.events.dragover += DOMEventHandler(self._handle_dragover, True, True)
            document.events.drop += DOMEventHandler(self._handle_drop, True, True)


            #drag_handler = DOMEventHandler(self._handle_dragover)
            #drop_handler = DOMEventHandler(self._handle_drop)
            #targets: List[object] = []
            #body = getattr(document, "body", None)
            #if body is not None:
            #    targets.append(body)
            #targets.append(document)
            #for target in targets:
            #    events = getattr(target, "events", None)
            #    if events is None:
            #        continue
            #    events.dragover += drag_handler
            #    events.drop += drop_handler
            #    self._dom_handlers.append((target, "dragover", drag_handler))
            #    self._dom_handlers.append((target, "drop", drop_handler))
            #if self._dom_handlers:
            #    try:
            #        self._window.evaluate_js("window.__nativeDragSupported = true;")
            #    except Exception:
            #        logger.debug("Failed to flag native drag support", exc_info=True)
        except Exception:
            logger.debug("Failed to attach DOM drag/drop handlers", exc_info=True)

    def _handle_dragover(self, _event: dict | None) -> dict:
        return {'preventDefault': True, 'stopPropagation': True}

    def _handle_drop(self, event: dict | None) -> dict:
        if self._window is None or not event:
            return {'preventDefault': True, 'stopPropagation': True}
        raw_event: Optional[dict] = None
        if isinstance(event, dict):
            raw_event = event
        else:
            try:
                raw_event = json.loads(event) if isinstance(event, str) else None
            except Exception:
                raw_event = None
        files: List[object] = []
        if isinstance(raw_event, dict):
            data_transfer = raw_event.get('dataTransfer') or {}
            if isinstance(data_transfer, dict):
                for key in ('files', 'items'):
                    entries = data_transfer.get(key)
                    if isinstance(entries, list):
                        files.extend(entries)
                for key in ('filePaths', 'paths'):
                    extra_paths = data_transfer.get(key)
                    if isinstance(extra_paths, list):
                        for entry in extra_paths:
                            if isinstance(entry, str):
                                files.append({'path': entry})

        paths: List[str] = []
        for file in files or []:
            candidate = None
            if isinstance(file, dict):
                candidate = (
                    file.get('pywebviewFullPath')
                    or file.get('path')
                    or file.get('fullPath')
                    or file.get('absolutePath')
                    or file.get('filePath')
                    or file.get('value')
                )
                if candidate is None:
                    inner = file.get('file')
                    if isinstance(inner, dict):
                        candidate = inner.get('path') or inner.get('name')
            if isinstance(candidate, str) and candidate:
                paths.append(candidate)
        if not paths and isinstance(raw_event, dict):
            fallback = raw_event.get('files')
            if isinstance(fallback, list):
                for entry in fallback:
                    if isinstance(entry, str):
                        paths.append(entry)
        if paths:
            try:
                payload = json.dumps(paths)
                self._window.evaluate_js(
                    f"window.dispatchEvent(new CustomEvent('native:files-dropped', {{ detail: {payload} }}));"
                )
            except Exception:
                logger.debug("Failed to forward native drop event to frontend", exc_info=True)
        return {'preventDefault': True, 'stopPropagation': True}

    def open_path(self, target: Optional[str]) -> bool:
        if not target:
            return False
        path = Path(target).expanduser()
        if not path.exists():
            return False
        try:
            if sys.platform.startswith('win'):
                os.startfile(str(path))  # type: ignore[attr-defined]
            elif sys.platform == 'darwin':
                subprocess.Popen(['open', str(path)])
            else:
                subprocess.Popen(['xdg-open', str(path)])
            return True
        except Exception:
            return False


def find_available_port(host: str, starting_port: int, attempts: int = 20) -> int:
    """Find an available TCP port, incrementing when the preferred one is busy."""

    port = starting_port
    for _ in range(attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((host, port))
            except OSError:
                port += 1
                continue
        return port
    raise RuntimeError(f"No free port found starting from {starting_port}")


def _start_api_server(host: str, port: int) -> None:
    app = create_app()
    config = uvicorn.Config(
        app=app,
        host=host,
        port=port,
        log_level="debug",
        access_log=False,
    )
    server = uvicorn.Server(config)
    app.state.uvicorn_server = server

    loop: asyncio.AbstractEventLoop
    if hasattr(config, "get_loop_factory"):
        factory = config.get_loop_factory()
        if callable(factory):
            loop = factory()
        else:
            loop = asyncio.new_event_loop()
    else:
        loop = config.setup_event_loop()
    try:
        current_loop = asyncio.get_event_loop()
    except RuntimeError:
        current_loop = None
    if current_loop is not loop:
        asyncio.set_event_loop(loop)

    default_handler = loop.get_exception_handler()

    def quiet_connection_reset(loop_obj: asyncio.AbstractEventLoop, context: dict) -> None:
        exc = context.get("exception")
        message = context.get("message", "")
        if isinstance(exc, ConnectionResetError) or (
            isinstance(message, str) and "ConnectionResetError" in message
        ):
            return
        if default_handler is not None:
            default_handler(loop_obj, context)
        else:
            loop_obj.default_exception_handler(context)

    loop.set_exception_handler(quiet_connection_reset)
    loop.run_until_complete(server.serve())
    loop.close()


def main() -> None:
    host = "127.0.0.1"
    preferred_port = 5050
    port = find_available_port(host, preferred_port)
    if port != preferred_port:
        logger.warning("Port %s busy, using %s", preferred_port, port)
    title = "DeepFunGen"
    width = 1680
    height = 960

    api_thread = threading.Thread(
        target=_start_api_server,
        args=(host, port),
        daemon=True,
        name="uvicorn-server",
    )
    api_thread.start()
    logger.info("API server starting on http://%s:%s", host, port)

    bridge = DesktopBridge()
    window = webview.create_window(
        title=title,
        url=f"http://{host}:{port}",
        width=width,
        height=height,
        resizable=True,
        min_size=(1280, 720),
        js_api=bridge,
    )

    def _on_loaded() -> None:
        bridge.bind(window)
    window.events.loaded += _on_loaded

    def _trigger_shutdown(*_args: object) -> None:
        shutdown_url = f"http://{host}:{port}/api/system/shutdown"
        try:
            urlrequest.urlopen(urlrequest.Request(shutdown_url, data=b"", method="POST"), timeout=2)
        except Exception:  # pragma: no cover - best effort shutdown
            logger.debug("Shutdown request failed", exc_info=True)

    if hasattr(window, "events"):
        try:
            window.events.closing += _trigger_shutdown
        except AttributeError:
            pass
        try:
            window.events.closed += _trigger_shutdown
        except AttributeError:
            pass

    webview.start()
    _trigger_shutdown()
    api_thread.join(timeout=5)
    logger.info("DeepFunGen window closed")


if __name__ == "__main__":
    if "--cli" in sys.argv:
        idx = sys.argv.index("--cli")
        cli_args = sys.argv[idx + 1 :]
        # Strip flag before handing over to CLI to avoid confusing argparse later
        sys.argv = sys.argv[:idx]
        from cli import main as cli_main

        sys.exit(cli_main(cli_args))
    main()
