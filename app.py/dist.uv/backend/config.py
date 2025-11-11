"""Configuration helpers for the DeepFunGen Python application."""
from __future__ import annotations

import os
import sys
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = APP_ROOT.parent

FRONTEND_DIR = APP_ROOT / "frontend"
FRONTEND_STATIC_DIR = FRONTEND_DIR / "static"
FRONTEND_TEMPLATES_DIR = FRONTEND_DIR / "templates"

APP_MODELS_DIR = APP_ROOT / "models"
APP_MODELS_DIR.mkdir(parents=True, exist_ok=True)


def _model_search_paths() -> list[Path]:
    candidates = [APP_MODELS_DIR]
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        candidates.append(exe_dir / "models")
        parent = os.environ.get("NUITKA_ONEFILE_PARENT")
        if parent:
            candidates.append(Path(parent) / "models")

    resolved: set[Path] = set()
    unique: list[Path] = []
    for candidate in candidates:
        try:
            key = candidate.resolve()
        except OSError:
            key = candidate
        if key in resolved:
            continue
        resolved.add(key)
        unique.append(candidate)
    return unique


# Directories where ONNX models may reside (mirrors WinForms search paths)
MODEL_SEARCH_PATHS = _model_search_paths()

STATE_DIR = APP_ROOT / "state"
STATE_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = STATE_DIR / "jobs-state.json"


__all__ = [
    "APP_ROOT",
    "REPO_ROOT",
    "APP_MODELS_DIR",
    "MODEL_SEARCH_PATHS",
    "STATE_DIR",
    "STATE_FILE",
    "FRONTEND_DIR",
    "FRONTEND_STATIC_DIR",
    "FRONTEND_TEMPLATES_DIR",
]
