"""Copy app assets into app.py/dist.uv for manual uv packaging."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


APP_DIR = Path(__file__).resolve().parent
DIST_DIR = APP_DIR / "dist.uv"
IGNORE_PATTERNS = shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo", ".DS_Store", "Thumbs.db")
DIRECTORIES = ("backend", "frontend", "models", "state")
FILES = ("main.py",)


def clean_dist() -> None:
    if DIST_DIR.exists():
        shutil.rmtree(DIST_DIR)


def ensure_dist_root() -> None:
    DIST_DIR.mkdir(parents=True, exist_ok=True)


def copy_directories() -> None:
    for directory in DIRECTORIES:
        src = APP_DIR / directory
        dst = DIST_DIR / directory
        if not src.exists():
            continue
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst, ignore=IGNORE_PATTERNS)


def copy_files() -> None:
    for filename in FILES:
        src = APP_DIR / filename
        if not src.exists():
            raise SystemExit(f"Required file missing: {src}")
        dst = DIST_DIR / filename
        shutil.copy2(src, dst)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Copy app directories into dist.uv for uv distribution")
    parser.add_argument("--clean", action="store_true", help="Remove existing dist.uv before assembling")
    args = parser.parse_args(argv)

    if args.clean:
        clean_dist()

    ensure_dist_root()
    copy_directories()
    copy_files()

    print(f"[build_uv] Copied backend/frontend/models/state into {DIST_DIR}")


if __name__ == "__main__":
    main()
