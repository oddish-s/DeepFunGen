"""Command-line entry point for DeepFunGen inference."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set, Tuple

from backend.bootstrap import bootstrap_backend
from backend.config import MODEL_SEARCH_PATHS
from backend.jobs import JobRecord
from backend.models import JobStatus
from backend.routes import SUPPORTED_VIDEO_EXTENSIONS


logger = logging.getLogger("deepfungen.cli")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run DeepFunGen inference without the desktop UI.")
    parser.add_argument(
        "inputs",
        nargs="*",
        help="Video files or directories to process.",
    )
    parser.add_argument(
        "--model",
        help="Override the default ONNX model path.",
    )
    parser.add_argument(
        "--output-dir",
        help="Destination directory for generated outputs (defaults to video folder).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recurse into subdirectories when a folder is provided.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List discovered ONNX models and exit.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    return parser


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def _iter_video_candidates(path: Path, recursive: bool) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    if not path.is_dir():
        return
    iterator = path.rglob("*") if recursive else path.iterdir()
    for candidate in iterator:
        if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS:
            yield candidate


def _resolve_inputs(raw_inputs: Sequence[str], recursive: bool) -> List[Path]:
    resolved: List[Path] = []
    seen: Set[Path] = set()
    for raw in raw_inputs:
        path = Path(raw).expanduser()
        if not path.exists():
            logger.warning("Input %s does not exist; skipping.", raw)
            continue
        for candidate in _iter_video_candidates(path.resolve(), recursive):
            try:
                resolved_path = candidate.resolve()
            except OSError:
                resolved_path = candidate
            if resolved_path in seen:
                continue
            seen.add(resolved_path)
            resolved.append(resolved_path)
    return resolved


def _normalise_path(path: Path) -> Path:
    try:
        return path.expanduser().resolve()
    except OSError:
        return path.expanduser()


def _discover_models() -> List[Path]:
    seen: Set[str] = set()
    models: List[Path] = []
    for root in MODEL_SEARCH_PATHS:
        path = Path(root)
        if not path.exists() or not path.is_dir():
            continue
        for candidate in path.rglob("*.onnx"):
            resolved = _normalise_path(candidate)
            key = str(resolved).lower()
            if key in seen:
                continue
            seen.add(key)
            models.append(resolved)
    models.sort(key=lambda p: p.stem.lower(), reverse=True)
    models.sort(key=lambda p: "vr" in p.stem.lower())
    return models


def _list_models(default_model: Optional[str]) -> int:
    resolved_default = None
    if default_model:
        resolved_default = _normalise_path(Path(default_model))
    print("Known model search paths:")
    for root in MODEL_SEARCH_PATHS:
        print(f"  - {root}")
    models = _discover_models()
    if not models:
        print("\nNo ONNX models found.")
        return 0
    print("\nDiscovered models (VR models listed last):")
    for idx, candidate in enumerate(models, start=1):
        marker = ""
        if resolved_default is not None and _normalise_path(candidate) == resolved_default:
            marker = " (default)"
        print(f"  [{idx}] {candidate}{marker}")
    return 0


def _ensure_model_path(
    provided: Optional[str],
    default_model: Optional[str],
    available_models: Sequence[Path],
) -> Tuple[Optional[Path], bool]:
    if provided:
        candidate = Path(provided).expanduser()
        if not candidate.exists():
            raise FileNotFoundError(f"Model path not found: {candidate}")
        return (_normalise_path(candidate), False)
    if default_model:
        candidate = Path(default_model).expanduser()
        if candidate.exists():
            return (_normalise_path(candidate), False)
    if available_models:
        return (_normalise_path(available_models[0]), True)
    # Defer to runtime resolution; worker will search MODEL_SEARCH_PATHS
    return (None, False)


def _summarise_job(job: JobRecord) -> str:
    parts = [f"{Path(job.video_path).name}"]
    if job.status == JobStatus.COMPLETED:
        if job.prediction_path:
            parts.append(f"csv={job.prediction_path}")
        if job.script_path:
            parts.append(f"funscript={job.script_path}")
    elif job.error:
        parts.append(f"error={job.error}")
    return " | ".join(parts)


def run_cli(args: argparse.Namespace) -> int:
    _configure_logging(args.verbose)
    services = bootstrap_backend(load_state=True)
    settings = services.settings_manager.current()

    available_models = _discover_models()

    if args.list_models:
        return _list_models(settings.default_model_path)

    video_paths = _resolve_inputs(args.inputs, args.recursive)
    if not video_paths:
        logger.error("No valid video inputs found.")
        return 1

    try:
        model_path, auto_selected = _ensure_model_path(
            args.model,
            settings.default_model_path,
            available_models,
        )
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    if model_path is None:
        logger.info("No explicit model configured; worker will search default locations.")
    elif auto_selected:
        logger.info("Using %s as default model (sorted order).", model_path)

    if settings.default_postprocess:
        logger.info("Using default postprocess settings from config.")

    output_directory: Optional[str] = None
    if args.output_dir:
        resolved_output = Path(args.output_dir).expanduser()
        resolved_output.mkdir(parents=True, exist_ok=True)
        output_directory = str(resolved_output.resolve())
        logger.info("Outputs will be written to %s", output_directory)

    exit_code = 0
    try:
        for index, video_path in enumerate(video_paths, start=1):
            logger.info("[%s/%s] Processing %s", index, len(video_paths), video_path)
            job = JobRecord(
                video_path=str(video_path),
                model_path=str(model_path) if model_path else args.model,
                output_directory=output_directory,
                postprocess_options=settings.default_postprocess,
            )
            services.worker.run_job_sync(job)
            if job.status == JobStatus.COMPLETED:
                logger.info("✅ %s", _summarise_job(job))
            else:
                logger.error("❌ %s (status: %s)", _summarise_job(job), job.status.value)
                if exit_code == 0:
                    exit_code = 2
    except KeyboardInterrupt:
        logger.warning("Interrupted by user.")
        services.worker.cancel_active_jobs("Cancelled by CLI interrupt")
        exit_code = 130
    finally:
        services.telemetry.shutdown()
        # Persist latest state to capture completed jobs/settings updates
        services.state_repository.dump_from(
            services.job_store,
            services.settings_manager.current(),
        )

    return exit_code


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run_cli(args)


if __name__ == "__main__":
    sys.exit(main())
