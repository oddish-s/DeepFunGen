"""Helper to launch multiple training runs for hyper-parameter sweeps."""
from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import yaml


KEY_ALIASES = {
    "optimizer_config.lr": "lr",
    "optimizer_config.weight_decay": "wd",
    "model_config.sequence_encoder.dropout": "seqdrop",
    "model_config.head.dropout": "headdrop",
    "training_config.scheduler_config.t_max": "tmax",
    "training_config.grad_accum_steps": "accum",
    "data_config.horizontal_flip": "flip",
}


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _write_yaml(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=False)


def _set_by_dotted(mapping: Dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    current: Dict[str, Any] = mapping
    for part in parts[:-1]:
        if part not in current or not isinstance(current[part], dict):
            current[part] = {}
        current = current[part]
    current[parts[-1]] = value


def _sanitize_fragment(fragment: str) -> str:
    return (
        fragment.replace("/", "-")
        .replace("\\", "-")
        .replace(" ", "_")
        .replace(":", "-")
    )


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4g}".replace(".", "p")
    if isinstance(value, (list, tuple)):
        return "_".join(_format_value(v) for v in value)
    return _sanitize_fragment(str(value))


def _alias_key(key: str) -> str:
    return KEY_ALIASES.get(key, key.split(".")[-1])


def _build_run_suffix(overrides: Mapping[str, Any], *, max_length: int = 80) -> str:
    parts = [f"{_alias_key(key)}={_format_value(overrides[key])}" for key in sorted(overrides)]
    suffix = "__".join(parts)
    if len(suffix) <= max_length:
        return suffix

    digest_source = json.dumps(overrides, sort_keys=True, default=str).encode("utf-8")
    digest = hashlib.md5(digest_source).hexdigest()[:10]
    compact = "__".join(f"{_alias_key(key)}" for key in sorted(overrides))
    base = f"{compact}__{digest}"
    return base[:max_length] if len(base) > max_length else base


def _cartesian_product(options: Mapping[str, Sequence[Any]]) -> Iterable[Dict[str, Any]]:
    keys = list(options.keys())
    value_lists = [options[k] for k in keys]
    for combination in itertools.product(*value_lists):
        yield {keys[i]: combination[i] for i in range(len(keys))}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a grid search over training configs")
    parser.add_argument("base_config", help="Path to base YAML config")
    parser.add_argument(
        "--grid",
        required=True,
        help="Path to YAML/JSON describing dotted keys and lists of values",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="configs/_sweeps",
        help="Directory to store generated config files",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Override --num-workers passed to train.py",
    )
    parser.add_argument(
        "--force-preprocess",
        action="store_true",
        help="Forward --force-preprocess to each training run",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned commands without executing them",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_config_path = Path(args.base_config)
    if not base_config_path.exists():
        raise FileNotFoundError(f"Base config not found: {base_config_path}")
    grid_path = Path(args.grid)
    if not grid_path.exists():
        raise FileNotFoundError(f"Grid file not found: {grid_path}")

    base_config = _load_yaml(base_config_path)
    base_experiment = None
    if isinstance(base_config, dict):
        mlflow_cfg = base_config.get("mlflow")
        if isinstance(mlflow_cfg, dict):
            experiment_raw = mlflow_cfg.get("experiment")
            if isinstance(experiment_raw, str) and experiment_raw.strip():
                base_experiment = experiment_raw.strip()
    with grid_path.open("r", encoding="utf-8") as handle:
        grid_data = json.load(handle) if grid_path.suffix.lower() == ".json" else yaml.safe_load(handle)

    if not isinstance(grid_data, dict):
        raise ValueError("Grid file must contain a mapping of dotted keys to value lists")

    options: Dict[str, List[Any]] = {}
    for key, values in grid_data.items():
        if not isinstance(values, Iterable) or isinstance(values, (str, bytes)):
            raise ValueError(f"Grid entry for '{key}' must be a list of values")
        options[key] = list(values)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    combinations = list(_cartesian_product(options))
    print(f"Planning {len(combinations)} runs from grid: {options}")

    for idx, overrides in enumerate(combinations, start=1):
        run_suffix = _build_run_suffix(overrides)
        generated_name = f"{base_config_path.stem}__{run_suffix}.yaml"
        generated_config_path = output_dir / generated_name

        config_copy = copy.deepcopy(base_config)
        for dotted_key, value in overrides.items():
            _set_by_dotted(config_copy, dotted_key, value)

        _write_yaml(generated_config_path, config_copy)

        cmd = [sys.executable, "scripts/train.py", str(generated_config_path), "--run-name", run_suffix]
        if args.num_workers is not None:
            cmd.extend(["--num-workers", str(args.num_workers)])
        if base_experiment:
            cmd.extend(["--experiment", base_experiment])
        if args.force_preprocess:
            cmd.append("--force-preprocess")

        print(f"[{idx}/{len(combinations)}] Running: {' '.join(cmd)}")
        if args.dry_run:
            continue

        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            print(f"[Warning] Run {run_suffix} exited with code {result.returncode}")


if __name__ == "__main__":
    main()
KEY_ALIASES = {
    "optimizer_config.lr": "lr",
    "optimizer_config.weight_decay": "wd",
    "model_config.sequence_encoder.dropout": "seqdrop",
    "model_config.head.dropout": "headdrop",
    "training_config.scheduler_config.t_max": "tmax",
    "training_config.grad_accum_steps": "accum",
    "data_config.horizontal_flip": "flip",
}
