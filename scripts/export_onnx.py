"""Export trained FunTorch5 checkpoints to ONNX."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping, Tuple

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.config import load_config
from scripts.core.dataset import DEFAULT_NORMALIZATION
from scripts.core.model import create_model


def _resolve_config(config_arg: str) -> tuple[Path, str]:
    candidate = Path(config_arg)
    if candidate.suffix:
        if not candidate.exists():
            raise FileNotFoundError(f"Config file not found: {candidate}")
        return candidate, candidate.stem

    search_roots = [Path("configs"), Path(".")]
    suffixes = [".yaml", ".yml", ".json"]
    for root in search_roots:
        for suffix in suffixes:
            prospect = root / f"{config_arg}{suffix}"
            if prospect.exists():
                return prospect, prospect.stem
    raise FileNotFoundError(
        f"Could not resolve config '{config_arg}'. Provide a valid path or name under the configs/ directory."
    )


def _resolve_checkpoint(config: Mapping[str, Any], config_name: str, explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path

    training_cfg = config.get("training_config") if isinstance(config, Mapping) else None
    checkpoint_root = Path(training_cfg.get("checkpoint_dir", "models")) if training_cfg else Path("models")
    model_dir = checkpoint_root / config_name
    best = model_dir / "best_checkpoint.pth"
    latest = model_dir / "latest_checkpoint.pth"
    if best.exists():
        return best
    if latest.exists():
        return latest
    raise FileNotFoundError(
        f"Could not locate checkpoint for config '{config_name}'. Looked for {best} and {latest}."
    )


def _determine_input_shape(config: Mapping[str, Any], overrides: argparse.Namespace) -> Tuple[int, int, int, int, int]:
    data_cfg = config.get("data_config") or {}
    model_cfg = config.get("model_config") or {}

    seq_len = overrides.sequence_length if overrides.sequence_length is not None else int(
        data_cfg.get("sequence_length", 32)
    )

    height = overrides.height
    width = overrides.width
    if height is None or width is None:
        target_size = data_cfg.get("target_size", (224, 224))
        if isinstance(target_size, Mapping):
            if height is None:
                height = int(target_size.get("height", 224))
            if width is None:
                width = int(target_size.get("width", 224))
        else:
            try:
                if height is None:
                    height = int(target_size[0])
                if width is None:
                    width = int(target_size[1])
            except Exception as exc:  # pragma: no cover - defensive guard
                raise ValueError(f"Could not infer target_size from {target_size!r}") from exc
    if height is None or width is None:
        raise ValueError("Frame height/width could not be resolved; provide overrides explicitly")

    frame_cfg = model_cfg.get("frame_encoder", {}) if isinstance(model_cfg, Mapping) else {}
    in_channels = (
        overrides.channels if overrides.channels is not None else int(frame_cfg.get("in_channels", 3))
    )

    batch = overrides.batch_size if overrides.batch_size is not None else 1
    return int(batch), int(seq_len), int(in_channels), int(height), int(width)


def _resolve_normalization(config: Mapping[str, Any]) -> Tuple[float, float]:
    data_cfg = config.get("data_config") or {}
    train_cfg = config.get("training_config") or {}

    def _coerce_pair(candidate) -> Tuple[float, float] | None:
        if isinstance(candidate, Mapping):
            mean = candidate.get("mean")
            std = candidate.get("std")
            if mean is not None and std is not None:
                return float(mean), float(std)
        elif isinstance(candidate, (list, tuple)) and len(candidate) == 2:
            return float(candidate[0]), float(candidate[1])
        return None

    for source in (train_cfg.get("normalization"), data_cfg.get("normalization")):
        pair = _coerce_pair(source)
        if pair is not None:
            return pair
    return DEFAULT_NORMALIZATION


def _embed_metadata(
    onnx_path: Path,
    shape: Tuple[int, int, int, int, int],
    normalization: Tuple[float, float],
) -> None:
    try:
        import onnx
        from onnx import helper
    except ImportError:  # pragma: no cover - optional dependency
        print("Warning: python-onnx package not available; skipping metadata embedding.")
        return

    model = onnx.load(str(onnx_path))
    batch, seq, channels, height, width = shape
    mean, std = normalization
    props = {
        "funtorch5.schema": "1",
        "funtorch5.input_layout": "batch,sequence,channels,height,width",
        "funtorch5.input_batch": str(batch),
        "funtorch5.input_sequence": str(seq),
        "funtorch5.input_channels": str(channels),
        "funtorch5.input_height": str(height),
        "funtorch5.input_width": str(width),
        "funtorch5.normalization_mean": f"{mean:.9f}",
        "funtorch5.normalization_std": f"{std:.9f}",
    }
    try:
        helper.set_model_props(model, props)
        onnx.save(model, str(onnx_path))
    except Exception as exc:  # pragma: no cover - defensive
        print(f"Warning: failed to embed ONNX metadata ({exc}); continuing with raw export.")


def _load_model(config: Mapping[str, Any], checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    model_cfg = config.get("model_config")
    if not model_cfg:
        raise KeyError("Config missing 'model_config' section")
    model = create_model(model_cfg)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("model_state", checkpoint)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def _build_dynamic_axes(args: argparse.Namespace) -> dict[str, dict[int, str]] | None:
    axes: dict[str, dict[int, str]] = {}
    if args.dynamic_batch:
        axes.setdefault("input", {})[0] = "batch"
        axes.setdefault("output", {})[0] = "batch"
    if args.dynamic_sequence:
        axes.setdefault("input", {})[1] = "sequence"
    return axes or None


def export_to_onnx(args: argparse.Namespace) -> Path:
    config_path, config_name = _resolve_config(args.config)
    config = load_config(config_path)
    checkpoint_path = _resolve_checkpoint(config, config_name, args.checkpoint)
    device = torch.device(args.device)
    model = _load_model(config, checkpoint_path, device)

    shape = _determine_input_shape(config, args)
    dummy = torch.zeros(shape, dtype=torch.float32, device=device)

    output_path = Path(args.output) if args.output else checkpoint_path.with_suffix(".onnx")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dynamic_axes = _build_dynamic_axes(args)
    torch.onnx.export(
        model,
        dummy,
        output_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=args.opset,
        dynamic_axes=dynamic_axes,
    )
    normalization = _resolve_normalization(config)
    _embed_metadata(output_path, shape, normalization)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a FunTorch5 checkpoint to ONNX")
    parser.add_argument("config", help="Path or name of config file (YAML/JSON)")
    parser.add_argument("--checkpoint", help="Explicit checkpoint path; defaults to models/<config_name>/best or latest")
    parser.add_argument("--output", help="Destination ONNX path; defaults beside the checkpoint")
    parser.add_argument("--device", default="cpu", help="Device for exporting (default: cpu)")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version (default: 17)")
    parser.add_argument("--batch-size", type=int, help="Override batch dimension for dummy input")
    parser.add_argument("--sequence-length", type=int, help="Override sequence length for dummy input")
    parser.add_argument("--height", type=int, help="Override frame height for dummy input")
    parser.add_argument("--width", type=int, help="Override frame width for dummy input")
    parser.add_argument("--channels", type=int, help="Override input channels for dummy input")
    parser.add_argument("--dynamic-batch", action="store_true", help="Export with dynamic batch dimension")
    parser.add_argument("--dynamic-sequence", action="store_true", help="Export with dynamic sequence length")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = export_to_onnx(args)
    print(f"Exported ONNX model to {output_path} (device={args.device})")


if __name__ == "__main__":
    main()
