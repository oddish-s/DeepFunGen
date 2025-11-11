"""Full training entry-point for FunTorch5."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple

import numpy as np
import math
import mlflow

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader

from scripts.config import load_config
from scripts.core.dataset import DEFAULT_NORMALIZATION, InterferenceDataset, InterferenceFrameDataset
from scripts.core.model import create_model
from scripts.core.preprocessor import VideoPreprocessor
from scripts.core.trainer import EpochMetrics, TrainingHistory, Trainer


class SequenceAugmentor:
    """Deterministic, picklable sequence augmentation pipeline."""

    def __init__(
        self,
        target_size: Tuple[int, int],
        *,
        use_horizontal_flip: bool,
        flip_probability: float,
        use_color_jitter: bool,
        use_random_affine: bool,
        use_random_resized_crop: bool,
        is_grayscale: bool = False,
    ) -> None:
        self.target_size = tuple(int(x) for x in target_size)
        self.use_horizontal_flip = bool(use_horizontal_flip)
        self.flip_probability = max(0.0, min(1.0, float(flip_probability)))
        self.use_color_jitter = bool(use_color_jitter)
        self.use_random_affine = bool(use_random_affine)
        self.use_random_resized_crop = bool(use_random_resized_crop)
        self.is_grayscale = bool(is_grayscale)

        self.brightness_delta = 0.25
        self.contrast_delta = 0.2
        self.saturation_delta = 0.15
        self.hue_delta = 0.02

        try:
            import torchvision.transforms.functional as _  # noqa: F401
            from torchvision.transforms import InterpolationMode as _  # noqa: F401
        except ImportError:  # pragma: no cover - torchvision optional during inference-only runs
            print("Warning: torchvision not available; training augmentations disabled.")
            self._available = False
        else:
            self._available = True

    def __call__(self, sequence: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
        if not self._available:
            return sequence

        frames = sequence

        if self.use_horizontal_flip and self.flip_probability > 0.0:
            if float(rng.random()) < self.flip_probability:
                frames = torch.flip(frames, dims=[3])

        if self.use_color_jitter:
            frames = self._apply_color_jitter(frames, rng)

        if self.use_random_affine:
            frames = self._apply_affine(frames, rng)

        if self.use_random_resized_crop:
            frames = self._apply_resized_crop(frames, rng)

        return torch.clamp(frames, 0.0, 1.0)

    # ------------------------------------------------------------------
    def _apply_color_jitter(self, frames: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
        from torchvision.transforms import functional as F
        brightness = float(rng.uniform(1.0 - self.brightness_delta, 1.0 + self.brightness_delta))
        contrast = float(rng.uniform(1.0 - self.contrast_delta, 1.0 + self.contrast_delta))
        saturation = float(rng.uniform(1.0 - self.saturation_delta, 1.0 + self.saturation_delta))
        hue = float(rng.uniform(-self.hue_delta, self.hue_delta))

        adjusted = []
        for frame in frames:
            out = F.adjust_brightness(frame, brightness)
            out = F.adjust_contrast(out, contrast)
            if not self.is_grayscale:
                out = F.adjust_saturation(out, saturation)
                out = F.adjust_hue(out, hue)
            adjusted.append(out)
        return torch.stack(adjusted, dim=0)

    # ------------------------------------------------------------------
    def _apply_affine(self, frames: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
        from torchvision.transforms import functional as F
        from torchvision.transforms import InterpolationMode
        height, width = frames.shape[-2:]
        angle = float(rng.uniform(-5.0, 5.0))
        max_dx = width * 0.02
        max_dy = height * 0.02
        translations = (
            float(rng.uniform(-max_dx, max_dx)),
            float(rng.uniform(-max_dy, max_dy)),
        )
        scale = float(rng.uniform(0.95, 1.05))
        shear = (0.0, 0.0)
        return torch.stack(
            [
                F.affine(
                    frame,
                    angle=angle,
                    translate=translations,
                    scale=scale,
                    shear=shear,
                    interpolation=InterpolationMode.BILINEAR,
                    fill=0.0,
                )
                for frame in frames
            ],
            dim=0,
        )

    # ------------------------------------------------------------------
    def _apply_resized_crop(self, frames: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
        from torchvision.transforms import functional as F
        from torchvision.transforms import InterpolationMode
        height, width = frames.shape[-2:]
        top, left, crop_h, crop_w = self._random_resized_crop_params(rng, height, width)
        return torch.stack(
            [
                F.resized_crop(
                    frame,
                    top,
                    left,
                    crop_h,
                    crop_w,
                    self.target_size,
                    interpolation=InterpolationMode.BILINEAR,
                )
                for frame in frames
            ],
            dim=0,
        )

    # ------------------------------------------------------------------
    def _random_resized_crop_params(
        self, rng: np.random.Generator, height: int, width: int
    ) -> tuple[int, int, int, int]:
        area = height * width
        scale_min, scale_max = 0.9, 1.0
        ratio_min, ratio_max = 0.95, 1.05
        for _ in range(10):
            target_area = rng.uniform(scale_min, scale_max) * area
            log_ratio = (np.log(ratio_min), np.log(ratio_max))
            aspect = np.exp(rng.uniform(*log_ratio))
            w = int(round(np.sqrt(target_area * aspect)))
            h = int(round(np.sqrt(target_area / aspect)))
            if 0 < w <= width and 0 < h <= height:
                top = int(rng.integers(0, height - h + 1))
                left = int(rng.integers(0, width - w + 1))
                return top, left, h, w
        target_h, target_w = self.target_size
        in_ratio = target_h / max(target_w, 1)
        if height < width:
            w = int(round(height / max(in_ratio, 1e-6)))
            w = min(w, width)
            h = height
        else:
            h = int(round(width * in_ratio))
            h = min(h, height)
            w = width
        top = (height - h) // 2
        left = (width - w) // 2
        return top, left, h, w


def _build_train_transform(
    target_size: Tuple[int, int],
    *,
    use_horizontal_flip: bool,
    flip_probability: float,
    use_color_jitter: bool,
    use_random_affine: bool,
    use_random_resized_crop: bool,
    is_grayscale: bool = False,
) -> Optional[SequenceAugmentor]:
    if not (
        use_horizontal_flip
        or use_color_jitter
        or use_random_affine
        or use_random_resized_crop
    ):
        return None
    return SequenceAugmentor(
        target_size,
        use_horizontal_flip=use_horizontal_flip,
        flip_probability=flip_probability,
        use_color_jitter=use_color_jitter,
        use_random_affine=use_random_affine,
        use_random_resized_crop=use_random_resized_crop,
        is_grayscale=is_grayscale,
    )


def _flatten_for_mlflow(data: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    items: dict[str, Any] = {}
    for key, value in data.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            items.update(_flatten_for_mlflow(value, name))
        elif isinstance(value, (list, tuple)):
            items[name] = json.dumps(value)
        else:
            items[name] = value
    return items


def _average_loss(metrics: Sequence[EpochMetrics]) -> Optional[float]:
    if not metrics:
        return None
    return float(sum(m.loss for m in metrics) / len(metrics))


def _loss_improvement_ratio(history: Sequence[EpochMetrics], window: int = 5) -> Optional[float]:
    if len(history) < 2:
        return None
    start_window = history[:window]
    end_window = history[-window:]
    start_avg = _average_loss(start_window)
    end_avg = _average_loss(end_window)
    if start_avg is None or end_avg is None or start_avg == 0.0:
        return None
    return (start_avg - end_avg) / abs(start_avg)


def _log_improvement_metrics(history: TrainingHistory, window: int = 5) -> None:
    train_ratio = _loss_improvement_ratio(history.train, window)
    if train_ratio is not None:
        mlflow.log_metric("train/loss_relative_improvement", train_ratio)
    val_ratio = _loss_improvement_ratio(history.val, window)
    if val_ratio is not None:
        mlflow.log_metric("val/loss_relative_improvement", val_ratio)


def _maybe_resume(args: argparse.Namespace, checkpoint_dir: Path) -> Path | None:
    if args.no_resume:
        return None
    if args.resume:
        return Path(args.resume)
    latest = checkpoint_dir / "latest_checkpoint.pth"
    return latest if latest.exists() else None


def _ensure_preprocessing(config: Mapping[str, Any], force: bool = False) -> None:
    data_cfg = config.get("data_config") or {}
    folders = data_cfg.get("video_folders", [])
    if not folders:
        raise ValueError("data_config.video_folders is empty; nothing to train on")
    target_size = tuple(data_cfg.get("target_size", (224, 224)))
    aspect_bounds_cfg = data_cfg.get("aspect_ratio_bounds", "__UNSPECIFIED__")
    vp_kwargs: dict[str, Any] = {"target_size": target_size}
    if isinstance(aspect_bounds_cfg, (list, tuple)) and len(aspect_bounds_cfg) == 2:
        vp_kwargs["aspect_ratio_bounds"] = (
            float(aspect_bounds_cfg[0]),
            float(aspect_bounds_cfg[1]),
        )
    elif aspect_bounds_cfg is None:
        vp_kwargs["aspect_ratio_bounds"] = None
    vp_kwargs["use_vr_left_half"] = bool(data_cfg.get("vr_video", False))

    preprocessor = VideoPreprocessor(**vp_kwargs)
    for folder in folders:
        result = preprocessor.process_video_folder(folder, force=force)
        reused = "(reused)" if result.reused_frames else "(new)"
        print(f"Preprocessed {result.video_folder} -> {result.frame_count} frames {reused}")


def _resolve_normalization(train_cfg: Mapping[str, Any]) -> tuple[float, float]:
    norm = train_cfg.get("normalization")
    if norm and isinstance(norm, (list, tuple)) and len(norm) == 2:
        return float(norm[0]), float(norm[1])
    return DEFAULT_NORMALIZATION


def _build_sequence_dataloaders(config: Mapping[str, Any], num_workers: int) -> tuple[DataLoader, DataLoader | None]:
    data_cfg = config.get("data_config") or {}
    train_cfg = config.get("training_config") or {}
    folders = data_cfg.get("video_folders", [])
    normalization = _resolve_normalization(train_cfg)
    scale_factor = train_cfg.get("scale_factor")
    target_size = tuple(int(x) for x in data_cfg.get("target_size", (224, 224)))
    use_grayscale = bool(data_cfg.get("grayscale", False))
    train_transform = _build_train_transform(
        target_size,
        use_horizontal_flip=bool(data_cfg.get("horizontal_flip", False)),
        flip_probability=float(data_cfg.get("flip_probability", 0.5)),
        use_color_jitter=bool(data_cfg.get("color_jitter", False)),
        use_random_affine=bool(data_cfg.get("random_affine", False)),
        use_random_resized_crop=bool(data_cfg.get("random_resized_crop", False)),
        is_grayscale=use_grayscale,
    )

    samples_per_epoch = train_cfg.get("samples_per_epoch")
    val_samples = train_cfg.get("val_samples_per_epoch", samples_per_epoch)
    base_kwargs = dict(
        video_folders=folders,
        train_ratio=train_cfg.get("train_split", 0.8),
        sequence_length=data_cfg.get("sequence_length", 32),
        prediction_offset=data_cfg.get("prediction_offset", 0),
        safety_gap=data_cfg.get("safety_gap", 10),
        split_mode=data_cfg.get("split_mode", "sequential"),
        normalization=normalization,
        scale_factor=float(scale_factor) if scale_factor is not None else None,
        use_absolute=bool(data_cfg.get("use_absolute", False)),
        include_ignored=bool(data_cfg.get("include_ignored_ranges", False)),
        sequence_targets=bool(data_cfg.get("sequence_targets", False)),
        grayscale=use_grayscale,
    )
    split_seed = train_cfg.get("seed", 42)
    train_dataset = InterferenceDataset(
        samples_per_epoch=samples_per_epoch,
        train_mode=True,
        dataset_split="train",
        seed=split_seed,
        transform=train_transform,
        **base_kwargs,
    )
    val_dataset = None
    if train_cfg.get("use_validation", True):
        try:
            val_dataset = InterferenceDataset(
                samples_per_epoch=val_samples,
                train_mode=False,
                dataset_split="val",
                seed=split_seed,
                **base_kwargs,
            )
            if val_samples:
                requested_val = int(val_samples)
                actual_val = len(getattr(val_dataset, "_population", val_dataset))
                if actual_val < requested_val:
                    print(
                        f"Warning: validation set has only {actual_val} unique samples, "
                        f"less than requested val_samples_per_epoch={requested_val}. "
                        "Validation loader will reuse sequences."
                    )
        except ValueError:
            print("Warning: validation split could not be created (insufficient data). Continuing without val set.")
            val_dataset = None

    batch_size = train_cfg.get("batch_size", 16)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=max(0, num_workers // 2),
            pin_memory=torch.cuda.is_available(),
        )
        if val_dataset is not None
        else None
    )
    return train_loader, val_loader


def _build_frame_classifier_dataloaders(
    config: Mapping[str, Any], num_workers: int
) -> tuple[DataLoader, DataLoader | None]:
    data_cfg = config.get("data_config") or {}
    train_cfg = config.get("training_config") or {}
    folders = data_cfg.get("video_folders", [])
    if not folders:
        raise ValueError("data_config.video_folders must be provided for frame classifier training.")
    include_setting = data_cfg.get("include_ignored_ranges", True)
    if not include_setting:
        print("Warning: include_ignored_ranges disabled in config; overriding to True for frame classifier.")
    include_ignored = True
    seed = train_cfg.get("seed", 42)
    max_negative_ratio = train_cfg.get("max_negative_ratio")
    use_grayscale = bool(data_cfg.get("grayscale", False))

    base_dataset = InterferenceFrameDataset(
        video_folders=folders,
        include_ignored=include_ignored,
        transform=None,
        seed=seed,
        max_negative_ratio=max_negative_ratio,
        grayscale=use_grayscale,
    )
    all_samples = base_dataset.all_samples
    total_samples = len(all_samples)
    if total_samples == 0:
        raise ValueError("No samples generated for frame classifier.")

    use_validation = bool(train_cfg.get("use_validation", True)) and total_samples > 1
    split_ratio = float(train_cfg.get("train_split", 0.8))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(total_samples)
    if use_validation:
        split_idx = max(1, int(np.floor(total_samples * split_ratio)))
        if split_idx >= total_samples:
            split_idx = total_samples - 1
        train_indices = perm[:split_idx]
        val_indices = perm[split_idx:]
        if len(val_indices) == 0:
            val_indices = None
    else:
        train_indices = perm
        val_indices = None

    train_dataset = InterferenceFrameDataset(
        video_folders=folders,
        include_ignored=include_ignored,
        transform=None,
        seed=seed,
        max_negative_ratio=max_negative_ratio,
        samples=all_samples,
        indices=train_indices,
        grayscale=use_grayscale,
    )
    val_dataset = (
        InterferenceFrameDataset(
            video_folders=folders,
            include_ignored=include_ignored,
            transform=None,
            seed=seed,
            max_negative_ratio=max_negative_ratio,
            samples=all_samples,
            indices=val_indices,
            grayscale=use_grayscale,
        )
        if val_indices is not None
        else None
    )

    batch_size = train_cfg.get("batch_size", 64)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=max(0, num_workers // 2),
            pin_memory=torch.cuda.is_available(),
        )
        if val_dataset is not None
        else None
    )
    return train_loader, val_loader


def _build_dataloaders(config: Mapping[str, Any], num_workers: int) -> tuple[DataLoader, DataLoader | None]:
    model_cfg = config.get("model_config") or {}
    model_name = str(model_cfg.get("name", "sequence_regressor")).lower()
    if model_name == "frame_classifier":
        return _build_frame_classifier_dataloaders(config, num_workers)
    return _build_sequence_dataloaders(config, num_workers)

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


def main() -> None:
    parser = argparse.ArgumentParser(description="FunTorch5 training harness")
    parser.add_argument("config", help="Path or name of config file (YAML/JSON)")
    parser.add_argument("--resume", type=str, help="Checkpoint path to resume from")
    parser.add_argument("--no-resume", action="store_true", help="Ignore existing checkpoints")
    parser.add_argument("--force-preprocess", action="store_true", help="Re-run preprocessing even if cached")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--run-name", type=str, help="Optional suffix to distinguish this run (affects checkpoints/MLflow run name)")
    parser.add_argument("--experiment", type=str, help="Override MLflow experiment name")
    parser.add_argument("--skip-existing", action="store_true", help="Skip run if checkpoint directory already exists")
    args = parser.parse_args()

    config_path, config_name = _resolve_config(args.config)
    config = load_config(config_path)
    _ensure_preprocessing(config, force=args.force_preprocess)

    model = create_model(config["model_config"])
    optimizer_cfg = config.get("optimizer_config", {"name": "adamw", "lr": 5e-4})
    trainer_cfg = config.get("training_config", {})
    scheduler_cfg = trainer_cfg.get("scheduler_config")

    checkpoint_root = Path(trainer_cfg.get("checkpoint_dir", "models"))
    run_identifier = config_name
    if args.run_name:
        safe_suffix = args.run_name.strip().replace(" ", "_")
        if safe_suffix:
            run_identifier = f"{config_name}__{safe_suffix}"
    checkpoint_dir = checkpoint_root / run_identifier
    checkpoint_exists = checkpoint_dir.exists()
    if checkpoint_exists and args.skip_existing:
        print(f"Run '{run_identifier}' already has checkpoints. Skipping as requested.")
        return
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    resume_path = _maybe_resume(args, checkpoint_dir)
    train_loader, val_loader = _build_dataloaders(config, args.num_workers)

    trainer = Trainer(
        model=model,
        optimizer_cfg=optimizer_cfg,
        training_cfg=trainer_cfg,
        scheduler_cfg=scheduler_cfg,
        checkpoint_dir=checkpoint_dir,
        resume_path=resume_path,
    )

    experiment_name = args.experiment or config.get("mlflow", {}).get("experiment", config_name)
    mlflow.set_experiment(experiment_name)

    run_tags = {"config_name": config_name}
    if args.run_name:
        run_tags["run_suffix"] = args.run_name
    with mlflow.start_run(run_name=run_identifier, tags=run_tags):
        mlflow.log_params(_flatten_for_mlflow(config))
        mlflow.log_text(json.dumps(config, indent=2), "config_snapshot.json")

        def _log_epoch(epoch_idx: int, train_metrics: EpochMetrics, val_metrics: Optional[EpochMetrics]) -> None:
            step = epoch_idx + 1
            metrics = {
                "train/loss": train_metrics.loss,
            }
            if train_metrics.output_variance is not None:
                metrics["train/variance"] = train_metrics.output_variance
            if train_metrics.accuracy is not None:
                metrics["train/accuracy"] = train_metrics.accuracy
            if val_metrics is not None:
                metrics["val/loss"] = val_metrics.loss
                if val_metrics.correlation is not None:
                    metrics["val/correlation"] = val_metrics.correlation
                if val_metrics.abs_error is not None:
                    metrics["val/abs_error"] = val_metrics.abs_error
                if val_metrics.output_variance is not None:
                    metrics["val/variance"] = val_metrics.output_variance
                if val_metrics.accuracy is not None:
                    metrics["val/accuracy"] = val_metrics.accuracy
            mlflow.log_metrics(metrics, step=step)

        history = trainer.fit(train_loader, val_loader, epoch_callback=_log_epoch)

        if trainer.best_val_loss < math.inf:
            mlflow.log_metric("best_val_loss", trainer.best_val_loss)

        history_path = checkpoint_dir / "training_history.json"
        history_payload = {
            "train": [vars(m) for m in history.train],
            "val": [vars(m) for m in history.val],
        }
        with history_path.open("w", encoding="utf-8") as handle:
            json.dump(history_payload, handle, indent=2)
        print(f"Training history saved to {history_path}")
        mlflow.log_artifact(str(history_path))
        _log_improvement_metrics(history)
        if resume_path:
            mlflow.set_tag("resumed_from", str(resume_path))

        best_ckpt = checkpoint_dir / "best_checkpoint.pth"
        if best_ckpt.exists():
            mlflow.log_artifact(str(best_ckpt))

        latest_ckpt = checkpoint_dir / "latest_checkpoint.pth"
        if latest_ckpt.exists():
            mlflow.log_artifact(str(latest_ckpt))


if __name__ == "__main__":
    main()
