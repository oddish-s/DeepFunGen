"""Inference orchestration for sequence models."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence

import re

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from scripts.core.dataset import DEFAULT_NORMALIZATION, build_sequences_for_video, load_frame_tensor
from scripts.core.model import create_model
from scripts.core.preprocessor import VideoPreprocessor


@dataclass
class InferenceResult:
    video_folder: Path
    predictions_path: Path
    frame_count: int
    reused_predictions: bool


class _AuxiliaryClassifier:
    """Wrapper around the frame-level ignore classifier."""

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        *,
        threshold: float = 0.5,
        batch_size: int = 64,
    ) -> None:
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.threshold = float(threshold)
        self.batch_size = max(1, int(batch_size))

    def predict(self, paths: Sequence[Path], loader: Callable[[Path], torch.Tensor]) -> np.ndarray:
        if not paths:
            return np.empty(0, dtype=np.float32)
        outputs: List[float] = []
        for start in range(0, len(paths), self.batch_size):
            batch_paths = paths[start : start + self.batch_size]
            tensors = [loader(path) for path in batch_paths]
            batch = torch.stack(tensors, dim=0).to(self.device, non_blocking=True)
            with torch.no_grad():
                logits = self.model(batch)
                probs = torch.sigmoid(logits).detach().cpu().numpy()
            outputs.extend(float(p) for p in probs)
        return np.asarray(outputs, dtype=np.float32)


class InferenceEngine:
    """Run inference over videos using the sequence regression model."""

    def __init__(
        self,
        model_path: str | Path,
        config: Mapping[str, object],
        device: Optional[str] = None,
        predictions_name: str | None = None,
    ) -> None:
        self.config = config
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {self.model_path}")

        self.device = torch.device(device) if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.predictions_name = self._resolve_predictions_name(predictions_name)
        self.model = self._load_model(self.config.get("model_config", {}))
        self.model.to(self.device)
        self.model.eval()

        data_cfg = self.config.get("data_config", {})
        target_size = tuple(data_cfg.get("target_size", (224, 224)))
        self.sequence_length = int(data_cfg.get("sequence_length", 32))
        self.prediction_offset = int(data_cfg.get("prediction_offset", 0))
        self.use_absolute = bool(data_cfg.get("use_absolute", False))
        self.include_ignored = bool(data_cfg.get("include_ignored_ranges", False))
        self.use_vr_left_half = bool(data_cfg.get("vr_video", False))
        self.use_grayscale = bool(data_cfg.get("grayscale", False))
        self.preprocessor = VideoPreprocessor(
            target_size=target_size,
            use_vr_left_half=self.use_vr_left_half,
        )
        self.mean, self.std = self._resolve_normalization(data_cfg)
        self._frame_cache: Dict[Path, Dict[int, Path]] = {}
        aux_cfg = self.config.get("auxiliary_classifier")
        if isinstance(aux_cfg, Mapping) and aux_cfg:
            self.ignore_classifier = self._load_auxiliary_classifier(aux_cfg)
        else:
            self.ignore_classifier = None

    # ------------------------------------------------------------------

    def _resolve_predictions_name(self, override: Optional[str]) -> str:
        if override:
            return self._sanitize_component(override, fallback="default")

        config_name = str(self.config.get("name", "default"))
        model_cfg = self.config.get("model_config", {})
        model_name = str(model_cfg.get("name", "model"))

        config_part = self._sanitize_component(config_name, fallback="default")
        model_part = self._sanitize_component(model_name, fallback="model")
        return f"{config_part}({model_part})"

    @staticmethod
    def _sanitize_component(value: str, fallback: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9._()-]+", "_", value)
        cleaned = cleaned.strip("._()-")
        return cleaned or fallback

    def _load_model(self, model_cfg: Mapping[str, object]) -> torch.nn.Module:
        model = create_model(model_cfg)
        checkpoint = torch.load(self.model_path, map_location="cpu")
        state = checkpoint.get("model_state", checkpoint)
        model.load_state_dict(state)
        return model

    def _load_auxiliary_classifier(self, cfg: Mapping[str, object]) -> Optional[_AuxiliaryClassifier]:
        if not cfg.get("enabled", True):
            return None
        model_cfg = cfg.get("model_config")
        if model_cfg is None:
            config_path = cfg.get("config_path")
            if config_path:
                from scripts.config import load_config

                loaded = load_config(Path(config_path))
                model_cfg = loaded.get("model_config")
        if model_cfg is None:
            raise ValueError("auxiliary_classifier requires a 'model_config' or 'config_path'.")
        checkpoint_path = cfg.get("checkpoint")
        if not checkpoint_path:
            raise ValueError("auxiliary_classifier requires a 'checkpoint'.")
        model = create_model(model_cfg)
        checkpoint = torch.load(Path(checkpoint_path), map_location="cpu")
        state = checkpoint.get("model_state", checkpoint)
        model.load_state_dict(state)
        threshold = float(cfg.get("threshold", 0.5))
        batch_size = int(cfg.get("batch_size", 64))
        return _AuxiliaryClassifier(model=model, device=self.device, threshold=threshold, batch_size=batch_size)

    # ------------------------------------------------------------------
    def _resolve_normalization(self, data_cfg: Mapping[str, object]):
        norm_cfg = data_cfg.get("normalization") if isinstance(data_cfg, Mapping) else None
        if norm_cfg:
            if isinstance(norm_cfg, Mapping):
                mean = float(norm_cfg.get("mean", DEFAULT_NORMALIZATION[0]))
                std = float(norm_cfg.get("std", DEFAULT_NORMALIZATION[1]))
                return mean, std
            if isinstance(norm_cfg, (list, tuple)) and len(norm_cfg) == 2:
                return float(norm_cfg[0]), float(norm_cfg[1])
        training_cfg = self.config.get("training_config") if isinstance(self.config, Mapping) else None
        if isinstance(training_cfg, Mapping):
            train_norm = training_cfg.get("normalization")
            if isinstance(train_norm, Mapping):
                mean = float(train_norm.get("mean", DEFAULT_NORMALIZATION[0]))
                std = float(train_norm.get("std", DEFAULT_NORMALIZATION[1]))
                return mean, std
            if isinstance(train_norm, (list, tuple)) and len(train_norm) == 2:
                return float(train_norm[0]), float(train_norm[1])
        return DEFAULT_NORMALIZATION

    # ------------------------------------------------------------------
    def predict_video(self, video_folder: str | Path, force: bool = False) -> InferenceResult:
        folder = Path(video_folder)
        preprocess_result = self.preprocessor.process_video_folder(folder, force=force)
        predictions_file = folder / f"predictions-{self.predictions_name}.csv"
        legacy_file = folder / "predictions.csv"
        if predictions_file.exists() and not force:
            existing = self.load_existing_predictions(predictions_file, preprocess_result.frame_count)
            if existing is not None:
                return InferenceResult(
                    video_folder=folder,
                    predictions_path=predictions_file,
                    frame_count=len(existing),
                    reused_predictions=True,
                )
        if legacy_file.exists() and not force:
            existing = self.load_existing_predictions(legacy_file, preprocess_result.frame_count)
            if existing is not None:
                return InferenceResult(
                    video_folder=folder,
                    predictions_path=legacy_file,
                    frame_count=len(existing),
                    reused_predictions=True,
                )

        predictions = self._run_inference(folder, preprocess_result.frame_count)
        predictions.to_csv(predictions_file, index=False)
        return InferenceResult(
            video_folder=folder,
            predictions_path=predictions_file,
            frame_count=len(predictions),
            reused_predictions=False,
        )

    # ------------------------------------------------------------------
    def load_existing_predictions(self, predictions_file: Path, expected_frames: int) -> Optional[pd.DataFrame]:
        df = pd.read_csv(predictions_file)
        if len(df) != expected_frames:
            print(
                f"Existing predictions length mismatch (have {len(df)} vs expected {expected_frames})."
                " Recomputing..."
            )
            return None
        return df

    # ------------------------------------------------------------------
    def _run_inference(self, folder: Path, frame_count: int) -> pd.DataFrame:
        sequences = build_sequences_for_video(
            folder,
            sequence_length=self.sequence_length,
            prediction_offset=self.prediction_offset,
            include_ignored=self.include_ignored,
        )
        if not sequences:
            raise RuntimeError("No valid sequences generated for inference")

        frames_dir = folder / "frames"
        frame_paths = sorted(frames_dir.glob("*.jpg"))
        frame_map = {int(path.stem): path for path in frame_paths if path.stem.isdigit()}
        if frame_map:
            self._frame_cache[folder] = frame_map
        if len(frame_paths) != frame_count:
            frame_count = len(frame_paths)

        meta_path = folder / "meta.csv"
        if meta_path.exists():
            meta_df = pd.read_csv(meta_path).sort_values("frame_index")
            base_value = float(meta_df.iloc[0]["interference_value"]) if not meta_df.empty else 0.0
            timestamps = meta_df["timestamp_ms"].tolist()
        else:
            meta_df = None
            base_value = 0.0
            timestamps = [i * (1000 / 30.0) for i in range(frame_count)]

        sums = np.zeros(frame_count, dtype=np.float32)
        counts = np.zeros(frame_count, dtype=np.float32)

        for sample in tqdm(sequences, desc=f"Infer {folder.name}", unit="seq"):
            tensor = self._load_sequence(sample.frame_paths)
            tensor = tensor.unsqueeze(0).to(self.device)
            with torch.no_grad():
                predictions = self.model(tensor).detach().cpu().squeeze(0)
            if predictions.ndim == 0:
                predictions = predictions.reshape(1)
            denormalized = (predictions.numpy() * self.std) + self.mean
            frame_indices = getattr(sample, "frame_indices", None)
            if frame_indices is None:
                frame_indices = [sample.target_frame_index]
            target_indices = frame_indices
            if len(denormalized) == 1 and len(frame_indices) > 1:
                target_indices = [sample.target_frame_index]
            for local_idx, frame_idx in enumerate(target_indices):
                if local_idx >= len(denormalized):
                    break
                idx = int(frame_idx)
                if 0 <= idx < frame_count:
                    sums[idx] += float(denormalized[local_idx])
                    counts[idx] += 1

        valid_mask = counts > 0
        averaged = np.zeros(frame_count, dtype=np.float32)
        averaged[valid_mask] = sums[valid_mask] / counts[valid_mask]

        if self.use_absolute:
            predicted_value = averaged.copy()
            if valid_mask.any():
                first_idx = int(np.argmax(valid_mask))
                fill_value = base_value if meta_df is not None else predicted_value[first_idx]
                predicted_value[:first_idx] = fill_value
                for i in range(first_idx + 1, frame_count):
                    if not valid_mask[i]:
                        predicted_value[i] = predicted_value[i - 1]
            else:
                predicted_value.fill(base_value)
            predicted_change = np.zeros(frame_count, dtype=np.float32)
            if frame_count > 1:
                predicted_change[1:] = np.diff(predicted_value)
        else:
            predicted_change = averaged
            predicted_change[~valid_mask] = 0.0
            predicted_value = np.zeros(frame_count, dtype=np.float32)
            predicted_value[0] = base_value
            for i in range(1, frame_count):
                predicted_value[i] = predicted_value[i - 1] + predicted_change[i]

        if meta_df is not None and len(timestamps) < frame_count:
            timestamps = list(timestamps) + [timestamps[-1] + 33.3] * (frame_count - len(timestamps))

        df = pd.DataFrame(
            {
                "frame_index": np.arange(frame_count, dtype=np.int32),
                "timestamp_ms": timestamps[:frame_count],
                "predicted_change": predicted_change,
                "predicted_value": predicted_value,
            }
        )
        return df

    # ------------------------------------------------------------------
    def classify_frame_indices(
        self, video_folder: str | Path, frame_indices: Sequence[int]
    ) -> Dict[int, float]:
        if self.ignore_classifier is None or not frame_indices:
            return {}
        folder = Path(video_folder)
        mapping = self._frame_cache.get(folder)
        if mapping is None:
            frames_dir = folder / "frames"
            mapping = {int(path.stem): path for path in sorted(frames_dir.glob("*.jpg")) if path.stem.isdigit()}
            self._frame_cache[folder] = mapping
        ordered: List[tuple[int, Path]] = []
        for idx in frame_indices:
            frame_idx = int(idx)
            path = mapping.get(frame_idx)
            if path is not None:
                ordered.append((frame_idx, path))
        if not ordered:
            return {}
        probabilities = self.ignore_classifier.predict([path for _, path in ordered], self._load_frame)
        return {idx: float(prob) for (idx, _), prob in zip(ordered, probabilities)}

    # ------------------------------------------------------------------
    def _load_sequence(self, frame_paths: List[Path]) -> torch.Tensor:
        frames = [self._load_frame(path) for path in frame_paths]
        return torch.stack(frames, dim=0)

    def _load_frame(self, path: Path) -> torch.Tensor:
        return load_frame_tensor(path, grayscale=self.use_grayscale)


__all__ = ["InferenceEngine", "InferenceResult"]
