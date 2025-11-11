"""Sequence-based dataset for FunTorch5."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

IGNORE_FILE_NAME = "ignore.txt"
META_FILE_NAME = "meta.csv"
FRAME_GLOB = "*.jpg"
DEFAULT_SEQUENCE_LENGTH = 32
DEFAULT_PREDICTION_OFFSET = 0
DEFAULT_SAFETY_GAP = 10
DEFAULT_NORMALIZATION = (0.0015, 6.686)


@dataclass
class FrameRecord:
    frame_index: int
    path: Path
    change: float
    value: float


@dataclass
class SequenceSample:
    video_folder: Path
    frame_paths: List[Path]
    frame_indices: List[int]
    target_change: float
    target_value: float
    change_sequence: List[float]
    value_sequence: List[float]
    target_frame_index: int


def load_frame_tensor(path: Path, *, grayscale: bool = False) -> torch.Tensor:
    """Load a frame from disk into a torch tensor (C, H, W)."""
    with Image.open(path) as img:
        if grayscale:
            img = img.convert("L")
            arr = np.asarray(img, dtype=np.float32) / 255.0
            arr = np.expand_dims(arr, axis=0)
            return torch.from_numpy(arr)
        img = img.convert("RGB")
        arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def _parse_ignore_ranges(folder: Path) -> List[Tuple[int, int]]:
    ignore_path = folder / IGNORE_FILE_NAME
    if not ignore_path.exists():
        return []
    ranges: List[Tuple[int, int]] = []
    with ignore_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "-" not in line:
                continue
            start_str, end_str = line.split("-", 1)
            try:
                start = int(start_str)
                end = int(end_str)
            except ValueError:
                continue
            if end < start:
                start, end = end, start
            ranges.append((start, end))
    return sorted(ranges, key=lambda pair: pair[0])


def _load_meta_dataframe(folder: Path) -> pd.DataFrame:
    meta_path = folder / META_FILE_NAME
    if meta_path.exists():
        return pd.read_csv(meta_path)
    # Fallback: build empty meta with zeros so training can proceed
    frames_dir = folder / "frames"
    frame_paths = sorted(frames_dir.glob(FRAME_GLOB))
    frame_indices = [int(fp.stem) for fp in frame_paths if fp.stem.isdigit()]
    return pd.DataFrame(
        {
            "frame_index": frame_indices,
            "timestamp_ms": np.array(frame_indices, dtype=np.float32) * (1000 / 30.0),
            "interference_value": np.zeros(len(frame_indices), dtype=np.float32),
            "interference_change": np.zeros(len(frame_indices), dtype=np.float32),
        }
    )


def _apply_ignore_ranges(
    folder: Path,
    meta_df: pd.DataFrame,
    *,
    include_ignored: bool = False,
) -> pd.DataFrame:
    ranges = _parse_ignore_ranges(folder)
    if not ranges:
        return meta_df
    indices = meta_df["frame_index"].to_numpy(dtype=np.int32)
    if include_ignored:
        meta_df = meta_df.copy()
        mask = np.zeros(len(meta_df), dtype=bool)
        for start, end in ranges:
            mask |= (indices >= start) & (indices <= end)
        if mask.any():
            for column in ("interference_change", "interference_value"):
                if column in meta_df.columns:
                    meta_df.loc[mask, column] = 0.0
        return meta_df.reset_index(drop=True)

    mask = np.ones(len(meta_df), dtype=bool)
    for start, end in ranges:
        mask &= ~((indices >= start) & (indices <= end))
    return meta_df.loc[mask].reset_index(drop=True)


def _load_frame_records(folder: Path, *, include_ignored: bool = False) -> List[FrameRecord]:
    frames_dir = folder / "frames"
    if not frames_dir.exists():
        raise FileNotFoundError(f"Frames directory missing: {frames_dir}")

    frame_paths = sorted(frames_dir.glob(FRAME_GLOB))
    meta_df = _apply_ignore_ranges(
        folder,
        _load_meta_dataframe(folder),
        include_ignored=include_ignored,
    )
    meta_df = meta_df.sort_values("frame_index").reset_index(drop=True)

    meta_lookup = meta_df.set_index("frame_index")
    records: List[FrameRecord] = []
    for frame_path in frame_paths:
        if not frame_path.stem.isdigit():
            continue
        frame_index = int(frame_path.stem)
        if frame_index not in meta_lookup.index:
            continue
        change = float(meta_lookup.loc[frame_index, "interference_change"])
        value = float(meta_lookup.loc[frame_index, "interference_value"])
        records.append(FrameRecord(frame_index=frame_index, path=frame_path, change=change, value=value))
    return sorted(records, key=lambda rec: rec.frame_index)


def _split_contiguous_segments(records: List[FrameRecord]) -> List[List[FrameRecord]]:
    if not records:
        return []
    segments: List[List[FrameRecord]] = []
    current: List[FrameRecord] = [records[0]]
    for record in records[1:]:
        if record.frame_index == current[-1].frame_index + 1:
            current.append(record)
        else:
            segments.append(current)
            current = [record]
    segments.append(current)
    return segments


def build_sequences_for_video(
    folder: Path,
    sequence_length: int,
    prediction_offset: int,
    *,
    include_ignored: bool = False,
) -> List[SequenceSample]:
    """Construct sliding-window sequences for a single video."""
    records = _load_frame_records(folder, include_ignored=include_ignored)
    if not records:
        return []
    sequences: List[SequenceSample] = []
    min_len = sequence_length + prediction_offset
    for segment in _split_contiguous_segments(records):
        if len(segment) < min_len:
            continue
        for start in range(0, len(segment) - min_len + 1):
            window = segment[start : start + sequence_length]
            target_record = segment[start + sequence_length - 1 + prediction_offset]
            frame_indices = [rec.frame_index for rec in window]
            change_sequence = [float(rec.change) for rec in window]
            value_sequence = [float(rec.value) for rec in window]
            frame_paths = [rec.path for rec in window]
            sequences.append(
                SequenceSample(
                    video_folder=folder,
                    frame_paths=frame_paths,
                    frame_indices=frame_indices,
                    target_change=target_record.change,
                    target_value=target_record.value,
                    change_sequence=change_sequence,
                    value_sequence=value_sequence,
                    target_frame_index=target_record.frame_index,
                )
            )
    return sequences


class InterferenceDataset(Dataset):
    """Returns sequences of frames suitable for temporal models."""

    def __init__(
        self,
        video_folders: Sequence[str | Path],
        samples_per_epoch: Optional[int],
        train_mode: bool = True,
        dataset_split: str = "train",
        train_ratio: float = 0.8,
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
        prediction_offset: int = DEFAULT_PREDICTION_OFFSET,
        safety_gap: int = DEFAULT_SAFETY_GAP,
        normalization: Tuple[float, float] = DEFAULT_NORMALIZATION,
        scale_factor: Optional[float] = None,
        use_absolute: bool = False,
        seed: Optional[int] = None,
        transform=None,
        split_mode: str = "sequential",
        include_ignored: bool = False,
        sequence_targets: bool = False,
        grayscale: bool = False,
    ) -> None:
        super().__init__()
        if dataset_split not in {"train", "val"}:
            raise ValueError("dataset_split must be either 'train' or 'val'")
        split_mode = split_mode.lower()
        if split_mode not in {"sequential", "random_chunk", "video"}:
            raise ValueError("split_mode must be one of 'sequential', 'random_chunk', or 'video'")

        self.video_folders = [Path(p) for p in video_folders]
        self.samples_per_epoch = samples_per_epoch
        self.train_mode = train_mode
        self.dataset_split = dataset_split
        self.train_ratio = float(train_ratio)
        self.sequence_length = int(sequence_length)
        self.prediction_offset = int(prediction_offset)
        self.safety_gap = int(safety_gap)
        self.mean, self.std = normalization
        if scale_factor is not None:
            scale_factor = float(scale_factor)
            if scale_factor == 0.0:
                raise ValueError("scale_factor must be non-zero when provided")
            if not math.isfinite(scale_factor):
                raise ValueError("scale_factor must be a finite number")
        self.scale_factor = scale_factor
        self.use_absolute = bool(use_absolute)
        self.split_mode = split_mode
        self.transform = transform
        self._rng = np.random.default_rng(seed)
        self.include_ignored = bool(include_ignored)
        self.sequence_targets = bool(sequence_targets)
        self.grayscale = bool(grayscale)

        sequences_by_video: Dict[Path, List[SequenceSample]] = {}
        for folder in self.video_folders:
            sequences = build_sequences_for_video(
                folder,
                self.sequence_length,
                self.prediction_offset,
                include_ignored=self.include_ignored,
            )
            if sequences:
                sequences_by_video[folder] = sequences

        split_sequences = self._split_sequences(sequences_by_video)
        target_sequences = split_sequences[dataset_split]
        if not target_sequences:
            raise ValueError(f"No samples produced for split '{dataset_split}'")
        self._population = target_sequences
        self._active_population = self._prepare_epoch_population()

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._active_population)

    # ------------------------------------------------------------------
    def __getitem__(self, index: int):
        sample = self._active_population[index]

        frames = [self._load_frame(path) for path in sample.frame_paths]
        stacked = torch.stack(frames, dim=0)  # (seq, C, H, W)
        if self.transform is not None:
            stacked = self.transform(stacked, self._rng)
        if self.sequence_targets:
            if self.use_absolute:
                target_source = np.asarray(sample.value_sequence, dtype=np.float32)
            else:
                target_source = np.asarray(sample.change_sequence, dtype=np.float32)
            normalized = self._apply_normalization(target_source)
            target = torch.tensor(normalized, dtype=torch.float32)
        else:
            scalar_source = sample.target_value if self.use_absolute else sample.target_change
            normalized_scalar = float(self._apply_normalization(scalar_source))
            target = torch.tensor(normalized_scalar, dtype=torch.float32)
        return stacked, target

    # ------------------------------------------------------------------
    def _load_frame(self, path: Path) -> torch.Tensor:
        return load_frame_tensor(path, grayscale=self.grayscale)

    # ------------------------------------------------------------------
    def _split_sequences(
        self, sequences_by_video: Dict[Path, List[SequenceSample]]
    ) -> Dict[str, List[SequenceSample]]:
        if self.split_mode == "sequential":
            return self._split_sequential(sequences_by_video)
        if self.split_mode == "random_chunk":
            return self._split_random(sequences_by_video)
        return self._split_video_level(sequences_by_video)

    def _split_sequential(
        self, sequences_by_video: Dict[Path, List[SequenceSample]]
    ) -> Dict[str, List[SequenceSample]]:
        train_sequences: List[SequenceSample] = []
        val_sequences: List[SequenceSample] = []
        for folder, sequences in sequences_by_video.items():
            sequences = sorted(sequences, key=lambda s: s.target_frame_index)
            split_idx = max(1, int(np.floor(len(sequences) * self.train_ratio)))
            if split_idx >= len(sequences) and len(sequences) > 1:
                split_idx = len(sequences) - 1
            train_subset = sequences[:split_idx]
            val_subset = sequences[split_idx:]
            if val_subset and self.safety_gap > 0 and train_subset:
                boundary = train_subset[-1].target_frame_index
                val_subset = [
                    seq
                    for seq in val_subset
                    if seq.target_frame_index - boundary >= self.safety_gap
                ]
            train_sequences.extend(train_subset)
            val_sequences.extend(val_subset)
        if not train_sequences and val_sequences:
            train_sequences = val_sequences[:1]
            val_sequences = val_sequences[1:]
        if not val_sequences and train_sequences:
            if len(train_sequences) > 1:
                val_sequences = train_sequences[-1:]
                train_sequences = train_sequences[:-1]
            else:
                val_sequences = list(train_sequences)
        return {"train": train_sequences, "val": val_sequences}

    def _split_random(
        self, sequences_by_video: Dict[Path, List[SequenceSample]]
    ) -> Dict[str, List[SequenceSample]]:
        all_sequences: List[SequenceSample] = []
        for sequences in sequences_by_video.values():
            all_sequences.extend(sequences)
        if not all_sequences:
            return {"train": [], "val": []}
        perm = self._rng.permutation(len(all_sequences))
        split_idx = max(1, int(np.floor(len(all_sequences) * self.train_ratio)))
        if split_idx >= len(all_sequences) and len(all_sequences) > 1:
            split_idx = len(all_sequences) - 1
        desired_val = max(1, len(all_sequences) - split_idx)
        train_sequences: List[SequenceSample] = []
        val_sequences: List[SequenceSample] = []
        train_indices: Dict[Path, Set[int]] = {}

        def _has_conflict(sample: SequenceSample) -> bool:
            if self.safety_gap <= 0:
                return False
            indices = train_indices.get(sample.video_folder)
            if not indices:
                return False
            target = sample.target_frame_index
            gap = self.safety_gap
            for idx in indices:
                if abs(target - idx) < gap:
                    return True
            return False

        for i in perm:
            sample = all_sequences[i]
            if len(val_sequences) < desired_val and not _has_conflict(sample):
                val_sequences.append(sample)
                train_indices.setdefault(sample.video_folder, set()).add(sample.target_frame_index)
            else:
                train_sequences.append(sample)
                train_indices.setdefault(sample.video_folder, set()).add(sample.target_frame_index)

        if not val_sequences and train_sequences:
            candidate = train_sequences.pop()
            val_sequences.append(candidate)

        return {"train": train_sequences, "val": val_sequences}

    def _split_video_level(
        self, sequences_by_video: Dict[Path, List[SequenceSample]]
    ) -> Dict[str, List[SequenceSample]]:
        folders = [folder for folder, sequences in sequences_by_video.items() if sequences]
        if not folders:
            return {"train": [], "val": []}
        perm = self._rng.permutation(len(folders))
        split_idx = max(1, int(np.floor(len(folders) * self.train_ratio)))
        if split_idx >= len(folders) and len(folders) > 1:
            split_idx = len(folders) - 1
        train_folders = {folders[i] for i in perm[:split_idx]}
        val_folders = {folders[i] for i in perm[split_idx:]}
        if not val_folders and len(train_folders) > 1:
            folder_to_move = next(iter(train_folders))
            train_folders.remove(folder_to_move)
            val_folders.add(folder_to_move)
        train_sequences: List[SequenceSample] = []
        val_sequences: List[SequenceSample] = []
        for folder, sequences in sequences_by_video.items():
            if folder in train_folders:
                train_sequences.extend(sequences)
            elif folder in val_folders:
                val_sequences.extend(sequences)
        if not val_sequences and train_sequences:
            if len(train_sequences) > 1:
                val_sequences = train_sequences[-1:]
                train_sequences = train_sequences[:-1]
            else:
                val_sequences = list(train_sequences)
        return {"train": train_sequences, "val": val_sequences}

    def _enforce_safety_gap(
        self,
        train_sequences: List[SequenceSample],
        val_sequences: List[SequenceSample],
        *,
        prefer_val: bool = False,
    ) -> Tuple[List[SequenceSample], List[SequenceSample]]:
        if self.safety_gap <= 0:
            return train_sequences, val_sequences

        train_lookup: Dict[Path, Dict[int, SequenceSample]] = {}
        reserved_indices: Dict[Path, Set[int]] = {}
        for seq in train_sequences:
            video = seq.video_folder
            video_map = train_lookup.setdefault(video, {})
            video_map[seq.target_frame_index] = seq
            reserved_indices.setdefault(video, set()).add(seq.target_frame_index)

        safe_val: List[SequenceSample] = []

        def _ensure_video(video: Path) -> Tuple[Dict[int, SequenceSample], Set[int]]:
            video_map = train_lookup.setdefault(video, {})
            reserved = reserved_indices.setdefault(video, set())
            return video_map, reserved

        for seq in val_sequences:
            video_map, reserved = _ensure_video(seq.video_folder)
            conflicts = {
                idx for idx in reserved if abs(seq.target_frame_index - idx) < self.safety_gap
            }
            if not conflicts:
                reserved.add(seq.target_frame_index)
                safe_val.append(seq)
                continue

            if prefer_val:
                train_conflicts = [idx for idx in conflicts if idx in video_map]
                for idx in train_conflicts:
                    video_map.pop(idx, None)
                    reserved.discard(idx)
                remaining_conflict = any(
                    abs(seq.target_frame_index - idx) < self.safety_gap for idx in reserved
                )
                if remaining_conflict:
                    video_map[seq.target_frame_index] = seq
                    reserved.add(seq.target_frame_index)
                else:
                    reserved.add(seq.target_frame_index)
                    safe_val.append(seq)
            else:
                video_map[seq.target_frame_index] = seq
                reserved.add(seq.target_frame_index)

        train_out: List[SequenceSample] = []
        for video, mapping in train_lookup.items():
            train_out.extend(sorted(mapping.values(), key=lambda s: s.target_frame_index))
        train_out.sort(key=lambda s: (s.video_folder, s.target_frame_index))
        safe_val.sort(key=lambda s: (s.video_folder, s.target_frame_index))
        return train_out, safe_val

    # ------------------------------------------------------------------
    def _apply_normalization(self, interference_change: Union[float, Sequence[float], np.ndarray]) -> np.ndarray:
        value = np.asarray(interference_change, dtype=np.float32)
        if self.scale_factor is not None:
            value = value / self.scale_factor
        if self.std == 0.0:
            raise ValueError("Normalization std must be non-zero")
        return (value - self.mean) / self.std

    # ------------------------------------------------------------------
    def _prepare_epoch_population(self) -> List[SequenceSample]:
        if not self.samples_per_epoch:
            return list(self._population)

        desired = int(self.samples_per_epoch)
        total = len(self._population)
        if total == 0:
            raise ValueError("Population is empty; cannot prepare samples")

        if not self.train_mode and desired >= total:
            return list(self._population)

        if desired <= total:
            indices = self._rng.permutation(total)[:desired]
            if not self.train_mode:
                indices = np.sort(indices)
            return [self._population[int(i)] for i in indices]

        repetitions = math.ceil(desired / total)
        base_indices = list(range(total))
        expanded: List[int] = base_indices * (repetitions - 1)
        remaining = desired - len(expanded)
        if remaining > 0:
            tail_source = self._rng.permutation(total) if self.train_mode else np.arange(total)
            expanded.extend(int(i) for i in tail_source[:remaining])
        # When train_mode and repetitions > 1 we want order to feel shuffled.
        if self.train_mode:
            expanded = list(self._rng.permutation(expanded))
        return [self._population[i] for i in expanded]


class InterferenceFrameDataset(Dataset):
    """Dataset that labels frames as ignore/non-ignore according to ignore.txt."""

    def __init__(
        self,
        video_folders: Sequence[str | Path],
        *,
        include_ignored: bool = True,
        transform: Optional[Callable[[torch.Tensor, np.random.Generator], torch.Tensor]] = None,
        seed: int = 42,
        max_negative_ratio: Optional[float] = 4.0,
        samples: Optional[Sequence[Tuple[Path, float]]] = None,
        indices: Optional[Sequence[int]] = None,
        grayscale: bool = False,
    ) -> None:
        super().__init__()
        self.video_folders = [Path(p) for p in video_folders]
        self.include_ignored = bool(include_ignored)
        self.transform = transform
        self._rng = np.random.default_rng(seed)
        self.max_negative_ratio = None if max_negative_ratio is None else max(0.0, float(max_negative_ratio))
        self.grayscale = bool(grayscale)

        if samples is None:
            collected: List[Tuple[Path, float]] = []
            for folder in self.video_folders:
                records = _load_frame_records(folder, include_ignored=self.include_ignored)
                if not records:
                    continue
                ranges = _parse_ignore_ranges(folder)
                range_idx = 0
                for rec in records:
                    while range_idx < len(ranges) and rec.frame_index > ranges[range_idx][1]:
                        range_idx += 1
                    in_ignore = False
                    if range_idx < len(ranges):
                        start, end = ranges[range_idx]
                        if start <= rec.frame_index <= end:
                            in_ignore = True
                    collected.append((rec.path, 1.0 if in_ignore else 0.0))
            if not collected:
                raise ValueError("No frame samples found for InterferenceFrameDataset.")

            if self.max_negative_ratio is not None:
                positives = [sample for sample in collected if sample[1] == 1.0]
                negatives = [sample for sample in collected if sample[1] == 0.0]
                if positives and negatives:
                    max_negatives = int(len(positives) * self.max_negative_ratio)
                    if max_negatives and len(negatives) > max_negatives:
                        selected = self._rng.permutation(len(negatives))[:max_negatives]
                        negatives = [negatives[int(i)] for i in selected]
                    collected = positives + negatives
                else:
                    collected = positives if positives else negatives
            samples = collected
        else:
            samples = [(Path(path), float(label)) for path, label in samples]

        if indices is not None:
            subset = [samples[int(i)] for i in indices]
        else:
            subset = list(samples)

        if not subset:
            raise ValueError("InterferenceFrameDataset received an empty subset of samples.")

        self._samples = subset
        self._all_samples = list(samples)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path, label = self._samples[index]
        tensor = load_frame_tensor(path, grayscale=self.grayscale)
        if self.transform is not None:
            tensor = self.transform(tensor, self._rng)
        target = torch.tensor(float(label), dtype=torch.float32)
        return tensor, target

    @property
    def all_samples(self) -> Sequence[Tuple[Path, float]]:
        return tuple(self._all_samples)


__all__ = [
    "InterferenceDataset",
    "build_sequences_for_video",
    "DEFAULT_NORMALIZATION",
    "SequenceSample",
    "InterferenceFrameDataset",
    "load_frame_tensor",
]

