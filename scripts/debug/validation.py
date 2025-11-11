"""Validate train/val splits to guard against temporal leakage."""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Set

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from scripts.config import load_config
from scripts.core.dataset import InterferenceDataset


def _frames_from_dataset(dataset: InterferenceDataset) -> Dict[Path, Set[int]]:
    mapping: Dict[Path, Set[int]] = defaultdict(set)
    for sample in dataset._population:  # type: ignore[attr-defined]
        folder = sample.curr_frame.parent.parent
        mapping[folder].add(int(sample.curr_frame.stem))
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(description="Check temporal leakage between splits")
    parser.add_argument("config", help="Config file")
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    config = load_config(args.config)
    data_cfg = config.get("data_config") or {}
    train_cfg = config.get("training_config") or {}
    folders = data_cfg.get("video_folders", [])
    if not folders:
        raise ValueError("No video folders specified in config")

    train_ds = InterferenceDataset(
        video_folders=folders,
        samples_per_epoch=None,
        train_mode=True,
        dataset_split="train",
        train_ratio=train_cfg.get("train_split", 0.8),
        chunk_size=data_cfg.get("chunk_size", 100),
        safety_gap=data_cfg.get("safety_gap", 10),
    )
    val_ds = InterferenceDataset(
        video_folders=folders,
        samples_per_epoch=None,
        train_mode=False,
        dataset_split="val",
        train_ratio=train_cfg.get("train_split", 0.8),
        chunk_size=data_cfg.get("chunk_size", 100),
        safety_gap=data_cfg.get("safety_gap", 10),
    )

    train_frames = _frames_from_dataset(train_ds)
    val_frames = _frames_from_dataset(val_ds)
    leakage_detected = False

    for folder in folders:
        folder_path = Path(folder)
        overlap = train_frames.get(folder_path, set()) & val_frames.get(folder_path, set())
        if overlap:
            leakage_detected = True
            print(f"[Leakage] {folder_path} shares {len(overlap)} frame(s) between splits")

    if not leakage_detected:
        print("No overlapping frames detected between train and val splits.")


if __name__ == "__main__":
    main()
