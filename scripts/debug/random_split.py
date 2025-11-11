"""Utility to inspect train/val sampling for various split modes."""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.core.dataset import InterferenceDataset, SequenceSample


def _summarize(samples: Sequence[SequenceSample]) -> dict[str, object]:
    per_video = defaultdict(list)
    for sample in samples:
        per_video[sample.video_folder.name].append(sample.target_frame_index)
    summary = {
        "total": len(samples),
        "videos": {},
    }
    for video, indices in per_video.items():
        indices = sorted(indices)
        gaps = [j - i for i, j in zip(indices, indices[1:])]
        summary["videos"][video] = {
            "count": len(indices),
            "min_index": indices[0],
            "max_index": indices[-1],
            "min_gap": min(gaps) if gaps else None,
        }
    return summary


def _print_summary(name: str, samples: Sequence[SequenceSample]) -> None:
    info = _summarize(samples)
    print(f"[{name}] total={info['total']}")
    for video, meta in info["videos"].items():
        gap = meta["min_gap"]
        gap_str = "n/a" if gap is None else str(gap)
        print(
            f"  - {video}: count={meta['count']} span=({meta['min_index']}, {meta['max_index']}) min_gap={gap_str}"
        )


def build_dataset(args: argparse.Namespace, split: str, train_mode: bool) -> InterferenceDataset:
    return InterferenceDataset(
        video_folders=args.video_folders,
        samples_per_epoch=None,
        train_mode=train_mode,
        dataset_split=split,
        train_ratio=args.train_ratio,
        sequence_length=args.sequence_length,
        prediction_offset=args.prediction_offset,
        safety_gap=args.safety_gap,
        normalization=(0.0, 1.0),
        seed=args.seed,
        split_mode=args.split_mode,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect dataset splits for FunTorch5")
    parser.add_argument("video_folders", nargs="+", type=Path, help="Folders containing frames/meta.csv")
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--prediction-offset", type=int, default=0)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--safety-gap", type=int, default=10)
    parser.add_argument("--split-mode", type=str, default="random_chunk", choices=["sequential", "random_chunk", "video"])
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_ds = build_dataset(args, split="train", train_mode=True)
    val_ds = build_dataset(args, split="val", train_mode=False)
    _print_summary("train", train_ds._population)
    _print_summary("val", val_ds._population)


if __name__ == "__main__":
    main()
