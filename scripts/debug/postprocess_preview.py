"""Apply prominence-based post-processing to an existing predictions CSV."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.core.postprocessing import PostProcessConfig, apply_postprocessing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview post-processed interference values")
    parser.add_argument("predictions", type=Path, help="Path to predictions.csv")
    parser.add_argument("--frame-rate", type=float, default=30.0)
    parser.add_argument("--smooth-window", type=int, default=1, help="Rolling mean window for extrema detection")
    parser.add_argument("--prominence-ratio", type=float, default=0.3, help="Ratio of signal range used as prominence threshold")
    parser.add_argument("--min-prominence", type=float, help="Absolute minimum prominence")
    parser.add_argument("--prominence-window", type=int, default=301, help="Window (frames) for local prominence range")
    parser.add_argument("--min-slope", type=float, default=2.0, help="Minimum slope between extrema (value per frame)")
    parser.add_argument("--max-slope", type=float, default=10.0, help="Maximum slope between extrema (value per frame)")
    parser.add_argument("--merge-threshold-ms", type=float, default=120.0, help="Distance threshold in milliseconds for merging triplets")
    parser.add_argument("--output", type=Path, help="Optional path to save processed CSV")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.predictions.exists():
        raise FileNotFoundError(f"Predictions file not found: {args.predictions}")

    df = pd.read_csv(args.predictions)
    cfg = PostProcessConfig(
        frame_rate=args.frame_rate,
        smooth_window_frames=max(1, int(args.smooth_window)),
        prominence_ratio=args.prominence_ratio,
        min_prominence=args.min_prominence,
        prominence_window_frames=max(1, int(args.prominence_window)),
        min_slope=args.min_slope,
        max_slope=args.max_slope,
        merge_threshold_ms=args.merge_threshold_ms,
    )
    processed = apply_postprocessing(df, cfg)
    if args.output:
        processed.to_csv(args.output, index=False)
        print(f"Processed predictions saved to {args.output}")
    else:
        print(processed.head())


if __name__ == "__main__":
    main()
