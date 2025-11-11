from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from typing import Sequence

from scripts.core.prediction_compare import (
    PredictionComparisonViewer,
    load_meta_series,
    load_prediction_series,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare raw prediction CSVs produced by different models for the same video folder."
        )
    )
    parser.add_argument(
        "video_folder",
        type=Path,
        help="Path to the processed video folder containing predictions-*.csv files",
    )
    parser.add_argument(
        "--include",
        nargs="+",
        help=(
            "Optional substrings to filter prediction files. Match is performed against the file "
            "stem (predictions-<name>)."
        ),
    )
    parser.add_argument(
        "--metric",
        choices=["predicted_change", "predicted_value"],
        default="predicted_change",
        help="Prediction column to plot (default: predicted_change)",
    )
    parser.add_argument(
        "--window-frames",
        type=int,
        default=300,
        help="Number of frames shown in the zoomed panel (default: 300)",
    )
    parser.add_argument(
        "--skip-frames",
        type=int,
        default=32,
        help="Number of leading frames to ignore when plotting (default: 32)",
    )
    parser.add_argument(
        "--center",
        type=int,
        help="Initial center frame for the zoomed panel",
    )
    parser.add_argument(
        "--no-meta",
        action="store_true",
        help="Disable plotting meta.csv interference values even if available",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    series = load_prediction_series(
        args.video_folder, include=args.include, metric=args.metric, skip_frames=args.skip_frames
    )
    meta = None if args.no_meta else load_meta_series(args.video_folder, skip_frames=args.skip_frames)
    viewer = PredictionComparisonViewer(
        series,
        meta=meta,
        window_frames=args.window_frames,
        initial_center=args.center,
        metric_label=args.metric,
        skip_frames=args.skip_frames,
        video_folder=args.video_folder
    )
    viewer.show()


if __name__ == "__main__":
    main()
