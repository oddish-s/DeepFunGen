"""Diagnostic script comparing raw vs normalised interference distributions."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from scripts.core.preprocessor import VideoPreprocessor


def _load_meta(folder: Path) -> pd.DataFrame:
    meta_path = folder / "meta.csv"
    if not meta_path.exists():
        raise FileNotFoundError(f"meta.csv missing for {folder}. Run preprocessing first.")
    return pd.read_csv(meta_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse data distributions across videos")
    parser.add_argument("video_folders", nargs="+", help="One or more video folders")
    parser.add_argument("--force-preprocess", action="store_true")
    parser.add_argument("--show-plots", action="store_true")
    parser.add_argument("--target-size", type=int, nargs=2, default=[224, 224])
    args = parser.parse_args()

    preprocessor = VideoPreprocessor(target_size=tuple(args.target_size))
    meta_frames: List[pd.DataFrame] = []
    for folder_str in args.video_folders:
        folder = Path(folder_str)
        preprocessor.process_video_folder(folder, force=args.force_preprocess)
        meta = _load_meta(folder)
        meta["video_id"] = folder.name
        meta_frames.append(meta)

    combined = pd.concat(meta_frames, ignore_index=True)
    print("Global statistics:")
    print(combined[["interference_value", "interference_change"]].describe(percentiles=[0.1, 0.5, 0.9]))

    mean = combined["interference_change"].mean()
    std = combined["interference_change"].std()
    combined["normalized_change"] = (combined["interference_change"] - mean) / std
    print(f"Empirical normalization constants -> mean: {mean:.6f}, std: {std:.6f}")

    if args.show_plots:
        sns.histplot(data=combined, x="interference_change", hue="video_id", element="step", stat="density")
        plt.title("Raw interference change distribution")
        plt.figure()
        sns.histplot(data=combined, x="normalized_change", hue="video_id", element="step", stat="density")
        plt.title("Normalized interference change distribution")
        plt.show()


if __name__ == "__main__":
    main()
