"""Inference script that generates predictions and optional GUI visualization."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Mapping

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np

from scripts.config import load_config
from scripts.core.inference_engine import InferenceEngine, InferenceResult
from scripts.core.postprocessing import PostProcessConfig, apply_postprocessing


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


def _resolve_checkpoint(config_name: str, explicit: str | None, config: Mapping[str, object]) -> Path:
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


def _extract_candidate_indices(processed_df: pd.DataFrame) -> list[int]:
    if "phase_marker" not in processed_df.columns:
        return []
    marker = processed_df["phase_marker"]
    mask = marker.notna()
    if not mask.any():
        return []
    if "frame_index" in processed_df.columns:
        indices = processed_df.loc[mask, "frame_index"].astype(int).tolist()
    else:
        indices = [int(idx) for idx in processed_df.index[mask]]
    return indices


def _apply_ignore_classifier(
    engine: InferenceEngine,
    inference_result: InferenceResult,
    predictions_df: pd.DataFrame,
    processed_df: pd.DataFrame,
    post_cfg: PostProcessConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    classifier = getattr(engine, "ignore_classifier", None)
    if classifier is None:
        return processed_df, predictions_df

    candidate_indices = _extract_candidate_indices(processed_df)
    scores = engine.classify_frame_indices(inference_result.video_folder, candidate_indices)
    threshold = float(getattr(classifier, "threshold", 0.5))
    if not scores:
        processed_copy = processed_df.copy()
        processed_copy["ignore_probability"] = np.nan
        processed_copy["ignore_suppressed"] = False
        processed_copy.attrs["ignore_classifier"] = {"threshold": threshold, "scores": scores, "suppressed": []}
        predictions_copy = predictions_df.copy()
        if "pre_ignore_change" not in predictions_copy.columns:
            predictions_copy["pre_ignore_change"] = predictions_copy["predicted_change"]
        predictions_copy["ignore_probability"] = np.nan
        predictions_copy["ignore_suppressed"] = False
        predictions_copy.attrs["ignore_classifier"] = processed_copy.attrs["ignore_classifier"]
        return processed_copy, predictions_copy

    updated_predictions = predictions_df.copy()
    updated_predictions["pre_ignore_change"] = updated_predictions["predicted_change"]
    updated_predictions["ignore_probability"] = np.nan
    updated_predictions["ignore_suppressed"] = False

    if "frame_index" in updated_predictions.columns:
        frame_to_row = {int(row["frame_index"]): idx for idx, row in updated_predictions.iterrows()}
    else:
        frame_to_row = {idx: idx for idx in range(len(updated_predictions))}

    for frame_idx, score in scores.items():
        row_idx = frame_to_row.get(int(frame_idx))
        if row_idx is None:
            continue
        updated_predictions.at[row_idx, "ignore_probability"] = score

    suppressed_frames = [
        frame_idx for frame_idx, score in scores.items() if score >= threshold and frame_idx in frame_to_row
    ]
    filtered_predictions = updated_predictions.copy()
    if suppressed_frames:
        row_indices = [frame_to_row[idx] for idx in suppressed_frames]
        filtered_predictions.loc[row_indices, "ignore_suppressed"] = True
        filtered_predictions.loc[row_indices, "predicted_change"] = 0.0
        print(
            f"Ignore classifier suppressed {len(suppressed_frames)} extrema "
            f"(threshold={threshold:.2f}, candidates={len(candidate_indices)})."
        )
        print("Suppression applied in-memory; cached predictions CSV remains unchanged.")
    else:
        print(
            f"Ignore classifier evaluated {len(candidate_indices)} extrema "
            f"(threshold={threshold:.2f}); no suppression triggered."
        )

    filtered_predictions.attrs["ignore_classifier"] = {
        "threshold": threshold,
        "scores": scores,
        "suppressed": suppressed_frames,
    }

    filtered_processed = apply_postprocessing(filtered_predictions, post_cfg)
    for column in ("ignore_probability", "ignore_suppressed", "pre_ignore_change"):
        if column in filtered_predictions.columns:
            filtered_processed[column] = filtered_predictions[column]
    filtered_processed.attrs.setdefault("ignore_classifier", filtered_predictions.attrs["ignore_classifier"])
    return filtered_processed, filtered_predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="FunTorch5 inference")
    parser.add_argument("config", help="Path or name of config file")
    parser.add_argument("--video-folder", required=True, help="Video folder to run inference on")
    parser.add_argument("--checkpoint", help="Model checkpoint path (defaults to models/<config_name>/best or latest)")
    parser.add_argument("--force", action="store_true", help="Recompute predictions even if cached")
    parser.add_argument("--no-gui", action="store_true", help="Skip visualization step")
    parser.add_argument("--compare-gt", action="store_true", help="Warn if meta.csv is missing; overlay loads automatically when present")
    args = parser.parse_args()

    config_path, config_name = _resolve_config(args.config)
    config = load_config(config_path)
    checkpoint_path = _resolve_checkpoint(config_name, args.checkpoint, config)
    print(f"Using checkpoint: {checkpoint_path}")
    engine = InferenceEngine(
        model_path=checkpoint_path,
        config=config,
        predictions_name=config_name,
    )
    result = engine.predict_video(args.video_folder, force=args.force)
    print(
        f"Predictions saved to {result.predictions_path} (frames={result.frame_count}, reused={result.reused_predictions})"
    )

    predictions_df = pd.read_csv(result.predictions_path)
    post_cfg = PostProcessConfig.from_mapping(config.get("postprocess_config"))
    processed_df = apply_postprocessing(predictions_df, post_cfg)
    processed_df, predictions_df = _apply_ignore_classifier(
        engine, result, predictions_df, processed_df, post_cfg
    )

    if args.no_gui:
        return

    from scripts.core.visualizer import InterferenceVisualizer  # Lazy import to avoid Tk dependency when unused

    meta_df = None
    meta_path = Path(args.video_folder) / "meta.csv"
    if meta_path.exists():
        try:
            meta_df = pd.read_csv(meta_path)
            if not meta_df.empty:
                print(f"Loaded meta.csv with {len(meta_df)} rows for ground-truth overlay.")
        except Exception as exc:
            print(f"Warning: failed to read {meta_path}: {exc}")
            meta_df = None
    elif args.compare_gt:
        print("Warning: meta.csv not found; continuing without ground truth")
    visualizer = InterferenceVisualizer(
        args.video_folder,
        predictions_df=processed_df,
        meta_df=meta_df,
        postprocess_cfg=post_cfg,
        skip_frames=engine.sequence_length,
    )
    visualizer.create_gui()


if __name__ == "__main__":
    main()
