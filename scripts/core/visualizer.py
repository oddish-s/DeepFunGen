"""Interactive visualization for FunTorch5 predictions."""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import pandas as pd
from PIL import Image
import tkinter as tk

from scripts.core.postprocessing import PostProcessConfig, apply_postprocessing


class InterferenceVisualizer:
    """Displays raw and post-processed interference curves with video frames."""

    def __init__(
        self,
        video_folder: str | Path,
        predictions_df: pd.DataFrame,
        meta_df: Optional[pd.DataFrame] = None,
        postprocess_cfg: Optional[PostProcessConfig] = None,
        skip_frames: int = 32,
    ) -> None:
        self.video_folder = Path(video_folder)
        self.frames_dir = self.video_folder / "frames"
        if not self.frames_dir.exists():
            raise FileNotFoundError(f"Frames directory missing for visualization: {self.frames_dir}")
        self.frame_paths = sorted(self.frames_dir.glob("*.jpg"))
        if not self.frame_paths:
            raise RuntimeError("No frames available for visualization")

        if "frame_index" not in predictions_df.columns or "predicted_change" not in predictions_df.columns:
            raise ValueError("predictions_df must include 'frame_index' and 'predicted_change' columns")

        self.post_cfg = postprocess_cfg or PostProcessConfig()
        self.skip_frames = max(int(skip_frames), 0)

        raw_sorted = predictions_df.sort_values("frame_index").reset_index(drop=True)
        extra_columns = {
            column: raw_sorted[column].copy()
            for column in ("ignore_probability", "ignore_suppressed", "pre_ignore_change")
            if column in raw_sorted.columns
        }
        if {"processed_value", "processed_change"}.issubset(raw_sorted.columns) and "postprocess" in raw_sorted.attrs:
            processed_sorted = raw_sorted.copy()
        else:
            processed_sorted = apply_postprocessing(raw_sorted, self.post_cfg)
            for column, values in extra_columns.items():
                processed_sorted[column] = values
        raw_sorted = processed_sorted

        raw_index = raw_sorted["frame_index"].astype(int)
        filtered_series = raw_sorted.set_index(raw_index)["predicted_change"].astype(float)
        self.raw_change_series = filtered_series[filtered_series.index >= self.skip_frames]
        if self.raw_change_series.empty:
            raise ValueError("Not enough frames after skipping initial sequence_length")

        self.original_change_series = None
        if "pre_ignore_change" in raw_sorted.columns:
            original_series = raw_sorted.set_index(raw_index)["pre_ignore_change"].astype(float)
            self.original_change_series = original_series[original_series.index >= self.skip_frames]

        post_attrs = raw_sorted.attrs
        processed_sorted = raw_sorted[raw_sorted["frame_index"] >= self.skip_frames].reset_index(drop=True)
        processed_sorted.attrs = post_attrs
        processed_index = processed_sorted["frame_index"].astype(int)
        data_index = processed_sorted.set_index(processed_index)

        self.proc_value_series = data_index["processed_value"].astype(float) if "processed_value" in data_index else pd.Series(dtype=float)
        self.phase_series = data_index["phase_marker"] if "phase_marker" in data_index else None
        self.phase_source_series = data_index["phase_source"] if "phase_source" in data_index else None
        self.raw_extrema_series = data_index["raw_extrema_marker"] if "raw_extrema_marker" in data_index else None
        if self.phase_series is not None:
            self.phase_series = self.phase_series[self.phase_series.index >= self.skip_frames]
        if self.phase_source_series is not None:
            self.phase_source_series = self.phase_source_series[self.phase_source_series.index >= self.skip_frames]
        if self.raw_extrema_series is not None:
            self.raw_extrema_series = self.raw_extrema_series[self.raw_extrema_series.index >= self.skip_frames]

        self.ignore_prob_series = None
        self.ignore_suppressed_series = None
        self.ignore_threshold = None
        if "ignore_probability" in data_index:
            self.ignore_prob_series = data_index["ignore_probability"].astype(float)
            self.ignore_prob_series = self.ignore_prob_series[self.ignore_prob_series.index >= self.skip_frames]
        if "ignore_suppressed" in data_index:
            self.ignore_suppressed_series = data_index["ignore_suppressed"].astype(bool)
            self.ignore_suppressed_series = self.ignore_suppressed_series[
                self.ignore_suppressed_series.index >= self.skip_frames
            ]
        ignore_info = post_attrs.get("ignore_classifier") if isinstance(post_attrs, dict) else None
        if isinstance(ignore_info, dict) and "threshold" in ignore_info:
            try:
                self.ignore_threshold = float(ignore_info["threshold"])
            except (TypeError, ValueError):
                self.ignore_threshold = None

        self.post_meta = post_attrs.get("postprocess", {}) if isinstance(post_attrs, dict) else {}
        raw_signal = np.asarray(self.post_meta.get("raw_signal", []), dtype=float)
        signal_frames = np.arange(raw_signal.size, dtype=int)
        mask = signal_frames >= self.skip_frames
        self.raw_signal_frames = signal_frames[mask]
        self.raw_signal = raw_signal[mask]
        smoothed_signal = np.asarray(self.post_meta.get("smoothed_signal", []), dtype=float)
        self.smoothed_signal = smoothed_signal[mask] if smoothed_signal.size == raw_signal.size else smoothed_signal
        fft_signal = np.asarray(self.post_meta.get("fft_denoised_signal", []), dtype=float)
        if fft_signal.size == raw_signal.size:
            fft_index = signal_frames
        else:
            fft_index = np.arange(fft_signal.size, dtype=int)
        self.fft_signal_series_full = pd.Series(fft_signal, index=fft_index, dtype=float) if fft_signal.size else pd.Series(dtype=float)
        if not self.fft_signal_series_full.empty:
            fft_mask = self.fft_signal_series_full.index >= self.skip_frames
            self.fft_signal_series = self.fft_signal_series_full[fft_mask]
            self.fft_signal_frames = self.fft_signal_series.index.to_numpy(dtype=int)
            self.fft_signal = self.fft_signal_series.to_numpy(dtype=float)
        else:
            self.fft_signal_series = pd.Series(dtype=float)
            self.fft_signal_frames = np.array([], dtype=int)
            self.fft_signal = np.array([], dtype=float)

        self.stage1_df = pd.DataFrame(self.post_meta.get("raw_extrema", []))
        if not self.stage1_df.empty:
            self.stage1_df = self.stage1_df.sort_values("position").reset_index(drop=True)
            self.stage1_df = self.stage1_df[self.stage1_df["position"] >= self.skip_frames].reset_index(drop=True)
            self.stage1_df["frame_index"] = self.stage1_df["position"].round().astype(int)
            self.stage1_df["raw_value"] = (
                self.raw_change_series.reindex(self.stage1_df["frame_index"]).fillna(0.0).to_numpy()
            )

        self.stage2_df = pd.DataFrame(self.post_meta.get("stage_two_extrema", []))
        if not self.stage2_df.empty:
            self.stage2_df = self.stage2_df.sort_values("position").reset_index(drop=True)
            self.stage2_df = self.stage2_df[self.stage2_df["position"] >= self.skip_frames].reset_index(drop=True)
            self.stage2_df["frame_index"] = self.stage2_df["position"].round().astype(int)
            self.stage2_df["value"] = np.where(self.stage2_df["kind"] == "peak", 100.0, 0.0)

        self.graph_points_df = pd.DataFrame(self.post_meta.get("graph_points", []))
        if not self.graph_points_df.empty:
            self.graph_points_df = self.graph_points_df.sort_values("position").reset_index(drop=True)
            self.graph_points_df = self.graph_points_df[self.graph_points_df["position"] >= self.skip_frames].reset_index(drop=True)

        if not processed_index.empty:
            self.min_frame_index = int(processed_index.min())
            self.max_frame_index = int(processed_index.max())
        else:
            self.min_frame_index = self.skip_frames
            self.max_frame_index = self.skip_frames
        self.value_min, self.value_max = 0.0, 100.0

        if meta_df is not None and {"frame_index", "interference_change"}.issubset(meta_df.columns):
            meta_sorted = meta_df.sort_values("frame_index").reset_index(drop=True)
            meta_index = meta_sorted["frame_index"].astype(int)
            meta_series = meta_sorted.set_index(meta_index)["interference_change"].astype(float)
            self.meta_change_series = meta_series[meta_series.index >= self.skip_frames]
            if "interference_value" in meta_sorted.columns:
                meta_value_series = meta_sorted.set_index(meta_index)["interference_value"].astype(float)
                self.meta_value_series = meta_value_series[meta_value_series.index >= self.skip_frames]
            else:
                self.meta_value_series = None
        else:
            self.meta_change_series = None
            self.meta_value_series = None

        # TODO 임시로 코멘트 아웃
        #bounds_primary = self.raw_change_series
        #if self.original_change_series is not None:
        #    combined_series = pd.concat(
        #        [self.raw_change_series.dropna(), self.original_change_series.dropna()], ignore_index=True
        #    )
        #    if not combined_series.empty:
        #        bounds_primary = combined_series
        #self.change_min, self.change_max = self._compute_change_bounds(bounds_primary, self.meta_change_series)
        self.change_min, self.change_max = self._compute_change_bounds(self.raw_change_series, self.meta_change_series)

        self.current_index = self.min_frame_index
        self.playback_speed = 1.0
        self.playing = False
        self.zoom_window = 300

        self.root = tk.Tk()
        self.root.title(f"Interference Visualizer - {self.video_folder.name}")

        self._build_interface()
        self._update_display(self.current_index)

    # ------------------------------------------------------------------
    def _build_interface(self) -> None:
        self.fig = plt.figure(figsize=(12, 8))
        outer = self.fig.add_gridspec(3, 2, width_ratios=[1.2, 3], height_ratios=[3, 2, 2])

        self.ax_video = self.fig.add_subplot(outer[0, 0])
        self.ax_indicator = self.fig.add_subplot(outer[1:, 0])

        right_gs = outer[:, 1].subgridspec(4, 1, height_ratios=[1, 1, 1, 1])
        self.ax_stage1 = self.fig.add_subplot(right_gs[0])
        self.ax_stage2 = self.fig.add_subplot(right_gs[1])
        self.ax_stage3 = self.fig.add_subplot(right_gs[2])
        self.ax_stage4 = self.fig.add_subplot(right_gs[3])

        self.ax_video.axis("off")
        self.fig.subplots_adjust(hspace=0.45)

        self.ax_indicator.set_xlim(0, 1)
        self.ax_indicator.set_xticks([])
        self.ax_indicator.set_ylabel("Raw change")
        self.ax_indicator.axhline(0, color="gray", linestyle="--", linewidth=1)
        self.ax_indicator_value = self.ax_indicator.twinx()
        self.ax_indicator_value.set_ylabel("Processed value")
        self.ax_indicator_value.set_ylim(self.value_min, self.value_max)
        self.ax_indicator_value.set_xticks([])
        self.raw_bar = self.ax_indicator.bar([0.25], [0.0], width=0.2, color="tab:blue")[0]
        self.proc_bar = self.ax_indicator_value.bar([0.75], [50.0], width=0.2, color="tab:green")[0]
        self.ax_indicator.text(0.25, 0.02, "Raw", transform=self.ax_indicator.transAxes, ha="center", va="bottom")
        self.ax_indicator_value.text(0.75, 0.02, "Processed", transform=self.ax_indicator_value.transAxes, ha="center", va="bottom")

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect("button_press_event", self._handle_graph_click)

        controls = tk.Frame(self.root)
        controls.pack(fill=tk.X)

        tk.Button(controls, text="<<", command=self._step_backward).pack(side=tk.LEFT, padx=2, pady=4)
        self.play_button = tk.Button(controls, text="Play", command=self._toggle_playback)
        self.play_button.pack(side=tk.LEFT, padx=2, pady=4)
        tk.Button(controls, text=">>", command=self._step_forward).pack(side=tk.LEFT, padx=2, pady=4)
        tk.Button(controls, text="Quit", command=self._on_close).pack(side=tk.LEFT, padx=(12, 2))

        tk.Label(controls, text="Speed").pack(side=tk.LEFT, padx=(12, 2))
        self.speed_var = tk.DoubleVar(value=1.0)
        tk.Scale(
            controls,
            from_=0.25,
            to=4.0,
            resolution=0.25,
            orient=tk.HORIZONTAL,
            variable=self.speed_var,
            command=lambda _=None: self._update_speed(),
            length=160,
        ).pack(side=tk.LEFT, padx=2)

        tk.Label(controls, text="Seek").pack(side=tk.LEFT, padx=(12, 2))
        self.slider = tk.Scale(
            controls,
            from_=self.min_frame_index,
            to=max(self.max_frame_index, self.min_frame_index),
            orient=tk.HORIZONTAL,
            length=360,
            command=self._seek,
        )
        self.slider.pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)

        self.status_var = tk.StringVar(value="Ready")
        tk.Label(self.root, textvariable=self.status_var, anchor="w").pack(fill=tk.X, padx=6, pady=(0, 6))

    # ------------------------------------------------------------------
    def _play_loop(self) -> None:
        while self.playing:
            next_idx = self.current_index + 1
            if next_idx > self.max_frame_index:
                next_idx = self.min_frame_index
            self.root.after(0, lambda idx=next_idx: self._update_display(idx))
            time.sleep(max(0.01, 0.1 / self.playback_speed))

    # ------------------------------------------------------------------
    def _handle_graph_click(self, event) -> None:
        valid_axes = {self.ax_stage1, self.ax_stage2, self.ax_stage3, self.ax_stage4}
        if event.inaxes not in valid_axes:
            return
        if event.xdata is None:
            return
        self._update_display(int(round(event.xdata)))

    # ------------------------------------------------------------------
    def _update_graphs(self, frame_idx: int) -> None:
        for ax in (self.ax_stage1, self.ax_stage2, self.ax_stage3, self.ax_stage4):
            ax.cla()

        # Stage 1: raw change (full range)
        if self.raw_signal.size:
            self.ax_stage1.plot(self.raw_signal_frames, self.raw_signal, color="tab:blue", linewidth=1.2, label="Raw")
        if self.fft_signal.size:
            self.ax_stage1.plot(
                self.fft_signal_frames,
                self.fft_signal,
                color="tab:orange",
                linewidth=1.0,
                linestyle="--",
                alpha=0.9,
                label="FFT denoised",
            )
        else:
            raw_series = self.raw_change_series.dropna()
            if not raw_series.empty:
                self.ax_stage1.plot(raw_series.index, raw_series.values, color="tab:blue", linewidth=1.2)
        if self.original_change_series is not None:
            original_series = self.original_change_series.dropna()
            if not original_series.empty:
                self.ax_stage1.plot(
                    original_series.index,
                    original_series.values,
                    color="tab:gray",
                    linestyle=":",
                    linewidth=1.0,
                    alpha=0.7,
                    label="Pre-ignore change",
                )
        self.ax_stage1.axvline(frame_idx, color="tab:gray", linestyle="-.", linewidth=1.0)
        self.ax_stage1.set_xlim(self.min_frame_index, max(self.max_frame_index, self.min_frame_index + 1))
        self.ax_stage1.set_ylabel("Raw change")
        handles, labels = self.ax_stage1.get_legend_handles_labels()
        if handles:
            self.ax_stage1.legend(loc="upper right")

        # Stage 2: raw change (zoom)
        half_window = max(self.zoom_window // 2, 1)
        start = max(frame_idx - half_window, self.min_frame_index)
        end = min(frame_idx + half_window, self.max_frame_index)
        window_frames = np.arange(start, end + 1, dtype=int) if end >= start else np.array([], dtype=int)
        raw_window = self.raw_change_series.reindex(window_frames).dropna()
        y_values: list[float] = []
        if not raw_window.empty:
            self.ax_stage2.plot(raw_window.index, raw_window.values, color="tab:blue", linewidth=1.2)
            y_values.extend(raw_window.values.tolist())
        if self.original_change_series is not None and window_frames.size:
            original_window = self.original_change_series.reindex(window_frames).dropna()
            if not original_window.empty:
                self.ax_stage2.plot(
                    original_window.index,
                    original_window.values,
                    color="tab:gray",
                    linestyle=":",
                    linewidth=1.0,
                    alpha=0.7,
                )
                y_values.extend(original_window.values.tolist())
        if not self.fft_signal_series_full.empty and window_frames.size:
            fft_window = self.fft_signal_series_full.reindex(window_frames).dropna()
            if not fft_window.empty:
                self.ax_stage2.plot(
                    fft_window.index,
                    fft_window.values,
                    color="tab:orange",
                    linewidth=1.1,
                    linestyle="--",
                    alpha=0.9,
                )
                y_values.extend(fft_window.values.tolist())
        window_stage2 = self.stage2_df[
            (self.stage2_df["position"] >= start) & (self.stage2_df["position"] <= end)
        ] if not self.stage2_df.empty else pd.DataFrame()
        if not window_stage2.empty:
            styles = {
                ("peak", "original"): ("o", "tab:red"),
                ("trough", "original"): ("o", "tab:blue"),
                ("peak", "inserted"): ("^", "tab:red"),
                ("trough", "inserted"): ("v", "tab:blue"),
                ("peak", "merged"): ("s", "tab:red"),
                ("trough", "merged"): ("s", "tab:blue"),
            }
            for (kind, origin), (marker, color) in styles.items():
                subset = window_stage2[
                    (window_stage2["kind"] == kind) & (window_stage2["origin"] == origin)
                ]
                if subset.empty:
                    continue
                positions = subset["position"].round().astype(int)
                if not self.fft_signal_series_full.empty:
                    marker_series = self.fft_signal_series_full.reindex(positions)
                else:
                    marker_series = self.raw_change_series.reindex(positions)
                marker_values = marker_series.fillna(0.0)
                y_values.extend(marker_values.values.tolist())
                self.ax_stage2.scatter(
                    subset["position"],
                    marker_values,
                    marker=marker,
                    s=50,
                    color=color,
                    edgecolors="black",
                    linewidths=0.6,
                )
        if self.ignore_suppressed_series is not None and window_frames.size:
            suppressed_window = self.ignore_suppressed_series.reindex(window_frames).fillna(False)
            suppressed_indices = suppressed_window[suppressed_window].index.astype(int)
            if suppressed_indices.size:
                if self.original_change_series is not None:
                    suppressed_values = self.original_change_series.reindex(suppressed_indices).fillna(
                        self.raw_change_series.reindex(suppressed_indices)
                    )
                else:
                    suppressed_values = self.raw_change_series.reindex(suppressed_indices)
                suppressed_values = suppressed_values.fillna(0.0)
                y_values.extend(suppressed_values.values.tolist())
                existing_labels = set(self.ax_stage2.get_legend_handles_labels()[1])
                label = "Suppressed (ignore)" if "Suppressed (ignore)" not in existing_labels else None
                self.ax_stage2.scatter(
                    suppressed_indices,
                    suppressed_values.values,
                    marker="X",
                    s=90,
                    color="black",
                    edgecolors="white",
                    linewidths=1.0,
                    label=label,
                )
        self.ax_stage2.axvline(frame_idx, color="tab:gray", linestyle="-.", linewidth=1.0)
        self.ax_stage2.set_xlim(start, end if end > start else start + 1)
        if y_values:
            y_min = min(y_values)
            y_max = max(y_values)
            if y_min == y_max:
                pad = abs(y_min) * 0.1
                y_min -= pad
                y_max += pad
            else:
                margin = (y_max - y_min) * 0.1
                y_min -= margin
                y_max += margin
            self.ax_stage2.set_ylim(y_min, y_max)
        else:
            self.ax_stage2.set_ylim(self.change_min, self.change_max)
        self.ax_stage2.set_ylabel("Raw change")

        # Stage 3: alternating extrema markers (zoomed)
        window_stage2 = self.stage2_df[
            (self.stage2_df["position"] >= start) & (self.stage2_df["position"] <= end)
        ] if not self.stage2_df.empty else pd.DataFrame()
        if not window_stage2.empty:
            self.ax_stage3.plot(window_stage2["position"], window_stage2["value"], color="tab:purple", linewidth=1.5)
            styles = {
                ("peak", "original"): ("o", "tab:red", "Peak (original)"),
                ("trough", "original"): ("o", "tab:blue", "Trough (original)"),
                ("peak", "inserted"): ("^", "tab:red", "Peak (inserted)"),
                ("trough", "inserted"): ("v", "tab:blue", "Trough (inserted)"),
                ("peak", "merged"): ("s", "tab:red", "Peak (merged)"),
                ("trough", "merged"): ("s", "tab:blue", "Trough (merged)"),
            }
            legend_used = set()
            for key, (marker, color, label) in styles.items():
                kind, origin = key
                subset = window_stage2[
                    (window_stage2["kind"] == kind) & (window_stage2["origin"] == origin)
                ]
                if subset.empty:
                    continue
                plot_label = label if label not in legend_used else None
                self.ax_stage3.scatter(
                    subset["position"],
                    subset["value"],
                    marker=marker,
                    s=60,
                    color=color,
                    edgecolors="black",
                    linewidths=0.6,
                    label=plot_label,
                )
                legend_used.add(label)
            if self.ignore_suppressed_series is not None:
                frame_flags = self.ignore_suppressed_series.reindex(window_stage2["frame_index"].astype(int)).fillna(False)
                suppressed_subset = window_stage2[frame_flags.to_numpy(dtype=bool)]
                if not suppressed_subset.empty:
                    label = "Suppressed (ignore)" if "Suppressed (ignore)" not in legend_used else None
                    self.ax_stage3.scatter(
                        suppressed_subset["position"],
                        suppressed_subset["value"],
                        marker="X",
                        s=90,
                        color="black",
                        edgecolors="white",
                        linewidths=1.0,
                        label=label,
                    )
                    legend_used.add("Suppressed (ignore)")
            handles, labels = self.ax_stage3.get_legend_handles_labels()
            if handles:
                unique = {}
                for handle, label in zip(handles, labels):
                    if label:
                        unique.setdefault(label, handle)
                if unique:
                    self.ax_stage3.legend(
                        unique.values(),
                        unique.keys(),
                        loc="upper center",
                        bbox_to_anchor=(0.5, -0.15),
                        ncol=6,
                    )
        else:
            self.ax_stage3.text(
                0.5,
                0.5,
                "No extrema in window",
                transform=self.ax_stage3.transAxes,
                ha="center",
                va="center",
                color="0.4",
            )
        self.ax_stage3.set_xlim(start, end if end > start else start + 1)
        self.ax_stage3.set_ylim(-10, 110)
        self.ax_stage3.set_ylabel("Value (0-100)")
        self.ax_stage3.axvline(frame_idx, color="tab:gray", linestyle="-.", linewidth=1.0)

        # Stage 4: processed curve (zoom)
        window_frames = np.arange(start, end + 1, dtype=int) if end >= start else np.array([], dtype=int)
        if window_frames.size:
            window_series = self.proc_value_series.reindex(window_frames).dropna()
            if not window_series.empty:
                self.ax_stage4.plot(window_series.index, window_series.values, color="tab:green", linewidth=1.3)
            if self.meta_value_series is not None:
                meta_value_window = self.meta_value_series.reindex(window_frames).dropna()
                if not meta_value_window.empty:
                    self.ax_stage4.plot(
                        meta_value_window.index,
                        meta_value_window.values,
                        color="tab:orange",
                        linestyle="--",
                        linewidth=1.2,
                        label="Ground truth (meta)",
                    )
            if not self.graph_points_df.empty:
                window_points = self.graph_points_df[
                    (self.graph_points_df["position"] >= start)
                    & (self.graph_points_df["position"] <= end)
                ]
                if not window_points.empty:
                    styles = {
                        ("peak", "original"): ("o", "tab:red", "Peak (original)"),
                        ("trough", "original"): ("o", "tab:blue", "Trough (original)"),
                        ("peak", "inserted"): ("^", "tab:red", "Peak (inserted)"),
                        ("trough", "inserted"): ("v", "tab:blue", "Trough (inserted)"),
                        ("peak", "merged"): ("s", "tab:red", "Peak (merged)"),
                        ("trough", "merged"): ("s", "tab:blue", "Trough (merged)"),
                        ("boosted", "boosted"): ("D", "goldenrod", "Peak (boosted)"),
                    }
                    for (label_name, origin), (marker, color, label) in styles.items():
                        subset = window_points[
                            (window_points["label"] == label_name)
                            & (window_points["origin"] == origin)
                        ]
                        if subset.empty:
                            continue
                        self.ax_stage4.scatter(
                            subset["position"],
                            subset["value"],
                            marker=marker,
                            s=60,
                            color=color,
                            edgecolors="black",
                            linewidths=0.6,
                            label=label,
                        )
                    for (origin, direction), (marker, color, label) in {
                        ("max_slope", "to_peak"): ("X", "tab:purple", "Max slope (to peak)"),
                        ("max_slope", "to_trough"): ("X", "slateblue", "Max slope (to trough)"),
                        ("min_slope", "to_peak"): ("+", "tab:orange", "Min slope (to peak)"),
                        ("min_slope", "to_trough"): ("+", "tab:brown", "Min slope (to trough)"),
                    }.items():
                        subset = window_points[
                            (window_points["label"] == "intermediate")
                            & (window_points["origin"] == origin)
                            & (window_points["direction"].fillna("") == direction)
                        ]
                        if subset.empty:
                            continue
                        self.ax_stage4.scatter(
                            subset["position"],
                            subset["value"],
                            marker=marker,
                            s=70,
                            color=color,
                            linewidths=1.2,
                            label=label,
                        )
            if self.ignore_suppressed_series is not None and window_frames.size:
                suppressed_window = self.ignore_suppressed_series.reindex(window_frames).fillna(False)
                suppressed_frames = suppressed_window[suppressed_window].index.astype(int)
                if suppressed_frames.size:
                    suppressed_values = self.proc_value_series.reindex(suppressed_frames).fillna(0.0)
                    existing_labels = set(self.ax_stage4.get_legend_handles_labels()[1])
                    label = "Suppressed (ignore)" if "Suppressed (ignore)" not in existing_labels else None
                    self.ax_stage4.scatter(
                        suppressed_frames,
                        suppressed_values.values,
                        marker="X",
                        s=90,
                        color="black",
                        edgecolors="white",
                        linewidths=1.0,
                        label=label,
                    )
        handles, labels = self.ax_stage4.get_legend_handles_labels()
        if handles:
            unique = {}
            for handle, label in zip(handles, labels):
                if label:
                    unique.setdefault(label, handle)
            if unique:
                self.ax_stage4.legend(
                    unique.values(),
                    unique.keys(),
                    loc="upper center",
                    bbox_to_anchor=(0.5, -0.15),
                    ncol=6,
                )
        self.ax_stage4.set_xlim(start, end if end > start else start + 1)
        self.ax_stage4.set_ylim(0, 100)
        self.ax_stage4.set_ylabel("Value (0-100)")
        self.ax_stage4.axvline(frame_idx, color="tab:gray", linestyle="-.", linewidth=1.0)

    # ------------------------------------------------------------------
    def _update_display(self, frame_idx: int) -> None:
        frame_idx = max(self.min_frame_index, min(frame_idx, self.max_frame_index))
        self.current_index = frame_idx
        self.slider.set(frame_idx)

        frame = self._load_frame(frame_idx)
        self.ax_video.clear()
        self.ax_video.imshow(frame)
        self.ax_video.set_title(f"Frame {frame_idx}/{self.max_frame_index}")
        self.ax_video.axis("off")

        self._update_graphs(frame_idx)
        self._update_indicator(frame_idx)

        raw_change = self._get_raw_change(frame_idx)
        proc_value = self._get_proc_value(frame_idx)
        original_change = self._get_original_change(frame_idx)
        ignore_prob = self._get_ignore_probability(frame_idx)
        suppressed = self._is_ignore_suppressed(frame_idx)
        status_parts = [f"Frame: {frame_idx}", f"Raw change: {raw_change:.6f}", f"Value: {proc_value:.2f}"]
        if original_change is not None:
            status_parts.append(f"Pre-ignore: {original_change:.6f}")
            delta = raw_change - original_change
            if abs(delta) > 1e-6:
                status_parts.append(f"Δignore: {delta:.6f}")
        gt_change = self._get_meta_change(frame_idx)
        if gt_change is not None:
            status_parts.append(f"GT change: {gt_change:.6f}")
            status_parts.append(f"Abs err: {abs(raw_change - gt_change):.6f}")
        if ignore_prob is not None:
            status_parts.append(f"Ignore prob: {ignore_prob:.3f}")
            if self.ignore_threshold is not None:
                status_parts.append(f"Thresh: {self.ignore_threshold:.2f}")
        if suppressed:
            status_parts.append("Suppressed: yes")
        elif ignore_prob is not None:
            status_parts.append("Suppressed: no")
        self.status_var.set(" | ".join(status_parts))

        self.canvas.draw_idle()

    # ------------------------------------------------------------------
    def _step_forward(self) -> None:
        self._update_display(self.current_index + 1)

    # ------------------------------------------------------------------
    def _step_backward(self) -> None:
        self._update_display(self.current_index - 1)

    # ------------------------------------------------------------------
    def _toggle_playback(self) -> None:
        self.playing = not self.playing
        self.play_button.configure(text="Pause" if self.playing else "Play")
        if self.playing:
            threading.Thread(target=self._play_loop, daemon=True).start()

    # ------------------------------------------------------------------
    def _update_speed(self) -> None:
        self.playback_speed = max(0.25, float(self.speed_var.get()))

    # ------------------------------------------------------------------
    def _seek(self, value: str) -> None:
        try:
            idx = int(float(value))
        except ValueError:
            return
        self._update_display(idx)

    # ------------------------------------------------------------------
    def _on_close(self) -> None:
        self.playing = False
        self.root.quit()

    # ------------------------------------------------------------------
    def _load_frame(self, frame_idx: int) -> np.ndarray:
        frame_idx = max(self.min_frame_index, min(frame_idx, self.max_frame_index))
        img = Image.open(self.frame_paths[frame_idx])
        return np.asarray(img)
    
    def create_gui(self) -> None:
        """Run the Tkinter main loop."""
        self.root.bind("<Left>", lambda _: self._step_backward())
        self.root.bind("<Right>", lambda _: self._step_forward())
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.mainloop()


    # ------------------------------------------------------------------
    def _update_indicator(self, frame_idx: int) -> None:
        raw_change = self._get_raw_change(frame_idx)
        proc_value = self._get_proc_value(frame_idx)

        original_change = self._get_original_change(frame_idx)
        ignore_prob = self._get_ignore_probability(frame_idx)
        suppressed = self._is_ignore_suppressed(frame_idx)

        raw_height = abs(raw_change)
        raw_base = raw_change if raw_change < 0 else 0.0
        self.raw_bar.set_height(raw_height)
        self.raw_bar.set_y(raw_base)
        bar_color = "tab:blue" if raw_change >= 0 else "tab:red"
        if suppressed:
            bar_color = "dimgray"
        self.raw_bar.set_color(bar_color)

        self.proc_bar.set_height(proc_value)
        self.proc_bar.set_y(0)

        self._apply_indicator_limits()
        self.ax_indicator_value.set_ylim(self.value_min, self.value_max)
        title_parts = [f"Pred change: {raw_change:.3f}"]
        if original_change is not None:
            title_parts.append(f"Pre-ignore: {original_change:.3f}")
        if suppressed:
            title_parts.append("[suppressed]")
        title_parts.append(f"Value: {proc_value:.2f}")
        if ignore_prob is not None:
            title_parts.append(f"Ignore prob: {ignore_prob:.3f}")
        self.ax_indicator.set_title(" | ".join(title_parts))

    # ------------------------------------------------------------------
    def _apply_indicator_limits(self) -> None:
        lower, upper = self.change_min, self.change_max
        if lower >= upper:
            span = max(abs(lower), 1.0)
            lower = -span
            upper = span
        self.ax_indicator.set_ylim(lower, upper)

    # ------------------------------------------------------------------
    def _compute_change_bounds(
        self,
        primary: pd.Series,
        secondary: Optional[pd.Series] = None,
    ) -> tuple[float, float]:
        series_list = [primary.dropna()]
        if secondary is not None:
            series_list.append(secondary.dropna())
        combined = pd.concat(series_list) if series_list else pd.Series(dtype=float)
        if combined.empty:
            return -1.0, 1.0
        min_val = float(combined.min())
        max_val = float(combined.max())
        max_abs = max(abs(min_val), abs(max_val), 1.0)
        max_abs *= 1.05
        return -max_abs, max_abs

    # ------------------------------------------------------------------
    def _get_raw_change(self, frame_idx: int) -> float:
        value = self.raw_change_series.get(frame_idx)
        if value is None or pd.isna(value):
            return 0.0
        return float(value)

    # ------------------------------------------------------------------
    def _get_meta_change(self, frame_idx: int) -> Optional[float]:
        if self.meta_change_series is None:
            return None
        value = self.meta_change_series.get(frame_idx)
        if value is None or pd.isna(value):
            return None
        return float(value)

    # ------------------------------------------------------------------
    def _get_proc_value(self, frame_idx: int) -> float:
        value = self.proc_value_series.get(frame_idx)
        if value is None or pd.isna(value):
            return 0.0
        return float(value)

    # ------------------------------------------------------------------
    def _get_original_change(self, frame_idx: int) -> Optional[float]:
        if self.original_change_series is None:
            return None
        value = self.original_change_series.get(frame_idx)
        if value is None or pd.isna(value):
            return None
        return float(value)

    # ------------------------------------------------------------------
    def _get_ignore_probability(self, frame_idx: int) -> Optional[float]:
        if self.ignore_prob_series is None:
            return None
        value = self.ignore_prob_series.get(frame_idx)
        if value is None or pd.isna(value):
            return None
        return float(value)

    # ------------------------------------------------------------------
    def _is_ignore_suppressed(self, frame_idx: int) -> bool:
        if self.ignore_suppressed_series is None:
            return False
        value = self.ignore_suppressed_series.get(frame_idx)
        if value is None or pd.isna(value):
            return False
        return bool(value)


__all__ = ["InterferenceVisualizer"]
