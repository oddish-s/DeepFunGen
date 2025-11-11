"""Standalone utilities for comparing multiple prediction CSV files."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image
import numpy as np
import pandas as pd


@dataclass
class PredictionSeries:
    label: str
    frame_index: np.ndarray
    timestamps: np.ndarray
    values: np.ndarray


@dataclass
class MetaSeries:
    frame_index: np.ndarray
    timestamps: np.ndarray
    values: np.ndarray


class PredictionComparisonViewer:
    """Interactive matplotlib viewer for comparing raw prediction series."""

    def __init__(
        self,
        series: Sequence[PredictionSeries],
        meta: Optional[MetaSeries] = None,
        window_frames: int = 300,
        initial_center: Optional[int] = None,
        metric_label: str = "predicted_change",
        skip_frames: int = 32,
        video_folder: str = ""
    ) -> None:
        if not series:
            raise ValueError("At least one prediction series is required for comparison")

        self.skip_frames = max(int(skip_frames), 0)
        self.series = [self._trim_series(s) for s in series]
        if not self.series:
            raise ValueError("No prediction data remains after skipping initial frames")

        self.meta = self._trim_meta(meta)
        self.window = max(int(window_frames), 1)
        self.metric_label = metric_label
        self.symmetric = False
        self.colors = self._generate_colors(len(self.series))

        self.reference = self.series[0]
        self.min_frame = int(min(s.frame_index[0] for s in self.series if s.frame_index.size))
        self.max_frame = int(max(s.frame_index[-1] for s in self.series if s.frame_index.size))
        if self.min_frame > self.max_frame:
            raise ValueError("Invalid frame bounds after trimming")

        if initial_center is None:
            self.center = (self.min_frame + self.max_frame) // 2
        else:
            self.center = int(np.clip(initial_center, self.min_frame, self.max_frame))

        self.global_min, self.global_max = self._compute_bounds(
            [s.values for s in self.series], symmetric=self.symmetric
        )
        if self.meta is not None:
            self.meta_min, self.meta_max = self._compute_bounds([self.meta.values], symmetric=False)
        else:
            self.meta_min = self.meta_max = None

        self.series_bounds = [
            self._compute_bounds([series.values], symmetric=self.symmetric) for series in self.series
        ]

        self.fig = plt.figure(figsize=(16, 8))
        gs = self.fig.add_gridspec(2, 2, width_ratios=[3, 1], hspace=0.35, wspace=0.2)
        
        self.ax_full = self.fig.add_subplot(gs[0, 0])
        self.ax_zoom = self.fig.add_subplot(gs[1, 0])
        self.ax_video = self.fig.add_subplot(gs[:, 1])

        self.ax_full.grid(True, linewidth=0.5, alpha=0.3)
        self.ax_zoom.grid(True, linewidth=0.5, alpha=0.3)
        self.ax_zoom.set_xlabel("Frame index")
        
        # y축 눈금 숨기기
        self.ax_full.set_yticks([])
        self.ax_zoom.set_yticks([])

        self.ax_full_meta = None
        self.ax_zoom_meta = None

        self.video_folder = video_folder
        self.frames_dir = self.video_folder / "frames"
        if not self.frames_dir.exists():
            raise FileNotFoundError(f"Frames directory missing for visualization: {self.frames_dir}")
        self.frame_paths = sorted(self.frames_dir.glob("*.jpg"))
        if not self.frame_paths:
            raise RuntimeError("No frames available for visualization")

        self._init_full_plot()
        self._connect_events()
        self._update_zoom()

    # ------------------------------------------------------------------
    def show(self) -> None:
        plt.show()

    # ------------------------------------------------------------------
    def _connect_events(self) -> None:
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    # ------------------------------------------------------------------
    def _init_full_plot(self) -> None:
        self.full_lines = []
        self.zoom_lines = []
        self.full_normalized = []
        self.zoom_normalized = []

        legend_handles: List[Line2D] = []
        legend_labels: List[str] = []

        for idx, series in enumerate(self.series):
            color = self.colors[idx]
            
            # 정규화된 값 계산 (0~1 범위) - 실제 min/max 사용
            y_min = float(series.values.min()) if series.values.size > 0 else 0.0
            y_max = float(series.values.max()) if series.values.size > 0 else 1.0
            y_range = y_max - y_min
            if y_range > 0:
                normalized = (series.values - y_min) / y_range
            else:
                normalized = np.full_like(series.values, 0.5)
            
            self.full_normalized.append(normalized)
            
            line_full, = self.ax_full.plot(
                series.frame_index,
                normalized,
                label=series.label,
                color=color,
                linewidth=1.5,
            )

            line_zoom, = self.ax_zoom.plot([], [], color=color, label=series.label, linewidth=1.5)

            self.full_lines.append(line_full)
            self.zoom_lines.append(line_zoom)
            legend_handles.append(line_full)
            legend_labels.append(series.label or f"series {idx + 1}")

        # full plot 범위 설정
        self.ax_full.set_xlim(self.min_frame, self.max_frame)
        self.ax_full.set_ylim(0, 1)
        self.ax_zoom.set_ylim(0, 1)

        if self.meta is not None:
            # meta도 정규화 - 실제 min/max 사용
            if self.meta.values.size > 0:
                meta_min = float(self.meta.values.min())
                meta_max = float(self.meta.values.max())
                meta_range = meta_max - meta_min
                if meta_range > 0:
                    meta_normalized = (self.meta.values - meta_min) / meta_range
                else:
                    meta_normalized = np.full_like(self.meta.values, 0.5)
            else:
                meta_normalized = np.array([])
            
            self.meta_normalized = meta_normalized
            
            (self.meta_full_line,) = self.ax_full.plot(
                self.meta.frame_index,
                meta_normalized,
                color='tab:gray',
                linestyle='--',
                label='meta value',
                linewidth=1.0,
                alpha=0.7,
            )

            (self.meta_zoom_line,) = self.ax_zoom.plot(
                [], [], color='tab:gray', linestyle='--', label='meta value', linewidth=1.0, alpha=0.7
            )

            legend_handles.append(self.meta_full_line)
            legend_labels.append('meta value')
        else:
            self.meta_full_line = None
            self.meta_zoom_line = None

        if legend_handles:
            self.ax_full.legend(legend_handles, legend_labels, loc='upper right')

        self.full_cursor = self.ax_full.axvline(self.center, color='k', linestyle='--', linewidth=1.0)
        self.zoom_cursor = self.ax_zoom.axvline(self.center, color='k', linestyle='--', linewidth=1.0)
        self.status_text = self.fig.text(0.01, 0.01, "", fontsize=9)
        
        # 초기 프레임 표시
        self.ax_video.axis("off")

    # ------------------------------------------------------------------
    def _load_frame(self, frame_idx: int) -> np.ndarray:
        frame_idx = max(self.min_frame, min(frame_idx, self.max_frame))
        img = Image.open(self.frame_paths[frame_idx])
        return np.asarray(img)

    # ------------------------------------------------------------------
    def _on_click(self, event) -> None:
        if event.inaxes not in {self.ax_full, self.ax_zoom}:
            return
        if event.xdata is None:
            return
        frame = int(round(event.xdata))
        self.center = int(np.clip(frame, self.min_frame, self.max_frame))
        self._update_zoom()

    # ------------------------------------------------------------------
    def _on_key(self, event) -> None:
        if event.key in {"left", "right"}:
            step = 1
            if event.key == "left":
                self.center -= step
            else:
                self.center += step
            self.center = int(np.clip(self.center, self.min_frame, self.max_frame))
            self._update_zoom()

    # ------------------------------------------------------------------
    def _update_zoom(self) -> None:
        half = max(self.window // 2, 1)
        start = max(self.center - half, self.min_frame)
        end = min(self.center + half, self.max_frame)

        for idx, (series, line) in enumerate(zip(self.series, self.zoom_lines)):
            mask = (series.frame_index >= start) & (series.frame_index <= end)
            x = series.frame_index[mask]
            y = series.values[mask]
            
            # zoom 범위 내에서 다시 정규화
            if y.size > 0:
                y_min = float(y.min())
                y_max = float(y.max())
                y_range = y_max - y_min
                if y_range > 0:
                    y_normalized = (y - y_min) / y_range
                else:
                    y_normalized = np.full_like(y, 0.5)
            else:
                y_normalized = np.array([])
            
            line.set_data(x, y_normalized)

        self.ax_zoom.set_xlim(start, end if end > start else start + 1)
        self.ax_zoom.set_ylim(0, 1)

        if self.meta is not None and self.meta_zoom_line is not None:
            mask = (self.meta.frame_index >= start) & (self.meta.frame_index <= end)
            x = self.meta.frame_index[mask]
            y = self.meta.values[mask]
            
            if y.size > 0:
                y_min = float(y.min())
                y_max = float(y.max())
                y_range = y_max - y_min
                if y_range > 0:
                    y_normalized = (y - y_min) / y_range
                else:
                    y_normalized = np.full_like(y, 0.5)
            else:
                y_normalized = np.array([])
            
            self.meta_zoom_line.set_data(x, y_normalized)

        legend_handles = list(self.zoom_lines)
        legend_labels = [series.label or f"series {idx + 1}" for idx, series in enumerate(self.series)]
        if self.meta_zoom_line is not None:
            legend_handles.append(self.meta_zoom_line)
            legend_labels.append('meta value')
        self.ax_zoom.legend(legend_handles, legend_labels, loc='upper right')

        self.full_cursor.set_xdata([self.center, self.center])
        self.zoom_cursor.set_xdata([self.center, self.center])

        status_parts = [f"frame: {self.center}"]
        timestamp = self._frame_to_timestamp(self.center)
        if timestamp is not None:
            status_parts.append(f"time: {timestamp:.1f} ms")
        for series in self.series:
            value = self._value_at(series, self.center)
            status_parts.append(f"{series.label}: {value:.4f}")
        if self.meta is not None:
            meta_value = self._meta_value_at(self.center)
            if meta_value is not None:
                status_parts.append(f"meta: {meta_value:.2f}")
        self.status_text.set_text(" | ".join(status_parts))
        self.fig.canvas.draw_idle()

        frame = self._load_frame(self.center)
        self.ax_video.clear()
        self.ax_video.imshow(frame)
        self.ax_video.axis("off")

    # ------------------------------------------------------------------
    def _frame_to_timestamp(self, frame: int) -> Optional[float]:
        timestamps = self.reference.timestamps
        if timestamps.size == 0:
            return None
        idx = np.searchsorted(self.reference.frame_index, frame)
        idx = int(np.clip(idx, 0, len(timestamps) - 1))
        return float(timestamps[idx])

    # ------------------------------------------------------------------
    def _value_at(self, series: PredictionSeries, frame: int) -> float:
        if series.frame_index.size == 0:
            return 0.0
        idx = np.searchsorted(series.frame_index, frame)
        idx = int(np.clip(idx, 0, len(series.frame_index) - 1))
        return float(series.values[idx]) if series.values.size else 0.0

    # ------------------------------------------------------------------
    def _meta_value_at(self, frame: int) -> Optional[float]:
        if self.meta is None or self.meta.frame_index.size == 0:
            return None
        idx = np.searchsorted(self.meta.frame_index, frame)
        idx = int(np.clip(idx, 0, len(self.meta.frame_index) - 1))
        return float(self.meta.values[idx])

    # ------------------------------------------------------------------
    def _trim_series(self, series: PredictionSeries) -> PredictionSeries:
        mask = series.frame_index >= self.skip_frames
        if not mask.any():
            raise ValueError(
                f"Series '{series.label}' does not contain frames beyond skip_frames={self.skip_frames}"
            )
        return PredictionSeries(
            label=series.label,
            frame_index=series.frame_index[mask],
            timestamps=series.timestamps[mask],
            values=series.values[mask],
        )

    # ------------------------------------------------------------------
    def _trim_meta(self, meta: Optional[MetaSeries]) -> Optional[MetaSeries]:
        if meta is None:
            return None
        mask = meta.frame_index >= self.skip_frames
        if not mask.any():
            return None
        return MetaSeries(
            frame_index=meta.frame_index[mask],
            timestamps=meta.timestamps[mask],
            values=meta.values[mask],
        )

    # ------------------------------------------------------------------
    def _generate_colors(self, count: int) -> List[tuple]:
        if count <= 0:
            return []
        
        # 가시성 좋은 구분되는 색상들
        base_colors = [
            '#1f77b4',  # blue
            '#ff7f0e',  # orange
            '#2ca02c',  # green
            '#d62728',  # red
            '#9467bd',  # purple
            '#8c564b',  # brown
            '#e377c2',  # pink
            '#7f7f7f',  # gray
            '#bcbd22',  # yellow-green
            '#17becf',  # cyan
            '#aec7e8',  # light blue
            '#ffbb78',  # light orange
            '#98df8a',  # light green
            '#ff9896',  # light red
            '#c5b0d5',  # light purple
        ]
        
        # 필요한 만큼 반복하여 사용
        colors = []
        for i in range(count):
            hex_color = base_colors[i % len(base_colors)]
            # hex를 RGB tuple로 변환
            rgb = tuple(int(hex_color[j:j+2], 16) / 255.0 for j in (1, 3, 5))
            colors.append(rgb)
        
        return colors

    # ------------------------------------------------------------------
    @staticmethod
    def _compute_bounds(arrays: Sequence[np.ndarray], symmetric: bool) -> tuple[float, float]:
        values = [np.asarray(a, dtype=float).ravel() for a in arrays if np.size(a)]
        if not values:
            return (-0.1, 0.1) if symmetric else (-0.1, 0.1)
        stacked = np.concatenate([v[~np.isnan(v)] for v in values if v.size])
        if stacked.size == 0:
            return (-0.1, 0.1) if symmetric else (-0.1, 0.1)
        v_min = float(stacked.min())
        v_max = float(stacked.max())
        if symmetric:
            max_abs = max(abs(v_min), abs(v_max))
            if np.isclose(max_abs, 0.0):
                bound = 0.1
            else:
                bound = max_abs * 1.05
            return (-bound, bound)
        span = v_max - v_min
        if np.isclose(span, 0.0):
            pad = max(abs(v_min) * 0.1, 0.1)
            return (v_min - pad, v_max + pad)
        margin = max(span * 0.05, 0.05 * max(abs(v_min), abs(v_max), 0.1))
        return (v_min - margin, v_max + margin)


# ----------------------------------------------------------------------

def load_prediction_series(
    folder: Path,
    include: Optional[Sequence[str]] = None,
    metric: str = "predicted_change",
    skip_frames: int = 32,
) -> List[PredictionSeries]:
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Video folder not found: {folder}")

    include_tokens = _normalize_include_tokens(include)
    skip_frames = max(int(skip_frames), 0)

    series: List[PredictionSeries] = []
    for path in sorted(folder.glob("predictions-*.csv")):
        label = path.stem[len("predictions-"):]
        if include_tokens and not _matches_include(label, include_tokens):
            continue
        df = pd.read_csv(path)
        required = {"frame_index", "timestamp_ms", metric}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"Missing columns {missing} in {path}")
        frame_index = df["frame_index"].to_numpy(dtype=np.int32)
        timestamps = df["timestamp_ms"].to_numpy(dtype=float)
        values = df[metric].to_numpy(dtype=float)
        mask = frame_index >= skip_frames
        if not mask.any():
            raise ValueError(
                f"File {path.name} has no frames beyond skip_frames={skip_frames}."
            )
        series.append(
            PredictionSeries(
                label=label,
                frame_index=frame_index[mask],
                timestamps=timestamps[mask],
                values=values[mask],
            )
        )

    if not series:
        raise RuntimeError("No prediction CSV files matched the criteria")
    return series


def load_meta_series(folder: Path, skip_frames: int = 32) -> Optional[MetaSeries]:
    folder = Path(folder)
    path = folder / "meta.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "interference_value" not in df.columns:
        return None
    frame_index = (
        df["frame_index"].to_numpy(dtype=np.int32)
        if "frame_index" in df.columns
        else np.arange(len(df), dtype=np.int32)
    )
    timestamps = (
        df["timestamp_ms"].to_numpy(dtype=float)
        if "timestamp_ms" in df.columns
        else frame_index.astype(float)
    )
    values = df["interference_value"].to_numpy(dtype=float)
    mask = frame_index >= max(int(skip_frames), 0)
    if not mask.any():
        return None
    return MetaSeries(
        frame_index=frame_index[mask],
        timestamps=timestamps[mask],
        values=values[mask],
    )


def _normalize_include_tokens(include: Optional[Sequence[str]]) -> List[str]:
    if not include:
        return []
    tokens: List[str] = []
    for token in include:
        token = token.strip()
        if not token:
            continue
        cleaned = token
        if cleaned.startswith("predictions-"):
            cleaned = cleaned[len("predictions-"):]
        if cleaned.endswith(".csv"):
            cleaned = cleaned[:-4]
        tokens.append(cleaned.lower())
    return tokens


def _matches_include(label: str, tokens: Sequence[str]) -> bool:
    lower = label.lower()
    return any(token in lower for token in tokens)


__all__ = [
    "PredictionSeries",
    "MetaSeries",
    "PredictionComparisonViewer",
    "load_prediction_series",
    "load_meta_series",
]