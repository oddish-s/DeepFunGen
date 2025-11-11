"""Prediction post-processing helpers for interference phase extraction."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


@dataclass
class PostProcessConfig:
    """Configuration for prominence and slope-aware extrema processing."""

    frame_rate: float = 30.0
    smooth_window_frames: int = 3
    prominence_ratio: float = 0.1
    min_prominence: float = 0.0
    max_slope: float = 10.0
    boost_slope: float = 7.0
    min_slope: float = 0.0
    merge_threshold_ms: float = 120.0
    fft_denoise: bool = False
    fft_frames_per_component: int = 40
    fft_window_frames: int | None = None

    @classmethod
    def from_mapping(cls, cfg: Mapping[str, object] | None) -> "PostProcessConfig":
        if not cfg:
            return cls()
        kwargs: Dict[str, object] = {}
        aliases = {
            "frame_rate": "frame_rate",
            "smooth_window_frames": "smooth_window_frames",
            "prominence_ratio": "prominence_ratio",
            "min_prominence": "min_prominence",
            "max_slope": "max_slope",
            "min_slope": "min_slope",
            "merge_threshold_ms": "merge_threshold_ms",
        }
        for key, field in aliases.items():
            if key in cfg:
                kwargs[field] = cfg[key]
        if 'boost_slope' in cfg:
            kwargs['boost_slope'] = cfg['boost_slope']
        if "merge_threshold_seconds" in cfg and "merge_threshold_ms" not in kwargs:
            kwargs["merge_threshold_ms"] = float(cfg["merge_threshold_seconds"]) * 1000.0
        if "merge_threshold" in cfg and "merge_threshold_ms" not in kwargs:
            kwargs["merge_threshold_ms"] = cfg["merge_threshold"]
        if "min_gradient" in cfg and "min_slope" not in kwargs:
            kwargs["min_slope"] = cfg["min_gradient"]
        if "max_gradient" in cfg and "max_slope" not in kwargs:
            kwargs["max_slope"] = cfg["max_gradient"]
        if "prominence" in cfg and "min_prominence" not in kwargs:
            kwargs["min_prominence"] = cfg["prominence"]
        if "fft_denoise" in cfg:
            kwargs["fft_denoise"] = bool(cfg["fft_denoise"])
        if "fft_frames_per_component" in cfg:
            kwargs["fft_frames_per_component"] = int(cfg["fft_frames_per_component"])
        if "fft_window_frames" in cfg:
            value = cfg["fft_window_frames"]
            kwargs["fft_window_frames"] = int(value) if value is not None else None
        return cls(**kwargs)  # type: ignore[arg-type]


@dataclass
class _Extremum:
    frame: float
    kind: str  # "peak" or "trough"
    raw_change: float
    processed_change: float | None = None
    origin: str = "original"
    prominence: float = 0.0
    window_y_range: Tuple[float, float] = (0.0, 0.0)
    width: float = 0.0

@dataclass
class _GraphPoint:
    position: float
    value: float
    label: str
    origin: str
    direction: str | None = None


_PEAK_VALUE = 100.0
_TROUGH_VALUE = 0.0


def apply_postprocessing(predictions: pd.DataFrame, config: PostProcessConfig) -> pd.DataFrame:
    """Transform raw change predictions into a bounded interference curve."""

    if "predicted_change" not in predictions.columns:
        raise KeyError("predictions must include a 'predicted_change' column")

    change = predictions["predicted_change"].astype(float).reset_index(drop=True)
    n_frames = len(change)
    if n_frames < 3:
        return _fallback_postprocess(predictions)

    indices = change.index
    raw_array = change.to_numpy(dtype=float)
    processing_values = raw_array.copy()
    fft_denoised: np.ndarray | None = None
    if config.fft_denoise and n_frames > 2:
        fft_denoised = _fft_denoise(
            processing_values,
            frames_per_component=int(max(1, config.fft_frames_per_component)),
            window_frames=config.fft_window_frames,
        )
        processing_values = fft_denoised

    delay_profile = np.array([], dtype=float)
    delay_samples_pos = np.array([], dtype=float)
    delay_samples_val = np.array([], dtype=float)
    average_delay_frames = 0.0

    change = pd.Series(raw_array, index=indices, dtype=float)
    processing_series = pd.Series(processing_values, index=change.index, dtype=float)
    smooth_window = max(1, int(config.smooth_window_frames))
    smoothed = (
        processing_series.rolling(smooth_window, center=True, min_periods=1).mean()
        if smooth_window > 1
        else processing_series
    )

    raw_extrema = _detect_extrema(smoothed, processing_series, change, config)
    if not raw_extrema:
        return _fallback_postprocess(predictions)

    stage_two = _insert_missing_extrema(raw_extrema, n_frames, change, processing_series)
    merge_frames = _ms_to_frames(config.merge_threshold_ms, config.frame_rate)
    if merge_frames > 0:
        stage_two = _merge_triplets(stage_two, merge_frames, change)
    stage_two = _sort_extrema(stage_two)
    if len(stage_two) < 2:
        return _fallback_postprocess(predictions)

    graph_points = _apply_slope_constraints(stage_two, config.min_slope, config.max_slope, config.boost_slope)
    if len(graph_points) < 2:
        return _fallback_postprocess(predictions)

    processed_value, phase_marker, phase_source = _build_processed_arrays(graph_points, n_frames)
    raw_marker = _build_raw_markers(raw_extrema, n_frames)

    processed_change = np.zeros_like(processed_value)
    processed_change[1:] = processed_value[1:] - processed_value[:-1]

    result = predictions.copy()
    result["processed_value"] = processed_value
    result["processed_change"] = processed_change
    result["phase_marker"] = phase_marker
    result["phase_source"] = phase_source
    result["raw_extrema_marker"] = raw_marker

    result.attrs["postprocess"] = {
        "raw_signal": change.to_numpy(copy=True),
        "smoothed_signal": smoothed.to_numpy(copy=True),
        "fft_denoised_signal": fft_denoised.copy() if fft_denoised is not None else np.array([], dtype=float),
        "raw_extrema": [_extremum_to_dict(ext) for ext in _sort_extrema(raw_extrema)],
        "stage_two_extrema": [_extremum_to_dict(ext) for ext in stage_two],
        "graph_points": [_graph_point_to_dict(pt) for pt in graph_points],
        "temporal_shift_frames": int(round(average_delay_frames)),
        "temporal_delay_profile": delay_profile.copy(),
        "temporal_delay_markers": [
            {"position": float(pos), "delay": float(val)}
            for pos, val in zip(delay_samples_pos, delay_samples_val)
        ],
        "config": config,
    }
    return result


def _fallback_postprocess(predictions: pd.DataFrame) -> pd.DataFrame:
    result = predictions.copy()
    n = len(result)
    result["processed_value"] = np.full(n, 50.0, dtype=np.float64)
    result["processed_change"] = np.zeros(n, dtype=np.float64)
    result["phase_marker"] = np.full(n, np.nan, dtype=np.float64)
    result["phase_source"] = np.full(n, "", dtype=object)
    result["raw_extrema_marker"] = np.full(n, np.nan, dtype=np.float64)
    result.attrs["postprocess"] = {
        "raw_signal": result["predicted_change"].to_numpy(copy=True) if n else np.array([], dtype=float),
        "smoothed_signal": result["predicted_change"].to_numpy(copy=True) if n else np.array([], dtype=float),
        "fft_denoised_signal": np.array([], dtype=float),
        "raw_extrema": [],
        "stage_two_extrema": [],
        "graph_points": [],
        "temporal_shift_frames": 0,
        "temporal_delay_profile": np.array([], dtype=float),
        "temporal_delay_markers": [],
        "config": None,
    }
    return result


def _fft_denoise(
    signal: np.ndarray,
    *,
    frames_per_component: int,
    window_frames: int | None = None,
) -> np.ndarray:
    n = signal.size
    if n == 0:
        return signal.copy()

    def _apply_filter(chunk: np.ndarray) -> np.ndarray:
        chunk_len = chunk.size
        if chunk_len == 0:
            return chunk
        keep = max(1, chunk_len // frames_per_component)
        if keep * 2 >= chunk_len:
            return chunk
        spectrum = np.fft.fft(chunk)
        cutoff = min(keep, chunk_len // 2)
        spectrum[cutoff : chunk_len - cutoff] = 0
        reconstructed = np.fft.ifft(spectrum)
        return reconstructed.real.astype(float, copy=False)

    window = window_frames if window_frames and window_frames > 0 else None
    if window is None or window >= n:
        return _apply_filter(signal.copy())

    output = np.empty_like(signal, dtype=float)
    start = 0
    while start < n:
        end = min(start + window, n)
        output[start:end] = _apply_filter(signal[start:end])
        start = end
    return output


def _detect_extrema(
    smoothed: pd.Series,
    processed: pd.Series,
    raw: pd.Series,
    config: PostProcessConfig,
) -> list[_Extremum]:
    values = smoothed.to_numpy()
    if values.size == 0:
        return []

    window_size = 200
    candidates: list[_Extremum] = []
    
    # Traverse the signal in overlapping windows to evaluate local extrema.
    for start_idx in range(0, len(values), window_size // 2):
        end_idx = min(start_idx + window_size, len(values))
        window_values = values[start_idx:end_idx]
        
        if window_values.size == 0:
            continue

        window_y_min: float = float(np.min(window_values))
        window_y_max: float = float(np.max(window_values))

        data_range_window = float(np.ptp(window_values))
        threshold = max(data_range_window * float(config.prominence_ratio), float(config.min_prominence))
        
        # height = (window_y_max - window_y_min) / 4.0
        height_min = (window_y_max - window_y_min) / 10.0 

        peak_kwargs = {"width": 1}
        if threshold > 0.0:
            peak_kwargs["prominence"] = threshold
        peaks, peak_props = find_peaks(window_values, **peak_kwargs)
        troughs, trough_props = find_peaks(-window_values, **peak_kwargs)

        peak_prom = np.asarray(peak_props.get("prominences", np.zeros(peaks.size, dtype=float)), dtype=float)
        peak_widths = np.asarray(peak_props.get("widths", np.zeros(peaks.size, dtype=float)), dtype=float)
        for local_idx, peak_idx in enumerate(peaks):
            global_idx = int(start_idx + peak_idx)
            if 0 <= global_idx < len(raw):
                prominence = float(peak_prom[local_idx]) if local_idx < peak_prom.size else 0.0
                width = float(peak_widths[local_idx]) if local_idx < peak_widths.size else 0.0
                candidate = _Extremum(
                    frame=float(global_idx),
                    kind="peak",
                    raw_change=float(raw.iloc[global_idx]),
                    processed_change=float(processed.iloc[global_idx]),
                    origin="original",
                    prominence=prominence,
                    window_y_range=(window_y_min, window_y_max),
                    width=width,
                )
                candidates.append(candidate)

        trough_prom = np.asarray(trough_props.get("prominences", np.zeros(troughs.size, dtype=float)), dtype=float)
        trough_widths = np.asarray(trough_props.get("widths", np.zeros(troughs.size, dtype=float)), dtype=float)
        for local_idx, trough_idx in enumerate(troughs):
            global_idx = int(start_idx + trough_idx)
            if 0 <= global_idx < len(raw):
                prominence = float(trough_prom[local_idx]) if local_idx < trough_prom.size else 0.0
                width = float(trough_widths[local_idx]) if local_idx < trough_widths.size else 0.0
                candidate = _Extremum(
                    frame=float(global_idx),
                    kind="trough",
                    raw_change=float(raw.iloc[global_idx]),
                    processed_change=float(processed.iloc[global_idx]),
                    origin="original",
                    prominence=prominence,
                    window_y_range=(window_y_min, window_y_max),
                    width=width,
                )
                candidates.append(candidate)
    
    # Deduplicate frames by keeping the most prominent candidate per position.
    frame_to_extremum = {}
    for candidate in candidates:
        frame = candidate.frame
        if frame not in frame_to_extremum or candidate.prominence > frame_to_extremum[frame].prominence:
            frame_to_extremum[frame] = candidate
    
    extrema = list(frame_to_extremum.values())
    
    # Optional global filtering hooks retained for future tuning.
    # data_range = float(np.ptp(values)) if values.size else 0.0
    # threshold = max(data_range * float(config.prominence_ratio), float(config.min_prominence))
    
    #filtered_extrema = [e for e in extrema if e.prominence >= threshold]
    
    #if not filtered_extrema and extrema:
    #    extrema_sorted = sorted(extrema, key=lambda e: e.prominence, reverse=True)
    #    filtered_extrema = extrema_sorted[:2]
    
    return _sort_extrema(extrema)

def _insert_missing_extrema(
    extrema: Sequence[_Extremum],
    total_frames: int,
    raw_change: pd.Series,
    processed_change: pd.Series,
) -> list[_Extremum]:
    if not extrema:
        return []
    result: list[_Extremum] = []
    for i, current in enumerate(extrema):
        result.append(current)
        if i == len(extrema) - 1:
            continue
        nxt = extrema[i + 1]
        if current.kind == nxt.kind:
            current_frame = int(round(current.frame))
            next_frame = int(round(nxt.frame))
            mid = current_frame if current_frame == next_frame else (current_frame + next_frame) // 2
            mid = int(np.clip(mid, 0, total_frames - 1))
            opposite = "peak" if current.kind == "trough" else "trough"
            processed_val = float(processed_change.iloc[mid]) if 0 <= mid < len(processed_change) else 0.0
            raw_val = float(raw_change.iloc[mid]) if 0 <= mid < len(raw_change) else 0.0
            if current.width > 0.0 and nxt.width > 0.0:
                estimated_width = 0.5 * (current.width + nxt.width)
            else:
                estimated_width = max(current.width, nxt.width, 1.0)
            result.append(
                _Extremum(
                    frame=float(mid),
                    kind=opposite,
                    raw_change=raw_val,
                    processed_change=processed_val,
                    origin="inserted",
                    prominence=0.0,
                    width=float(estimated_width),
                )
            )
    return _sort_extrema(result)


def _merge_triplets(extrema: Sequence[_Extremum], threshold_frames: int, change: pd.Series) -> list[_Extremum]:
    if len(extrema) < 3:
        return list(extrema)
    merged: list[_Extremum] = []
    i = 0
    total_frames = len(change)
    while i < len(extrema):
        if i + 2 < len(extrema):
            a, b, c = extrema[i], extrema[i + 1], extrema[i + 2]
            if (
                a.kind == c.kind
                and a.kind != b.kind
                and (c.frame - a.frame) <= float(threshold_frames)
            ):
                # Place the merged extremum midway between the outer extrema.
                mid_frame = int(np.clip(round((a.frame + c.frame) / 2), 0, total_frames - 1))
                merged.append(
                    _Extremum(
                        frame=float(mid_frame),
                        kind=a.kind,  # Preserve the shared peak/trough label.
                        raw_change=float(change.iloc[mid_frame]),
                        origin="merged",
                        prominence=max(a.prominence, c.prominence),  # Carry over the stronger prominence.
                        window_y_range=(min(a.window_y_range[0], c.window_y_range[0]), max(a.window_y_range[1], c.window_y_range[1])),
                        width=float(max(a.width, c.width)),
                    )
                )
                i += 3
                continue
        merged.append(extrema[i])
        i += 1
    return merged


def _apply_slope_constraints(
    extrema: Sequence[_Extremum],
    min_slope: float,
    max_slope: float,
    boost_slope: float,
) -> list[_GraphPoint]:
    if not extrema:
        return []
    min_slope = max(0.0, float(min_slope))
    max_slope = float(max_slope)
    if max_slope <= 0.0:
        max_slope = float("inf")
    if max_slope < min_slope:
        max_slope = min_slope

    graph: list[_GraphPoint] = []
    current = extrema[0]
    current_x = float(current.frame)
    current_y = _make_target_y(current)
    graph.append(_GraphPoint(current_x, current_y, current.kind, current.origin))

    offset_decay = 0.82
    offset_clamp = 15.0
    current_offset = 0.0

    i = 1
    while i < len(extrema):
        target = extrema[i]
        target_y = _make_target_y(target)

        base_dx = float(target.frame) - current_x
        base_dy = target_y - current_y
        base_slope = abs(base_dy) / base_dx if base_dx > 1e-9 else float("inf")
        desired_offset = _compute_slope_offset(base_slope)

        current_offset = current_offset * offset_decay + desired_offset * (1.0 - offset_decay)
        current_offset = float(np.clip(current_offset, 0.0, offset_clamp))

        candidate_x = float(target.frame) + current_offset
        if i + 1 < len(extrema):
            next_frame = float(extrema[i + 1].frame)
            candidate_x = min(candidate_x, next_frame - 0.5)
        candidate_x = max(candidate_x, float(target.frame))
        candidate_x = max(candidate_x, current_x + 1e-4)

        target_x = candidate_x

        dx = target_x - current_x
        if dx <= 0:
            i += 1
            continue
        dy = target_y - current_y
        actual_slope = abs(dy) / dx if dx else float("inf")
        direction = "to_peak" if dy > 0 else "to_trough"

        if actual_slope > max_slope:
            max_delta = max_slope * dx
            reachable = current_y + np.sign(dy) * max_delta
            reachable = float(np.clip(reachable, _TROUGH_VALUE, _PEAK_VALUE))
            graph.append(_GraphPoint(target_x, reachable, "intermediate", "max_slope", direction))
            current_x = target_x
            current_y = reachable
            i += 1
            continue

        if actual_slope < min_slope and abs(dy) > 1e-9:
            injected, current_x, current_y = _handle_min_slope_segment(
                current_x,
                current_y,
                target_x,
                target_y,
                direction,
                target.kind,
                target.origin,
                min_slope,
                max_slope,
            )
            graph.extend(injected)
            i += 1
            continue

        graph.append(_GraphPoint(target_x, target_y, target.kind, target.origin))
        current_x = target_x
        current_y = target_y
        i += 1
    boosted_graph = _boost_flat_peaks(graph, boost_slope)
    return _dedupe_graph_points(boosted_graph)


def _compute_slope_offset(slope: float) -> float:
    if not np.isfinite(slope):
        slope = 100.0

    min_offset = 1.0
    max_offset = 15.0
    gentle_slope = 2.0
    steep_slope = 12.0

    if slope <= gentle_slope:
        return max_offset
    if slope >= steep_slope:
        return min_offset

    normalized = (slope - gentle_slope) / (steep_slope - gentle_slope)
    offset = max_offset - normalized * (max_offset - min_offset)
    return float(np.clip(offset, min_offset, max_offset))

def _make_target_y(extremum: _Extremum) -> float:
    ## Clamp the extremum amplitude using its local window range.
    #window_y_min, window_y_max = extremum.window_y_range
    #
    ## Normalize the raw change according to that window span.
    #if window_y_max > window_y_min:
    #    normalized_value = (extremum.raw_change - window_y_min) / (window_y_max - window_y_min)
    #else:
    #    normalized_value = 0.5  # Fallback to the midpoint when the span collapses.

    normalized_value = abs(extremum.raw_change) * 5.0
    if normalized_value > 1.0:
        normalized_value = 1.0
    elif normalized_value < 0.1:
        normalized_value = 0.1

    middle = (_PEAK_VALUE - _TROUGH_VALUE) / 2.0

    if extremum.kind == "peak":
        return middle + ( _PEAK_VALUE - middle) * normalized_value
    elif extremum.kind == "trough":
        return middle - (middle - _TROUGH_VALUE) * normalized_value

    return 0.0

def _handle_min_slope_segment(
    current_x: float,
    current_y: float,
    target_x: float,
    target_y: float,
    direction: str,
    target_kind: str,
    target_origin: str,
    min_slope: float,
    max_slope: float,
) -> Tuple[list[_GraphPoint], float, float]:
    dx = target_x - current_x
    dy = target_y - current_y
    if dx <= 0:
        return [], current_x, current_y

    min_direct_dx = abs(dy) / min_slope if min_slope > 0 else dx
    extra_dx = dx - min_direct_dx
    if extra_dx <= 0:
        return [
            _GraphPoint(target_x, target_y, target_kind, target_origin)
        ], target_x, target_y

    full_amplitude = _PEAK_VALUE - _TROUGH_VALUE
    half_dx_min = full_amplitude / min_slope if min_slope > 0 else 0.0
    half_dx_max = (
        full_amplitude / max_slope
        if max_slope > 0 and not np.isinf(max_slope)
        else 0.0
    )
    half_dx = max(half_dx_min, half_dx_max, 1e-6)

    max_transitions = int(extra_dx / half_dx)
    is_at_extreme = abs(current_y - _PEAK_VALUE) < 1e-6 or abs(current_y - _TROUGH_VALUE) < 1e-6
    if is_at_extreme:
        num_transitions = (max_transitions // 2) * 2
    else:
        num_transitions = max_transitions if max_transitions % 2 == 1 else max_transitions - 1
        if num_transitions <= 0:
            num_transitions = 1

    points: list[_GraphPoint] = []
    if num_transitions <= 0:
        points.append(_GraphPoint(target_x, target_y, target_kind, target_origin))
        return points, target_x, target_y

    total_segments = num_transitions + 1
    segment_dx = dx / total_segments
    current_pos = current_x
    previous_value = current_y

    sequence: list[float] = []
    if is_at_extreme:
        sequence.append(target_y)
        current_value = target_y
        for _ in range(num_transitions - 1):
            current_value = _TROUGH_VALUE if current_value == _PEAK_VALUE else _PEAK_VALUE
            sequence.append(current_value)
    else:
        first_transition = _TROUGH_VALUE if direction == "to_peak" else _PEAK_VALUE
        sequence.append(first_transition)
        current_value = first_transition
        for _ in range(num_transitions - 1):
            current_value = _TROUGH_VALUE if current_value == _PEAK_VALUE else _PEAK_VALUE
            sequence.append(current_value)

    for value in sequence:
        current_pos += segment_dx
        dir_flag = "to_peak" if value > previous_value else "to_trough"
        points.append(_GraphPoint(current_pos, value, "intermediate", "min_slope", dir_flag))
        previous_value = value

    points.append(_GraphPoint(target_x, target_y, target_kind, target_origin))
    return points, target_x, target_y



def _boost_flat_peaks(
    points: Sequence[_GraphPoint],
    boost_slope: float,
) -> list[_GraphPoint]:
    if len(points) < 3:
        return list(points)
        
    peak_threshold = _PEAK_VALUE / 4.0 * 3.0
    boosted: list[_GraphPoint] = []
    total = len(points)
    for idx, point in enumerate(points):
        prev_point = points[idx - 1] if idx > 0 else None
        next_point = points[idx + 1] if idx + 1 < total else None
        if not prev_point or not next_point:
            boosted.append(point)
            continue
        if (
            prev_point.label != "trough"
            or next_point.label != "trough"
            or point.label != "peak"
            or point.value < peak_threshold
        ):
            boosted.append(point)
            continue

        left_dx = float(point.position) - float(prev_point.position)
        right_dx = float(next_point.position) - float(point.position)
        if left_dx <= 0.0 or right_dx <= 0.0:
            boosted.append(point)
            continue

        left_slope = abs(point.value - prev_point.value) / left_dx
        right_slope = abs(next_point.value - point.value) / right_dx
        if left_slope > boost_slope or right_slope > boost_slope:
            boosted.append(point)
            continue

        left_x = float(point.position) - left_dx * 0.4
        right_x = float(point.position) + right_dx * 0.4
        if left_x <= float(prev_point.position) or right_x >= float(next_point.position):
            boosted.append(point)
            continue

        neighbor_avg = (prev_point.value + next_point.value) / 2.0
        peak_to_neighbor = point.value - neighbor_avg
        if peak_to_neighbor <= 0.0:
            boosted.append(point)
            continue

        shoulder_y = point.value - peak_to_neighbor * 0.2
        if shoulder_y >= point.value or shoulder_y <= neighbor_avg:
            boosted.append(point)
            continue

        boosted.append(_GraphPoint(left_x, shoulder_y, "boosted", "boosted"))
        boosted.append(_GraphPoint(float(point.position), point.value, point.label, point.origin))
        boosted.append(_GraphPoint(right_x, shoulder_y, "boosted", "boosted"))
    return boosted

def _dedupe_graph_points(points: Iterable[_GraphPoint]) -> list[_GraphPoint]:
    cleaned: list[_GraphPoint] = []
    last_pos: float | None = None
    for pt in points:
        pos = float(pt.position)
        if last_pos is not None and pos <= last_pos:
            pos = last_pos + 1e-6
        cleaned.append(
            _GraphPoint(
                pos,
                float(np.clip(pt.value, _TROUGH_VALUE, _PEAK_VALUE)),
                pt.label,
                pt.origin,
                pt.direction,
            )
        )
        last_pos = pos
    return cleaned


def _build_processed_arrays(points: Sequence[_GraphPoint], n_frames: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    positions = np.array([pt.position for pt in points], dtype=np.float64)
    values = np.array([pt.value for pt in points], dtype=np.float64)
    if positions.size < 2:
        processed = np.full(n_frames, values[0] if values.size else 50.0, dtype=np.float64)
    else:
        processed = np.interp(
            np.arange(n_frames, dtype=np.float64),
            positions,
            values,
            left=values[0],
            right=values[-1],
        )
    phase_marker = np.full(n_frames, np.nan, dtype=np.float64)
    phase_source = np.full(n_frames, "", dtype=object)
    for pt in points:
        if pt.label in {"peak", "trough"}:
            idx = int(round(pt.position))
            if 0 <= idx < n_frames:
                phase_marker[idx] = _PEAK_VALUE if pt.label == "peak" else _TROUGH_VALUE
                phase_source[idx] = pt.origin
    return processed, phase_marker, phase_source


def _build_raw_markers(extrema: Sequence[_Extremum], n_frames: int) -> np.ndarray:
    marker = np.full(n_frames, np.nan, dtype=np.float64)
    for ext in extrema:
        idx = int(round(ext.frame))
        if 0 <= idx < n_frames:
            marker[idx] = _PEAK_VALUE if ext.kind == "peak" else _TROUGH_VALUE
    return marker


def _ms_to_frames(duration_ms: float, frame_rate: float) -> int:
    if duration_ms <= 0.0 or frame_rate <= 0.0:
        return 0
    return max(1, int(round(duration_ms / 1000.0 * frame_rate)))


def _sort_extrema(extrema: Sequence[_Extremum]) -> list[_Extremum]:
    return sorted(extrema, key=lambda e: (e.frame, 0 if e.kind == "peak" else 1))

def _extremum_to_dict(ext: _Extremum) -> Dict[str, float | str]:
    return {
        "position": float(ext.frame),
        "kind": ext.kind,
        "origin": ext.origin,
        "raw_change": float(ext.raw_change),
        "processed_change": float(ext.processed_change) if ext.processed_change is not None else np.nan,
        "prominence": float(ext.prominence),
    }


def _graph_point_to_dict(pt: _GraphPoint) -> Dict[str, float | str | None]:
    return {
        "position": float(pt.position),
        "value": float(pt.value),
        "label": pt.label,
        "origin": pt.origin,
        "direction": pt.direction,
    }


__all__ = ["PostProcessConfig", "apply_postprocessing"]





