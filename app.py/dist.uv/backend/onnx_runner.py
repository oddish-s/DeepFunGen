"""ONNX runtime wrapper mirroring the WinForms behaviour."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import numpy as np

try:  # Prefer DirectML build on Windows when available
    import onnxruntime_directml as ort  # type: ignore
except ImportError:  # pragma: no cover - fallback for other platforms
    import onnxruntime as ort  # type: ignore


logger = logging.getLogger("deepfungen.onnx")


class OnnxSequenceModel:
    """Loads an ONNX sequence regression network and runs inference."""

    SEQUENCE_LENGTH: int = 10
    CHANNELS: int = 3
    HEIGHT: int = 224
    WIDTH: int = 224
    NORMALIZATION_MEAN: float = 0.0
    NORMALIZATION_STD: float = 10.0

    def __init__(self, model_path: str | Path, prefer_gpu: bool = True) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.model_path}")
        self._session, self.execution_provider = self._create_session(prefer_gpu)
        self.sequence_length = self.SEQUENCE_LENGTH
        self.channels = self.CHANNELS
        self.height = self.HEIGHT
        self.width = self.WIDTH
        inputs = self._session.get_inputs()
        outputs = self._session.get_outputs()
        if not inputs or not outputs:
            raise RuntimeError("ONNX model must have at least one input and one output")
        self._input_name = inputs[0].name
        self._output_name = outputs[0].name
        self._apply_input_shape(inputs[0].shape)

    # ------------------------------------------------------------------
    def _create_session(self, prefer_gpu: bool) -> tuple[ort.InferenceSession, str]:
        providers: List[str] = []
        available = set(ort.get_available_providers())
        if prefer_gpu:
            if "DmlExecutionProvider" in available:
                providers.append("DmlExecutionProvider")
            if "CUDAExecutionProvider" in available:
                providers.append("CUDAExecutionProvider")
            if not providers:
                logger.info("GPU inference requested but only %s providers available", ", ".join(sorted(available)))
        providers.append("CPUExecutionProvider")
        session = ort.InferenceSession(str(self.model_path), providers=providers)
        active_provider = session.get_providers()[0]
        return session, active_provider

    # ------------------------------------------------------------------
    def _apply_input_shape(self, shape: List[int | str | None]) -> None:
        if not shape:
            return
        if len(shape) == 5:
            _, seq_dim, ch_dim, height_dim, width_dim = shape
        elif len(shape) == 4:
            seq_dim, height_dim, width_dim, ch_dim = shape
        else:
            return
        if isinstance(seq_dim, int) and seq_dim > 0:
            self.sequence_length = seq_dim
        if isinstance(ch_dim, int) and ch_dim > 0:
            self.channels = ch_dim
        if isinstance(height_dim, int) and height_dim > 0:
            self.height = height_dim
        if isinstance(width_dim, int) and width_dim > 0:
            self.width = width_dim

    # ------------------------------------------------------------------
    def infer(self, sequence: np.ndarray) -> np.ndarray:
        """Run inference over a single sequence of frames.

        Args:
            sequence: Array with shape (SEQ, H, W, C) or (SEQ, C, H, W)
        """
        if sequence.ndim != 4:
            raise ValueError("sequence must be 4D")
        seq = np.asarray(sequence, dtype=np.float32)
        if seq.shape[0] != self.sequence_length:
            raise ValueError(f"Expected sequence length {self.sequence_length}, got {seq.shape[0]}")
        if seq.shape[-1] == self.channels:
            # convert to (SEQ, C, H, W)
            seq = np.transpose(seq, (0, 3, 1, 2))
        if seq.shape[1:] != (self.channels, self.height, self.width):
            raise ValueError(
                "Unexpected frame shape; expected (SEQ, C, H, W) with "
                f"({self.channels}, {self.height}, {self.width}), got {seq.shape}"
            )
        input_tensor = seq.reshape(1, self.sequence_length, self.channels, self.height, self.width)
        outputs = self._session.run([self._output_name], {self._input_name: input_tensor})[0]
        flat = np.asarray(outputs, dtype=np.float32).reshape(-1)
        # Denormalise to match training target scale
        return flat * self.NORMALIZATION_STD + self.NORMALIZATION_MEAN


__all__ = ["OnnxSequenceModel"]
