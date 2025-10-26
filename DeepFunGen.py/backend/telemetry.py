"""System telemetry helpers for status bar metrics."""
from __future__ import annotations

from typing import Optional

try:
    import psutil
except ImportError:  # pragma: no cover - optional dependency
    psutil = None

try:
    import pynvml
except ImportError:  # pragma: no cover - optional dependency
    pynvml = None

from .models import SystemStatusModel


class TelemetryCollector:
    """Collects lightweight CPU/GPU utilisation figures when possible."""

    def __init__(self) -> None:
        self._nvml_initialised = False
        if pynvml is not None:
            try:
                pynvml.nvmlInit()
                self._nvml_initialised = True
            except Exception as error:  # pragma: no cover - optional dependency
                print(error)
                self._nvml_initialised = False

    def snapshot(self, *, queue_counts: dict, execution_provider: Optional[str]) -> SystemStatusModel:
        cpu_usage = self._cpu_usage()
        gpu_usage = self._gpu_usage()
        return SystemStatusModel(
            gpu_usage=gpu_usage,
            cpu_usage=cpu_usage,
            queue_total=queue_counts.get("total", 0),
            queue_pending=queue_counts.get("pending", 0),
            queue_processing=queue_counts.get("processing", 0),
            queue_completed=queue_counts.get("completed", 0),
            execution_provider=execution_provider,
        )

    def _cpu_usage(self) -> float:
        if psutil is None:
            return 0.0
        try:
            return float(psutil.cpu_percent(interval=None))
        except Exception:
            return 0.0

    def _gpu_usage(self) -> Optional[float]:
        if not self._nvml_initialised or pynvml is None:
            return None
        try:
            device_count = pynvml.nvmlDeviceGetCount()
            if device_count == 0:
                return None
            usages = []
            for idx in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                usages.append(float(util.gpu))
            if not usages:
                return None
            return sum(usages) / len(usages)
        except Exception:
            return None

    def shutdown(self) -> None:
        if self._nvml_initialised and pynvml is not None:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
            self._nvml_initialised = False


__all__ = ["TelemetryCollector"]
