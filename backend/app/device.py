"""Device reporting for the health endpoint.

VieNeu-TTS runs on a torch-free ONNX CPU runtime, so the backend is always CPU.
``VOICE_AI_DEVICE`` is honoured for compatibility; ``cpu`` is the only value it
accepts.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class DeviceInfo:
    device: str
    backend: str
    description: str

    @property
    def is_gpu(self) -> bool:
        return self.device != "cpu"


def resolve_device() -> DeviceInfo:
    """VieNeu is CPU-only (ONNX Runtime); report the CPU device."""
    forced = os.environ.get("VOICE_AI_DEVICE", "").strip().lower()
    if forced and forced != "cpu":
        raise RuntimeError(
            f"VOICE_AI_DEVICE={forced} is unsupported: VieNeu runs on CPU only"
        )
    suffix = " [forced]" if forced == "cpu" else ""
    return DeviceInfo("cpu", "cpu", f"CPU (ONNX Runtime){suffix}")
