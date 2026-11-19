"""Device reporting for the health endpoint.

OmniVoice runs under PyTorch. This machine has an AMD GPU with no CUDA
support, so the engine is pinned to CPU float32. ``VOICE_AI_DEVICE`` is
honoured for compatibility; ``cpu`` is the only value it accepts.
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
    """OmniVoice is pinned to CPU (PyTorch float32); report the CPU device."""
    forced = os.environ.get("VOICE_AI_DEVICE", "").strip().lower()
    if forced and forced != "cpu":
        raise RuntimeError(
            f"VOICE_AI_DEVICE={forced} is unsupported: OmniVoice runs on CPU here"
        )
    suffix = " [forced]" if forced == "cpu" else ""
    return DeviceInfo("cpu", "cpu", f"CPU (PyTorch float32){suffix}")
