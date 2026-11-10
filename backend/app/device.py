"""Device selection: prefer GPU (CUDA / ROCm), fall back to CPU.

PyTorch exposes AMD ROCm builds through ``torch.cuda`` as well, so
``torch.cuda.is_available()`` covers both NVIDIA and AMD ROCm. ``torch-directml``
(DirectML on Windows/AMD) is used when installed.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DeviceInfo:
    device: str
    backend: str
    description: str

    @property
    def is_gpu(self) -> bool:
        return self.device != "cpu"


def detect_device() -> DeviceInfo:
    """Pick the best available PyTorch backend."""
    if torch.cuda.is_available():
        hip = getattr(torch.version, "hip", None)
        if hip:
            name = torch.cuda.get_device_name(0)
            return DeviceInfo("cuda", "rocm", f"ROCm {hip} · {name}")
        name = torch.cuda.get_device_name(0)
        return DeviceInfo("cuda", "cuda", f"CUDA · {name}")

    # Optional AMD DirectML backend (pip install torch-directml).
    try:
        import torch_directml  # type: ignore[import-untyped]

        if torch_directml.is_available():
            name = torch_directml.device_name(0)
            return DeviceInfo("directml", "directml", f"DirectML · {name}")
    except ImportError:
        pass

    return DeviceInfo("cpu", "cpu", f"CPU ({torch.__version__})")


def resolve_device() -> DeviceInfo:
    """Honour ``VOICE_AI_DEVICE`` (``cpu`` forces CPU even when a GPU exists)."""
    import os

    forced = os.environ.get("VOICE_AI_DEVICE", "").strip().lower()
    if forced == "cpu":
        return DeviceInfo("cpu", "cpu", f"CPU ({torch.__version__}) [forced]")
    if forced == "cuda":
        if torch.cuda.is_available():
            return detect_device()
        raise RuntimeError("VOICE_AI_DEVICE=cuda requested but no CUDA/ROCm device is available")
    return detect_device()
