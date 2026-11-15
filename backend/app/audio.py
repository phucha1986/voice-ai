"""Audio decoding for reference clips.

Pipeline for an uploaded reference clip:

1. Decode to mono float32 PCM.
2. Enforce the min/max reference duration.
3. Write a WAV that VieNeu consumes as ``ref_audio``.

No other cleanup is applied: VieNeu denoises the reference itself
(``denoise=True``), and generated audio is saved exactly as the model emits it.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

from .config import (
    MAX_REF_SECONDS,
    MIN_REF_SECONDS,
    TARGET_SAMPLE_RATE,
)

# ffmpeg installed via winget is not always on PATH yet; probe known locations.
_FFMPEG_FALLBACK = Path(
    r"C:\Users\hahan\AppData\Local\Microsoft\WinGet\Packages"
    r"\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0.2-full_build\bin\ffmpeg.exe"
)


def ffmpeg_path() -> str | None:
    """Return a usable ffmpeg binary, or ``None``."""
    found = shutil.which("ffmpeg")
    if found:
        return found
    if _FFMPEG_FALLBACK.exists():
        return str(_FFMPEG_FALLBACK)
    return None


def decode_to_mono(path: Path, sample_rate: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    """Decode any supported audio file to mono float32 at ``sample_rate``."""
    ff = ffmpeg_path()
    if ff:
        cmd = [
            ff,
            "-nostdin",
            "-y",
            "-i",
            str(path),
            "-f",
            "f32le",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-",
        ]
        proc = subprocess.run(cmd, capture_output=True, check=False)
        if proc.returncode != 0:
            raise ValueError(
                f"ffmpeg could not decode {path.name}: "
                f"{proc.stderr.decode(errors='replace').strip()[-300:]}"
            )
        return np.frombuffer(proc.stdout, dtype=np.float32).copy()

    # Fallbacks when ffmpeg is unavailable.
    try:
        data, sr = sf.read(str(path), always_2d=True)
        mono = data.mean(axis=1)
        if sr != sample_rate:
            import librosa

            mono = librosa.resample(mono.astype(np.float32), orig_sr=sr, target_sr=sample_rate)
        return mono.astype(np.float32)
    except Exception as exc:  # noqa: BLE001 - surface a single clear error
        raise ValueError(
            f"Could not decode {path.name} ({exc}). Install ffmpeg for mp3/m4a/ogg support."
        ) from exc


def fit_duration(audio: np.ndarray, sample_rate: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    """Reject clips that are too short, and cap overly long references."""
    seconds = audio.size / sample_rate
    if seconds < MIN_REF_SECONDS:
        raise ValueError(
            f"Audio tham chiếu quá ngắn ({seconds:.1f}s). Cần ít nhất {MIN_REF_SECONDS:.0f}s giọng nói."
        )
    if seconds > MAX_REF_SECONDS:
        return audio[: int(MAX_REF_SECONDS * sample_rate)]
    return audio


def preprocess_reference(source: Path, dest: Path, sample_rate: int = TARGET_SAMPLE_RATE) -> dict:
    """Decode ``source`` and write ``dest`` as a WAV reference clip.

    VieNeu denoises the reference during enrolment, so the clip is passed
    through unmodified apart from the duration limits.
    """
    audio = decode_to_mono(source, sample_rate)
    audio = fit_duration(audio, sample_rate)
    dest.parent.mkdir(parents=True, exist_ok=True)
    sf.write(dest, audio, sample_rate, subtype="PCM_16")
    return {
        "path": str(dest),
        "sample_rate": sample_rate,
        "duration_seconds": round(audio.size / sample_rate, 2),
        "channels": 1,
    }
