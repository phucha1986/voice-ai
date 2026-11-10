"""Audio decoding and reference-audio preprocessing.

Pipeline for an uploaded reference clip:

1. Decode to mono float32 PCM (ffmpeg when available, else soundfile/librosa).
2. Resample to the model sample rate (24 kHz).
3. Strip leading/trailing silence.
4. Peak-normalise to -1 dBFS.
5. Enforce the min/max reference duration.
6. Write a clean WAV that XTTS consumes as ``ref_audio``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Iterator

import numpy as np
import soundfile as sf

from .config import MAX_GAP_MS, MAX_REF_SECONDS, MIN_REF_SECONDS, TARGET_SAMPLE_RATE

# ffmpeg installed via winget is not always on PATH yet; probe known locations.
# The BtbN *shared* build ships FFmpeg DLLs, which torchcodec needs on Windows;
# the Gyan build is static and only usable as a subprocess.
_FFMPEG_FALLBACK = Path(
    r"C:\Users\hahan\AppData\Local\Microsoft\WinGet\Packages"
    r"\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0.2-full_build\bin\ffmpeg.exe"
)


def ffmpeg_dll_dir() -> str | None:
    """Directory with FFmpeg shared DLLs (BtbN shared build), or ``None``."""
    local = os.environ.get("LOCALAPPDATA")
    if not local:
        return None
    pkgs = Path(local) / "Microsoft" / "WinGet" / "Packages"
    if not pkgs.exists():
        return None
    for pkg in sorted(pkgs.glob("BtbN.FFmpeg*")):
        for bin_dir in pkg.glob("*/bin"):
            if any(bin_dir.glob("avcodec-*.dll")):
                return str(bin_dir)
    return None


def ffmpeg_path() -> str | None:
    """Return a usable ffmpeg binary, or ``None``."""
    found = shutil.which("ffmpeg")
    if found:
        return found
    dll_dir = ffmpeg_dll_dir()
    if dll_dir:
        exe = Path(dll_dir) / "ffmpeg.exe"
        if exe.exists():
            return str(exe)
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


def strip_silence(
    audio: np.ndarray, threshold: float = 0.003, pad_ms: int = 120, sample_rate: int = TARGET_SAMPLE_RATE
) -> np.ndarray:
    """Trim leading/trailing silence, keeping a short pad on both ends."""
    if audio.size == 0:
        return audio
    above = np.abs(audio) > threshold
    idx = np.flatnonzero(above)
    if idx.size == 0:
        return audio
    start = max(0, idx[0] - pad_ms * sample_rate // 1000)
    end = min(audio.size, idx[-1] + pad_ms * sample_rate // 1000)
    return audio[start:end]


def normalize(audio: np.ndarray, peak_dbfs: float = -1.0) -> np.ndarray:
    """Scale to a target peak level in dBFS."""
    if audio.size == 0:
        return audio
    peak = float(np.max(np.abs(audio)))
    if peak <= 1e-6:
        return audio
    target = 10.0 ** (peak_dbfs / 20.0)
    return (audio * (target / peak)).astype(np.float32)


def collapse_long_gaps(
    audio: np.ndarray,
    *,
    silence_rms: float = 0.02,
    max_gap_ms: int = MAX_GAP_MS,
    sample_rate: int = TARGET_SAMPLE_RATE,
) -> np.ndarray:
    """Cap silence runs longer than ``max_gap_ms`` down to that length.

    XTTS leaves 200-380 ms of silence around each sentence chunk, so concatenating
    them produces an audible pause mid-speech. Silence shorter than the cap is a
    natural word boundary and is left untouched.
    """
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0 or max_gap_ms <= 0:
        return audio

    window = max(1, int(sample_rate * 0.02))
    starts = np.arange(0, audio.size, window)
    counts = np.minimum(window, audio.size - starts)
    rms = np.sqrt(np.add.reduceat(np.square(audio), starts) / counts)

    peak = float(np.max(np.abs(audio)))
    threshold = max(silence_rms, 0.05 * peak)
    quiet = rms < threshold

    max_samples = int(max_gap_ms * sample_rate / 1000)
    out: list[np.ndarray] = []
    index = 0

    while index < rms.size:
        end = index
        while end < rms.size and quiet[end] == quiet[index]:
            end += 1
        stop = starts[end] if end < starts.size else audio.size
        segment = audio[starts[index] : stop]
        if quiet[index] and segment.size > max_samples:
            # Keep the middle of the run so neither side loses speech.
            start = (segment.size - max_samples) // 2
            segment = segment[start : start + max_samples]
        out.append(segment)
        index = end

    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


def collapse_long_gaps_stream(
    chunks: Iterator[np.ndarray],
    *,
    silence_rms: float = 0.02,
    max_gap_ms: int = MAX_GAP_MS,
    sample_rate: int = TARGET_SAMPLE_RATE,
) -> Iterator[np.ndarray]:
    """Streaming variant of :func:`collapse_long_gaps`.

    The tail of the buffer is held back so a silence run that spans two chunks is
    still recognised as one run and capped once.
    """
    hold = int(max_gap_ms * sample_rate / 1000) + int(sample_rate * 0.02)
    buffer = np.zeros(0, dtype=np.float32)

    for chunk in chunks:
        buffer = np.concatenate([buffer, np.asarray(chunk, dtype=np.float32)])
        collapsed = collapse_long_gaps(
            buffer, silence_rms=silence_rms, max_gap_ms=max_gap_ms, sample_rate=sample_rate
        )
        if collapsed.size > hold:
            yield collapsed[: collapsed.size - hold]
            buffer = collapsed[collapsed.size - hold :]

    if buffer.size:
        yield collapse_long_gaps(
            buffer, silence_rms=silence_rms, max_gap_ms=max_gap_ms, sample_rate=sample_rate
        )


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
    """Decode, clean and write ``dest`` as a WAV reference clip."""
    audio = decode_to_mono(source, sample_rate)
    audio = strip_silence(audio, sample_rate=sample_rate)
    audio = fit_duration(audio, sample_rate)
    audio = normalize(audio)
    dest.parent.mkdir(parents=True, exist_ok=True)
    sf.write(dest, audio, sample_rate, subtype="PCM_16")
    return {
        "path": str(dest),
        "sample_rate": sample_rate,
        "duration_seconds": round(audio.size / sample_rate, 2),
        "channels": 1,
    }
