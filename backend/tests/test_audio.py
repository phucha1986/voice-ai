"""Unit tests for audio preprocessing (no model required)."""

from __future__ import annotations

import numpy as np
import soundfile as sf

from app.audio import decode_to_mono, ffmpeg_path, fit_duration, normalize, preprocess_reference, strip_silence
from app.config import MIN_REF_SECONDS, TARGET_SAMPLE_RATE


def _tone(seconds: float, freq: float = 220.0, sr: int = TARGET_SAMPLE_RATE, silence_edges: float = 0.0) -> np.ndarray:
    n = int(seconds * sr)
    t = np.arange(n) / sr
    audio = 0.5 * np.sin(2 * np.pi * freq * t).astype(np.float32)
    if silence_edges:
        edge = int(silence_edges * sr)
        audio[:edge] = 0.0
        audio[-edge:] = 0.0
    return audio


def _write(path, audio: np.ndarray, sr: int = TARGET_SAMPLE_RATE) -> None:
    sf.write(path, audio, sr, subtype="PCM_16")


def test_ffmpeg_is_available() -> None:
    assert ffmpeg_path() is not None, "ffmpeg should be installed for mp3/m4a decoding"


def test_decode_wav_to_mono(tmp_path) -> None:
    stereo = np.stack([_tone(2.0), _tone(2.0, 440.0)], axis=1)
    _write(tmp_path / "st.wav", stereo)
    mono = decode_to_mono(tmp_path / "st.wav")
    assert mono.ndim == 1
    assert mono.dtype == np.float32
    assert abs(len(mono) - 2 * TARGET_SAMPLE_RATE) <= TARGET_SAMPLE_RATE // 10


def test_decode_mp3_via_ffmpeg(tmp_path) -> None:
    ff = ffmpeg_path()
    assert ff is not None
    src = tmp_path / "in.wav"
    _write(src, _tone(3.0))
    mp3 = tmp_path / "out.mp3"
    import subprocess

    subprocess.run([ff, "-y", "-i", str(src), str(mp3)], capture_output=True, check=True)
    mono = decode_to_mono(mp3)
    assert len(mono) > TARGET_SAMPLE_RATE  # ~3s at 24 kHz


def test_strip_silence_trims_edges() -> None:
    audio = _tone(4.0, silence_edges=1.0)
    trimmed = strip_silence(audio)
    assert len(trimmed) < len(audio)
    assert len(trimmed) > 2 * TARGET_SAMPLE_RATE  # speech core survives
    assert float(np.max(np.abs(trimmed))) > 0.4  # speech kept
    assert abs(trimmed[0]) <= 0.003  # leading pad is near-silent


def test_normalize_targets_peak() -> None:
    audio = _tone(2.0) * 0.05
    out = normalize(audio)
    peak = float(np.max(np.abs(out)))
    # -1 dBFS == 10**(-1/20) == 0.891
    assert 0.88 < peak <= 0.9


def test_fit_duration_rejects_short_and_caps_long() -> None:
    short = _tone(1.0)
    try:
        fit_duration(short)
        raise AssertionError("expected ValueError for short clip")
    except ValueError as exc:
        assert "quá ngắn" in str(exc)

    long = _tone(MIN_REF_SECONDS + 40.0)
    capped = fit_duration(long)
    assert len(capped) <= int(30.0 * TARGET_SAMPLE_RATE)


def test_preprocess_reference_writes_clean_wav(tmp_path) -> None:
    src = tmp_path / "raw.wav"
    _write(src, _tone(5.0, silence_edges=0.8) * 0.02)
    dest = tmp_path / "ref.wav"
    meta = preprocess_reference(src, dest)

    assert dest.exists()
    assert meta["sample_rate"] == TARGET_SAMPLE_RATE
    assert meta["channels"] == 1
    assert meta["duration_seconds"] >= MIN_REF_SECONDS
    data, sr = sf.read(str(dest))
    assert sr == TARGET_SAMPLE_RATE
    assert float(np.max(np.abs(data))) <= 1.0
