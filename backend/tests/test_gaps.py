"""Tests for inter-sentence gap collapsing (no model required)."""

from __future__ import annotations

import numpy as np

from app.audio import collapse_long_gaps, collapse_long_gaps_stream
from app.config import MAX_GAP_MS, TARGET_SAMPLE_RATE


def _speech(seconds: float, amp: float = 0.5, freq: float = 220.0) -> np.ndarray:
    n = int(seconds * TARGET_SAMPLE_RATE)
    t = np.arange(n) / TARGET_SAMPLE_RATE
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * TARGET_SAMPLE_RATE), dtype=np.float32)


def _gaps(audio: np.ndarray, threshold: float = 0.02) -> list[float]:
    """Silence-run lengths (ms) in ``audio``."""
    window = int(TARGET_SAMPLE_RATE * 0.02)
    rms = np.sqrt(
        np.add.reduceat(np.square(audio), np.arange(0, audio.size, window)) / window
    )
    runs: list[float] = []
    count = 0
    for quiet in rms < threshold:
        if quiet:
            count += 1
        elif count:
            runs.append(count * 20)
            count = 0
    if count:
        runs.append(count * 20)
    return runs


def test_caps_long_gaps_keeps_word_boundaries() -> None:
    audio = np.concatenate([_speech(1.0), _silence(0.38), _speech(1.0), _silence(0.06), _speech(1.0)])
    out = collapse_long_gaps(audio)

    gaps = _gaps(out)
    assert gaps, "silence should still be present"
    assert max(gaps) <= MAX_GAP_MS + 20, f"gaps too long: {gaps}"
    assert min(gaps) < MAX_GAP_MS, "short word-boundary pauses must survive"

    # No speech is lost.
    assert np.count_nonzero(np.abs(out) > 0.1) >= np.count_nonzero(np.abs(audio) > 0.1)


def test_leading_and_trailing_silence_capped() -> None:
    audio = np.concatenate([_silence(0.5), _speech(1.0), _silence(0.6)])
    out = collapse_long_gaps(audio)
    assert out.size < audio.size
    assert len(out) <= int((0.15 + 1.0 + 0.15) * TARGET_SAMPLE_RATE) + TARGET_SAMPLE_RATE // 10


def test_short_silence_untouched() -> None:
    audio = np.concatenate([_speech(0.5), _silence(0.04), _speech(0.5)])
    out = collapse_long_gaps(audio)
    assert out.size == audio.size


def test_empty_and_silent_audio() -> None:
    assert collapse_long_gaps(np.zeros(0, dtype=np.float32)).size == 0
    silent = _silence(1.0)
    assert collapse_long_gaps(silent).size <= int(MAX_GAP_MS * TARGET_SAMPLE_RATE / 1000) + 40


def test_stream_matches_blocking() -> None:
    audio = np.concatenate([_speech(1.0), _silence(0.38), _speech(1.0), _silence(0.06), _speech(1.0)])
    blocking = collapse_long_gaps(audio)

    pieces = [audio[i : i + 900] for i in range(0, audio.size, 900)]
    streamed = np.concatenate(list(collapse_long_gaps_stream(iter(pieces))))

    assert abs(streamed.size - blocking.size) <= TARGET_SAMPLE_RATE // 10
    assert max(_gaps(streamed)) <= MAX_GAP_MS + 20
