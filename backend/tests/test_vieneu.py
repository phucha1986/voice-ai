"""VieNeu-TTS routing/resampling without downloading the model."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import soundfile as sf

from app.config import TARGET_SAMPLE_RATE
from app.engine import engine


class FakeVieneu:
    """Stand-in for ``Vieneu(...)``: reports a rate and returns deterministic audio."""

    def __init__(self, sample_rate: int = 48_000) -> None:
        self.sample_rate = sample_rate
        self.calls: list[dict] = []

    def infer(self, text: str, *, ref_audio: str, **kwargs) -> np.ndarray:
        self.calls.append({"text": text, "ref_audio": ref_audio, **kwargs})
        return np.full(self.sample_rate, 0.5, dtype=np.float32)  # 1 s at the model rate

    def infer_stream(self, text: str, *, ref_audio: str, **kwargs):
        self.calls.append({"text": text, "ref_audio": ref_audio, **kwargs, "stream": True})
        for _ in range(3):
            yield np.full(self.sample_rate // 3, 0.5, dtype=np.float32)


def test_vieneu_is_off_by_default(monkeypatch) -> None:
    monkeypatch.setattr("app.engine.VIENEU_ENABLED", False)
    assert engine._use_vieneu("vi") is False
    assert engine._use_vieneu("en") is False


def test_vieneu_only_serves_its_own_languages(monkeypatch) -> None:
    monkeypatch.setattr("app.engine.VIENEU_ENABLED", True)
    assert engine._use_vieneu("vi") is True
    assert engine._use_vieneu("en") is False  # English stays on XTTS v2


def test_kwargs_skip_watermark_and_map_sampling(monkeypatch) -> None:
    """resemble-perth is not installed, and v3 Turbo has no speed/seed knobs."""
    monkeypatch.setattr("app.engine.VIENEU_MODE", "v3turbo")
    kwargs = engine._vieneu_kwargs(temperature=0.75, speed=1.5, seed=42)
    assert kwargs["apply_watermark"] is False
    assert kwargs["denoise"] is True
    assert kwargs["temperature"] == 0.75
    assert "speed" not in kwargs
    assert "seed" not in kwargs

    monkeypatch.setattr("app.engine.VIENEU_MODE", "v3nano")
    kwargs = engine._vieneu_kwargs(temperature=0.75, speed=1.5, seed=42)
    assert kwargs["speed"] == 1.5
    assert kwargs["seed"] == 42
    assert "temperature" not in kwargs


def test_temperature_is_clamped(monkeypatch) -> None:
    monkeypatch.setattr("app.engine.VIENEU_MODE", "v3turbo")
    assert engine._vieneu_kwargs(temperature=0.0, speed=1.0, seed=None)["temperature"] == 0.3
    assert engine._vieneu_kwargs(temperature=1.0, speed=1.0, seed=None)["temperature"] == 0.9


def test_48khz_output_is_downsampled_to_target(monkeypatch, tmp_path) -> None:
    fake = FakeVieneu(48_000)
    monkeypatch.setattr("app.engine.VIENEU_ENABLED", True)
    monkeypatch.setattr("app.engine.VIENEU_MODE", "v3turbo")
    engine._vieneu = fake
    monkeypatch.setattr(engine, "load_vieneu", lambda: None)
    try:
        progress: list[float] = []
        wav = engine.synthesize(
            "Con cáo nâu nhanh nhẹn.",
            ref_wav="ref.wav",
            voice_id="ignored",
            language="vi",
            progress_cb=progress.append,
        )
        assert len(wav) == TARGET_SAMPLE_RATE  # 1 s of audio at 24 kHz
        assert progress == [1.0]
        assert fake.calls[0]["text"] == "Con cáo nâu nhanh nhẹn."
        assert fake.calls[0]["ref_audio"] == "ref.wav"

        meta = engine.save_wav(wav, str(tmp_path / "out.wav"))
        assert meta["sample_rate"] == TARGET_SAMPLE_RATE
        assert meta["duration_seconds"] == 1.0
        written_sr, _ = sf.read(str(tmp_path / "out.wav"))
        assert len(written_sr) == TARGET_SAMPLE_RATE
    finally:
        engine._vieneu = None


def test_native_24khz_output_is_not_resampled(monkeypatch) -> None:
    fake = FakeVieneu(24_000)
    monkeypatch.setattr("app.engine.VIENEU_ENABLED", True)
    monkeypatch.setattr("app.engine.VIENEU_MODE", "v3nano")
    engine._vieneu = fake
    monkeypatch.setattr(engine, "load_vieneu", lambda: None)
    try:
        wav = engine.synthesize("Xin chào.", ref_wav="ref.wav", voice_id="v", language="vi")
        assert len(wav) == TARGET_SAMPLE_RATE
        assert np.all(wav == 0.5)  # untouched, no resampling applied
    finally:
        engine._vieneu = None


def test_stream_is_resampled_and_flushed(monkeypatch) -> None:
    """Chunked 48 kHz output must reassemble into the exact 24 kHz length."""
    fake = FakeVieneu(48_000)
    monkeypatch.setattr("app.engine.VIENEU_ENABLED", True)
    monkeypatch.setattr("app.engine.VIENEU_MODE", "v3turbo")
    engine._vieneu = fake
    monkeypatch.setattr(engine, "load_vieneu", lambda: None)
    try:
        chunks = list(engine.synthesize_stream("Xin chào.", ref_wav="ref.wav", voice_id="v", language="vi"))
        assert sum(len(c) for c in chunks) == TARGET_SAMPLE_RATE
        assert fake.calls[0]["stream"] is True
    finally:
        engine._vieneu = None


def test_other_languages_stay_on_xtts(monkeypatch) -> None:
    """Enabling VieNeu must not change routing for the 17 XTTS languages."""
    monkeypatch.setattr("app.engine.VIENEU_ENABLED", True)
    stock = SimpleNamespace(synthesizer=SimpleNamespace(voice_dir="stock-voices"))
    engine._tts = stock
    try:
        assert engine._synth_for("en") is stock.synthesizer
    finally:
        engine._tts = None
