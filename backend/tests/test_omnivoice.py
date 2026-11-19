"""OmniVoice routing/prompt-caching without downloading the model."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from app.config import OMNIVOICE_PRESETS, TARGET_SAMPLE_RATE
from app.engine import engine


class FakePrompt:
    """Stand-in for ``VoiceClonePrompt``: records saves, writes a marker file."""

    def __init__(self, tag: str = "prompt") -> None:
        self.tag = tag

    def save(self, path: str) -> None:
        with open(path, "wb") as fh:
            fh.write(self.tag.encode("utf-8"))


class FakeOmniVoice:
    """Stand-in for ``OmniVoice.from_pretrained(...)``: deterministic audio at a rate."""

    def __init__(self, sample_rate: int = 24_000) -> None:
        self.sampling_rate = sample_rate
        self.calls: list[dict] = []

    def generate(self, *, text, language=None, voice_clone_prompt=None,
                 instruct=None, speed=None, generation_config=None, **kwargs):
        self.calls.append({
            "text": text,
            "language": language,
            "voice_clone_prompt": voice_clone_prompt,
            "instruct": instruct,
            "speed": speed,
            "generation_config": generation_config,
        })
        return [np.full(self.sampling_rate, 0.5, dtype=np.float32)]  # 1 s at the model rate

    def create_voice_clone_prompt(self, ref_audio, ref_text=None, preprocess_prompt=True):
        self.calls.append({"create_prompt": ref_audio, "ref_text": ref_text})
        return FakePrompt()


def _use_fake_model(monkeypatch, fake: FakeOmniVoice):
    monkeypatch.setattr(engine, "load", lambda: None)
    monkeypatch.setattr(engine, "_generation_config", lambda: SimpleNamespace(num_step=8))
    engine._model = fake
    engine._prompts.clear()


def _restore_model() -> None:
    engine._model = None
    engine._prompts.clear()


def test_synthesize_clones_via_prompt_and_maps_language(monkeypatch) -> None:
    fake = FakeOmniVoice(24_000)
    _use_fake_model(monkeypatch, fake)
    prompt = FakePrompt()
    engine._prompts["v1"] = prompt
    try:
        progress: list[float] = []
        wav = engine.synthesize(
            "Con cáo nâu nhanh nhẹn.",
            ref_wav="ref.wav",
            voice_id="v1",
            language="vi",
            speed=1.25,
            progress_cb=progress.append,
        )
        assert len(wav) == TARGET_SAMPLE_RATE  # 1 s at 24 kHz, untouched
        assert progress == [1.0]
        call = fake.calls[0]
        assert call["text"] == "Con cáo nâu nhanh nhẹn."
        assert call["language"] == "vietnamese"
        assert call["voice_clone_prompt"] is prompt
        assert call["instruct"] is None
        assert call["speed"] == 1.25
        assert call["generation_config"].num_step == 8
    finally:
        _restore_model()


def test_preset_uses_instruct_instead_of_prompt(monkeypatch) -> None:
    fake = FakeOmniVoice(24_000)
    _use_fake_model(monkeypatch, fake)
    try:
        wav = engine.synthesize("Xin chào.", ref_wav="", voice_id="", voice_preset="Mỹ Duyên")
        assert len(wav) == TARGET_SAMPLE_RATE
        call = fake.calls[0]
        assert call["voice_clone_prompt"] is None
        assert call["instruct"] == OMNIVOICE_PRESETS["Mỹ Duyên"]["instruct"]
    finally:
        _restore_model()


def test_48khz_output_is_downsampled_to_target(monkeypatch) -> None:
    fake = FakeOmniVoice(48_000)
    _use_fake_model(monkeypatch, fake)
    engine._prompts["v1"] = FakePrompt()
    try:
        wav = engine.synthesize("Xin chào.", ref_wav="ref.wav", voice_id="v1", language="vi")
        assert len(wav) == TARGET_SAMPLE_RATE
    finally:
        _restore_model()


def test_output_is_saved_raw(monkeypatch, tmp_path) -> None:
    """No gap collapsing: long silence in the model output survives untouched."""
    fake = FakeOmniVoice(24_000)
    _use_fake_model(monkeypatch, fake)
    engine._prompts["v1"] = FakePrompt()
    try:
        audio = np.zeros(24_000, dtype=np.float32)
        audio[:6000] = 0.5
        audio[-6000:] = 0.5
        fake.generate = lambda *, text, **kw: [audio]
        wav = engine.synthesize("Em rất nhanh nhẹn.", ref_wav="ref.wav", voice_id="v1", language="vi")
        assert len(wav) == TARGET_SAMPLE_RATE
        assert np.abs(wav[6000:18000]).max() == 0.0
        assert np.abs(wav[:6000]).max() == 0.5
        assert np.abs(wav[-6000:]).max() == 0.5

        meta = engine.save_wav(wav, str(tmp_path / "out.wav"))
        assert meta["sample_rate"] == TARGET_SAMPLE_RATE
        assert meta["duration_seconds"] == 1.0
        written, sr = sf.read(str(tmp_path / "out.wav"))
        assert sr == TARGET_SAMPLE_RATE
        assert len(written) == TARGET_SAMPLE_RATE
    finally:
        _restore_model()


def test_prompt_is_built_once_and_saved(monkeypatch, tmp_path) -> None:
    fake = FakeOmniVoice(24_000)
    _use_fake_model(monkeypatch, fake)
    monkeypatch.setattr(engine, "prompt_path", lambda vid: tmp_path / f"{vid}.prompt.pt")
    try:
        p1 = engine.build_prompt("ref.wav", voice_id="v1")
        assert fake.calls[0]["create_prompt"] == "ref.wav"
        assert (tmp_path / "v1.prompt.pt").exists()

        # Second call hits the in-memory cache — no second transcription.
        p2 = engine.build_prompt("ref.wav", voice_id="v1")
        assert p2 is p1
        assert len(fake.calls) == 1
    finally:
        _restore_model()


def test_prompt_reloads_from_disk_cache(monkeypatch, tmp_path) -> None:
    import omnivoice

    fake = FakeOmniVoice(24_000)
    _use_fake_model(monkeypatch, fake)
    monkeypatch.setattr(engine, "prompt_path", lambda vid: tmp_path / f"{vid}.prompt.pt")
    loaded: list[str] = []

    def fake_load(cls, path, map_location="cpu"):
        loaded.append(path)
        return FakePrompt("from-disk")

    monkeypatch.setattr(omnivoice.VoiceClonePrompt, "load", classmethod(fake_load))
    try:
        p1 = engine.build_prompt("ref.wav", voice_id="v1")
        assert p1.tag == "prompt"  # built via ASR and saved
        engine._prompts.clear()  # simulate a restart
        p2 = engine.build_prompt("ref.wav", voice_id="v1")
        assert p2.tag == "from-disk"
        assert loaded == [str(tmp_path / "v1.prompt.pt")]
        assert sum("create_prompt" in c for c in fake.calls) == 1  # built once, then reloaded
    finally:
        _restore_model()


def test_drop_prompt_clears_cache_and_file(monkeypatch, tmp_path) -> None:
    _use_fake_model(monkeypatch, FakeOmniVoice(24_000))
    monkeypatch.setattr(engine, "prompt_path", lambda vid: tmp_path / f"{vid}.prompt.pt")
    try:
        engine._prompts["v1"] = FakePrompt()
        (tmp_path / "v1.prompt.pt").write_bytes(b"x")
        engine.drop_prompt("v1")
        assert "v1" not in engine._prompts
        assert not (tmp_path / "v1.prompt.pt").exists()
    finally:
        _restore_model()


def test_presets_listed_featured_first() -> None:
    presets = engine.preset_voices()
    assert [p["name"] for p in presets] == ["Mỹ Duyên", "Hoàng Nam", "Thu Hà", "Đức Minh"]
    assert engine.has_preset("Mỹ Duyên") is True
    assert engine.has_preset("KhôngTồnTại") is False
    assert engine.has_preset("") is False


def test_stream_yields_single_chunk(monkeypatch) -> None:
    fake = FakeOmniVoice(24_000)
    _use_fake_model(monkeypatch, fake)
    engine._prompts["v1"] = FakePrompt()
    try:
        chunks = list(engine.synthesize_stream("Xin chào.", ref_wav="ref.wav", voice_id="v1"))
        assert len(chunks) == 1
        assert len(chunks[0]) == TARGET_SAMPLE_RATE
    finally:
        _restore_model()


def test_startup_preload_warms_newest_voice(monkeypatch) -> None:
    """Startup must warm OmniVoice and build the newest voice's prompt so job #1 is fast."""
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr("app.main.store", SimpleNamespace(list_voices=lambda: [
        {"id": "new", "ref_path": "new.wav"},
        {"id": "old", "ref_path": "old.wav"},
    ]))
    monkeypatch.setattr(engine, "warm_reference", lambda ref, vid="": calls.append((ref, vid)))
    monkeypatch.setattr("app.main.OMNIVOICE_PRELOAD", True)
    monkeypatch.setattr("app.main._spawn_preload", lambda target: target())

    from app.main import app

    with TestClient(app):
        assert calls == [("new.wav", "new")]


def test_startup_preload_can_be_disabled(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(engine, "warm_reference", lambda ref, vid="": calls.append((ref, vid)))
    monkeypatch.setattr("app.main.OMNIVOICE_PRELOAD", False)
    monkeypatch.setattr("app.main._spawn_preload", lambda target: target())

    from app.main import app

    with TestClient(app):
        assert calls == []
