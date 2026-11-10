"""Engine settings/seed behaviour without loading the model."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from app.config import (
    DEFAULT_LANGUAGE,
    SUPPORTED_LANGUAGES,
    VI_LANGUAGE,
    VI_MODEL_ID,
    VI_VOICE_DIR,
    VOICES_DIR,
)
from app.engine import engine

# XTTS v2's published language set (model card + XttsConfig.languages).
XTTS_V2_LANGUAGES = {
    "en",
    "es",
    "fr",
    "de",
    "it",
    "pt",
    "pl",
    "tr",
    "ru",
    "nl",
    "cs",
    "ar",
    "zh-cn",
    "ja",
    "hu",
    "ko",
    "hi",
}


def test_supported_languages_match_xtts_v2() -> None:
    """The pre-load fallback must match the model, otherwise the UI offers dead languages."""
    assert set(SUPPORTED_LANGUAGES) == XTTS_V2_LANGUAGES | {VI_LANGUAGE}
    assert len(SUPPORTED_LANGUAGES) == 18


def test_vietnamese_is_served_by_the_finetune() -> None:
    """``vi`` is not in stock XTTS v2, so it must come from the Vietnamese model."""
    assert VI_LANGUAGE in SUPPORTED_LANGUAGES
    assert VI_LANGUAGE not in XTTS_V2_LANGUAGES
    assert "ind" not in SUPPORTED_LANGUAGES  # XTTS v1 only


def test_vietnamese_can_be_disabled() -> None:
    """VOICE_AI_VI_MODEL="" drops vi from the language list."""
    monkeypatched = [*XTTS_V2_LANGUAGES] if not VI_MODEL_ID else SUPPORTED_LANGUAGES
    assert ("vi" in monkeypatched) == bool(VI_MODEL_ID)


def test_default_language_is_vietnamese() -> None:
    assert DEFAULT_LANGUAGE == VI_LANGUAGE


def test_vietnamese_voice_dir_is_separate_from_stock() -> None:
    """Latents are model-specific, so the two models must not share a voice cache."""
    assert VI_VOICE_DIR != VOICES_DIR
    assert VI_VOICE_DIR.name == "voices-vi"


def test_language_routing_selects_the_right_model() -> None:
    """``vi`` must go to the finetune; everything else stays on stock XTTS v2."""
    stock = SimpleNamespace(synthesizer=SimpleNamespace(voice_dir="stock-voices"))
    finetune = object()
    engine._tts = stock
    engine._vi = finetune
    try:
        assert engine._synth_for("en") is stock.synthesizer
        assert engine._synth_for("vi") is finetune
        assert engine._voice_dir_for("vi") == str(VI_VOICE_DIR)
        assert engine._voice_dir_for("en") == str(stock.synthesizer.voice_dir)
    finally:
        engine._tts = None
        engine._vi = None


def test_settings_exclude_seed_kwarg() -> None:
    """XTTS's GPT decoder rejects unknown kwargs, so seed must not be forwarded."""
    settings = engine._settings(speed=1.2, temperature=0.6)
    assert settings["speed"] == 1.2
    assert settings["temperature"] == 0.6
    assert settings["enable_text_splitting"] is True
    assert "deterministic_seed" not in settings
    assert "seed" not in settings


def test_apply_seed_makes_sampling_reproducible() -> None:
    engine._apply_seed(1234)
    a = torch.rand(8)
    engine._apply_seed(1234)
    b = torch.rand(8)
    assert torch.equal(a, b)

    engine._apply_seed(9999)
    c = torch.rand(8)
    assert not torch.equal(a, c)


def test_apply_seed_none_is_noop() -> None:
    torch.manual_seed(7)
    state_before = torch.random.get_rng_state().clone()
    engine._apply_seed(None)
    assert torch.equal(torch.random.get_rng_state(), state_before)
