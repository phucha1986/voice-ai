"""Tests for Vietnamese compound-word formatting (no model required)."""

from __future__ import annotations

import pytest

from app import vietnamese
from app.vietnamese import COMPOUND_SEPARATOR, format_vietnamese, segment_words, split_sentences


def test_compound_words_are_joined() -> None:
    assert format_vietnamese("Tôi rất nhanh nhẹn.") == "Tôi rất nhanh-nhẹn."
    assert format_vietnamese("Máy hoạt động tốt") == "Máy hoạt-động tốt"


def test_proper_nouns_stay_spaced() -> None:
    assert format_vietnamese("Hà Nội là thủ đô của Việt Nam.") == "Hà Nội là thủ-đô của Việt Nam."


def test_punctuation_is_attached_without_stray_spaces() -> None:
    out = format_vietnamese("Xin chào! Đây là bài kiểm tra nhân bản giọng nói thật.")
    assert out == "Xin chào! Đây là bài kiểm-tra nhân-bản giọng nói thật."
    assert " !" not in out and " ." not in out


def test_empty_and_whitespace_input() -> None:
    assert format_vietnamese("") == ""
    assert format_vietnamese("   ") == "   "


def test_output_is_stable_and_idempotent() -> None:
    once = format_vietnamese("Con cáo nâu nhanh nhẹn nhảy qua con chó lười.")
    assert format_vietnamese(once) == once
    assert COMPOUND_SEPARATOR in once


def test_falls_back_when_segmentation_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(_text: str) -> list[str]:
        raise RuntimeError("underthesea broken")

    monkeypatch.setattr(vietnamese, "segment_words", boom)
    assert format_vietnamese("nhanh nhẹn") == "nhanh nhẹn"


def test_falls_back_when_underthesea_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing_import():
        raise ImportError("no module named underthesea")

    monkeypatch.setattr(vietnamese, "_load_underthesea", missing_import)
    assert format_vietnamese("nhanh nhẹn") == "nhanh nhẹn"


def test_long_syllable_runs_are_not_joined(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(vietnamese, "segment_words", lambda _t: ["a b c d e"])
    assert format_vietnamese("whatever") == "a b c d e"


def test_split_sentences_keeps_terminators() -> None:
    assert split_sentences("Xin chào! Tôi tên là Nam.") == ["Xin chào!", "Tôi tên là Nam."]
    assert split_sentences("Một câu duy nhất") == ["Một câu duy nhất"]


def test_segment_words_returns_compounds_intact() -> None:
    words = segment_words("Tôi rất nhanh nhẹn.")
    assert "nhanh nhẹn" in words
    assert words[-1] == "."
