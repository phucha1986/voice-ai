"""Vietnamese text formatting for XTTS.

Vietnamese writes compound words (``nhanh nhẹn``, ``hoạt động``, ``thủ đô``) with a
space between syllables, but XTTS' BPE tokenizer encodes each syllable as its own
token separated by ``[SPACE]``. The Vietnamese finetune renders that boundary as
~100-160 ms of silence, so a compound is spoken as two separate words with an
audible pause in the middle ("nhanh" ... pause ... "nhẹn").

``underthesea`` segments the text into real Vietnamese words, and joining the
syllables of a compound with ``-`` (a single token in the finetune vocabulary, with
no BPE merges involving it) makes the model read them as one word. Compound syllables
that are capitalised -- proper nouns such as ``Hà Nội`` or ``Việt Nam`` -- are left
spaced, because joining them garbles the pronunciation ("Hà-Nội" -> "Hanoi").
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger("voice-ai.vietnamese")

#: Separator inserted between the syllables of a compound word.
COMPOUND_SEPARATOR = "-"

#: Punctuation that must be attached to the preceding word instead of spaced.
_PUNCT = {".", ",", "!", "?", ":", ";", '"', "'", "(", ")", "-", "…"}

#: Maximum syllables in a word we are willing to join; longer runs are segmentation
#: noise from underthesea and are left alone.
_MAX_SYLLABLES = 4

_underthesea = None


def _load_underthesea():
    global _underthesea
    if _underthesea is None:
        import underthesea

        _underthesea = underthesea
    return _underthesea


def segment_words(text: str) -> list[str]:
    """Segment ``text`` into Vietnamese words (compound syllables kept together).

    Returns ``[]`` for empty input. Raises if underthesea is unavailable or fails.
    """
    return list(_load_underthesea().word_tokenize(text))


def _joinable(word: str) -> bool:
    """True if ``word`` is a lowercase compound whose syllables should be joined."""
    syllables = word.split(" ")
    if len(syllables) < 2 or len(syllables) > _MAX_SYLLABLES:
        return False
    return all(s.isalpha() and not s[0].isupper() for s in syllables)


def format_vietnamese(text: str) -> str:
    """Rewrite ``text`` so XTTS reads Vietnamese compounds as single words.

    Falls back to the original text if segmentation is unavailable or fails, so a
    broken underthesea install never blocks synthesis.
    """
    if not text or not text.strip():
        return text
    try:
        words = segment_words(text)
    except Exception:
        logger.warning("Vietnamese word segmentation failed; using text as-is", exc_info=True)
        return text
    if not words:
        return text

    out: list[str] = []
    for word in words:
        if not word:
            continue
        if word in _PUNCT:
            out.append(word)
        elif _joinable(word):
            out.append(COMPOUND_SEPARATOR.join(word.split(" ")))
        else:
            out.append(word)

    text_out = ""
    for word in out:
        if word in _PUNCT:
            text_out += word
        else:
            text_out += (" " if text_out else "") + word
    return text_out


def split_sentences(text: str) -> list[str]:
    """Split on sentence punctuation, keeping the terminator attached to its sentence."""
    parts = re.split(r"(?<=[.!?;:])\s+", text)
    return [p for p in parts if p.strip()]
