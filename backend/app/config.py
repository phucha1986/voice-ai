"""Central configuration for the Voice AI backend."""

from __future__ import annotations

import os
from pathlib import Path

# --- Paths -------------------------------------------------------------

_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = Path(os.environ.get("VOICE_AI_DATA_DIR", _ROOT / "data"))
VOICES_DIR = DATA_DIR / "voices"
AUDIO_DIR = DATA_DIR / "audio"
TMP_DIR = DATA_DIR / "tmp"

# --- Model -------------------------------------------------------------

# Coqui TTS model registry name for XTTS v2 (voice cloning). Overridable via env.
MODEL_NAME = os.environ.get("VOICE_AI_MODEL", "tts_models/multilingual/multi-dataset/xtts_v2")
DEFAULT_LANGUAGE = os.environ.get("VOICE_AI_LANGUAGE", "vi")

# XTTS v2 itself cannot speak Vietnamese, so "vi" is served by a community XTTS
# v2 Vietnamese finetune on Hugging Face (its vocab has a [vi] token). Set
# VOICE_AI_VI_MODEL to "" to disable Vietnamese entirely.
VI_LANGUAGE = "vi"
VI_MODEL_ID = os.environ.get("VOICE_AI_VI_MODEL", "Nhat1106/xtts-vietnamese")
# Inference config inside the snapshot; config.json is a *training* config that
# points at files the repo does not ship.
VI_CONFIG_FILE = os.environ.get("VOICE_AI_VI_CONFIG", "config2.json")
# Latents are model-specific, so the Vietnamese model caches voices separately.
VI_VOICE_DIR = DATA_DIR / "voices-vi"

# Fallback used before the model loads: XTTS v2's 17 languages plus "vi" from
# the Vietnamese finetune.
XTTS_V2_LANGUAGES = [
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
]
SUPPORTED_LANGUAGES = [*XTTS_V2_LANGUAGES, VI_LANGUAGE] if VI_MODEL_ID else list(XTTS_V2_LANGUAGES)

# --- Audio -------------------------------------------------------------

# XTTS v2 outputs at 24 kHz; reference audio is normalised to this rate.
TARGET_SAMPLE_RATE = 24000
MIN_REF_SECONDS = 2.0
MAX_REF_SECONDS = float(os.environ.get("VOICE_AI_MAX_REF_SECONDS", 30.0))
MAX_UPLOAD_BYTES = int(os.environ.get("VOICE_AI_MAX_UPLOAD_BYTES", 25 * 1024 * 1024))

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".aac", ".opus", ".webm", ".mp4", ".json"}

# XTTS leaves 200-380 ms of silence at sentence boundaries, which sounds like a
# pause mid-speech. Longer silence runs are capped to this; shorter pauses (word
# boundaries) are left alone so the prosody stays natural.
MAX_GAP_MS = int(os.environ.get("VOICE_AI_MAX_GAP_MS", "150"))

# --- Queue -------------------------------------------------------------

QUEUE_WORKERS = int(os.environ.get("VOICE_AI_QUEUE_WORKERS", "1"))
MAX_TEXT_CHARS = int(os.environ.get("VOICE_AI_MAX_TEXT_CHARS", 4000))
CHUNK_MAX_CHARS = 220
