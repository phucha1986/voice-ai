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

# --- VieNeu-TTS (optional second Vietnamese engine) --------------------

# VieNeu-TTS (https://github.com/pitvnga1/vieneu-tts) speaks Vietnamese and
# English with voice cloning on a torch-free ONNX CPU runtime. Its phonemizer is
# phoneme-level (sea-g2p), so Vietnamese compound words ("nhanh nhẹn") are read
# as one word instead of one syllable per token, and its pause handling is
# type-aware, so neither format_vietnamese() nor collapse_long_gaps() is needed.
# Enabled by default; set VOICE_AI_VIENEU=0 to serve vi with the XTTS finetune
# again (it is ~5x faster on this machine, but pauses between sentences).
VIENEU_ENABLED = os.environ.get("VOICE_AI_VIENEU", "1") != "0"
# "v3turbo" (48 kHz, best quality) or "v3nano" (24 kHz, 48M params, for weak CPUs).
VIENEU_MODE = os.environ.get("VOICE_AI_VIENEU_MODE", "v3turbo")
# ONNX weights for v3turbo. Measured on a 6-core CPU (blocking infer, RTF =
# synthesis seconds / audio seconds): fp32 ~0.60-0.63, int8 ~0.38-0.43 (~1.45x
# faster). int8 needs a VNNI-capable CPU and drops tone-mark fidelity on
# compound words ("nhanh nhẹn" comes back as "nhanh nhẹ"), so fp32 stays default
# and int8 is opt-in when speed matters more than pronunciation.
VIENEU_PRECISION = os.environ.get("VOICE_AI_VIENEU_PRECISION", "fp32")
# ONNX Runtime intra-op threads; 0 = about half the physical cores, which is the
# fastest setting measured (3 threads is ~25% slower, 12 threads is no faster).
VIENEU_THREADS = int(os.environ.get("VOICE_AI_VIENEU_THREADS", "0"))
# Load VieNeu (and enrol the newest voice) in the background at startup instead
# of on the first job, which otherwise pays ~12 s of model download plus ~2 s of
# reference enrolment. Set VOICE_AI_VIENEU_PRELOAD=0 to keep it lazy.
VIENEU_PRELOAD = os.environ.get("VOICE_AI_VIENEU_PRELOAD", "1") != "0"
# Languages VieNeu serves. English stays on XTTS v2 (it is faster there).
VIENEU_LANGUAGES = ["vi"]

# --- Audio -------------------------------------------------------------

# XTTS v2 outputs at 24 kHz; reference audio is normalised to this rate.
# VieNeu v3 Turbo outputs at 48 kHz and is downsampled to this rate before saving.
TARGET_SAMPLE_RATE = 24000
MIN_REF_SECONDS = 2.0
MAX_REF_SECONDS = float(os.environ.get("VOICE_AI_MAX_REF_SECONDS", 30.0))
MAX_UPLOAD_BYTES = int(os.environ.get("VOICE_AI_MAX_UPLOAD_BYTES", 25 * 1024 * 1024))

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".aac", ".opus", ".webm", ".mp4", ".json"}

# XTTS leaves 200-380 ms of silence at sentence boundaries, which sounds like a
# pause mid-speech. Longer silence runs are capped to this; shorter pauses (word
# boundaries) are left alone so the prosody stays natural.
# 100 ms removes the audible pauses without garbling speech -- measured on the
# Vietnamese finetune, 150 ms still leaves 140 ms pauses and 80 ms mispronounces
# proper nouns ("Hà Nội" -> "tụi đô").
MAX_GAP_MS = int(os.environ.get("VOICE_AI_MAX_GAP_MS", "100"))

# --- Queue -------------------------------------------------------------

QUEUE_WORKERS = int(os.environ.get("VOICE_AI_QUEUE_WORKERS", "1"))
MAX_TEXT_CHARS = int(os.environ.get("VOICE_AI_MAX_TEXT_CHARS", 4000))
CHUNK_MAX_CHARS = 220
