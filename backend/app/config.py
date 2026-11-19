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
# Cached "nghe thử" samples for voices and presets (regenerated on demand).
PREVIEW_DIR = DATA_DIR / "previews"
# Phrase auditioned by GET /preview.
PREVIEW_TEXT = (
    "Đây là Tiếng nói Việt Nam, phát thanh từ Hà Nội, "
    "Thủ đô nước Việt Nam Dân chủ Cộng hòa"
)

# --- Model -------------------------------------------------------------

# OmniVoice (k2-fsa/OmniVoice) is a 0.6B diffusion LM TTS; this backend uses
# the Vietnamese fine-tune splendor1811/omnivoice-vietnamese (Apache-2.0),
# trained on dolly-audio-1000h-vietnamese. Output is native 24 kHz, and voices
# are cloned from a reference clip via a cached VoiceClonePrompt.
DEFAULT_LANGUAGE = os.environ.get("VOICE_AI_LANGUAGE", "vi")
VI_LANGUAGE = "vi"
# The Vietnamese fine-tune is served for Vietnamese only.
SUPPORTED_LANGUAGES = [VI_LANGUAGE]
# OmniVoice expects full language names; map the API's short codes onto them.
LANGUAGE_NAMES = {"vi": "vietnamese"}

OMNIVOICE_MODEL = os.environ.get(
    "VOICE_AI_OMNIVOICE_MODEL", "splendor1811/omnivoice-vietnamese"
)
# Diffusion denoising steps: 32 is the upstream default (best quality). Lower
# values (e.g. 8-16) trade quality for speed on CPU.
OMNIVOICE_NUM_STEP = int(os.environ.get("VOICE_AI_OMNIVOICE_NUM_STEP", "32"))
# Whisper model used to auto-transcribe uploaded reference audio when building
# a voice clone prompt. large-v3-turbo is the upstream default and handles
# Vietnamese well; it downloads ~1.6 GB on first use.
OMNIVOICE_ASR_MODEL = os.environ.get(
    "VOICE_AI_OMNIVOICE_ASR_MODEL", "openai/whisper-large-v3-turbo"
)
# Load OmniVoice (and warm the newest voice prompt) in the background at
# startup instead of on the first job, which otherwise pays minutes of model
# download plus ASR transcription. Set VOICE_AI_PRELOAD=0 to keep it lazy.
OMNIVOICE_PRELOAD = os.environ.get("VOICE_AI_PRELOAD", "1") != "0"

# Built-in voices ("giọng dựng sẵn"). OmniVoice has no speaker presets, so
# these are voice-design instructions fed to ``generate(instruct=...)``.
# Voice design was trained on zh/en descriptors, so the instructs stay English;
# quality on the Vietnamese fine-tune is unverified — cloning is the reliable path.
OMNIVOICE_PRESETS: dict[str, dict] = {
    "Mỹ Duyên": {
        "instruct": "female, young adult, moderate pitch",
        "description": "Giọng nữ trong trẻo, phù hợp đọc truyện",
        "gender": "female",
        "featured": 1,
    },
    "Hoàng Nam": {
        "instruct": "male, young adult, moderate pitch",
        "description": "Giọng nam rõ ràng, trung bình",
        "gender": "male",
        "featured": 2,
    },
    "Thu Hà": {
        "instruct": "female, middle-aged, low pitch",
        "description": "Giọng nữ trầm, ấm",
        "gender": "female",
        "featured": 3,
    },
    "Đức Minh": {
        "instruct": "male, middle-aged, very low pitch",
        "description": "Giọng nam trầm, chậm rãi",
        "gender": "male",
        "featured": None,
    },
}

# --- Audio -------------------------------------------------------------

# OmniVoice outputs 24 kHz natively; reference audio is normalised to this
# rate as well. A resample is kept as a safety net if the audio tokenizer ever
# reports a different rate.
TARGET_SAMPLE_RATE = 24000
MIN_REF_SECONDS = 2.0
MAX_REF_SECONDS = float(os.environ.get("VOICE_AI_MAX_REF_SECONDS", 30.0))
MAX_UPLOAD_BYTES = int(os.environ.get("VOICE_AI_MAX_UPLOAD_BYTES", 25 * 1024 * 1024))

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".aac", ".opus", ".webm", ".mp4", ".json"}

# Generated audio is saved exactly as OmniVoice emits it: no noise gate and no
# gap collapsing. OmniVoice trims/denoises the reference itself when building a
# voice clone prompt (``preprocess_prompt=True``).

# --- Queue -------------------------------------------------------------

QUEUE_WORKERS = int(os.environ.get("VOICE_AI_QUEUE_WORKERS", "1"))
MAX_TEXT_CHARS = int(os.environ.get("VOICE_AI_MAX_TEXT_CHARS", 4000))
CHUNK_MAX_CHARS = 220
