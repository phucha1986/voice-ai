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

# VieNeu-TTS is the only engine: Vietnamese voice cloning on a torch-free ONNX
# CPU runtime (https://github.com/pitvnga1/vieneu-tts). Its phonemizer is
# phoneme-level (sea-g2p), so Vietnamese compound words ("nhanh nhẹn") are read
# as one word instead of one syllable per token.
DEFAULT_LANGUAGE = os.environ.get("VOICE_AI_LANGUAGE", "vi")
VI_LANGUAGE = "vi"
# VieNeu speaks Vietnamese only — there is no other language to offer.
SUPPORTED_LANGUAGES = [VI_LANGUAGE]

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

# --- Audio -------------------------------------------------------------

# VieNeu v3 Turbo outputs at 48 kHz and is downsampled to this rate before
# saving; reference audio is normalised to this rate as well.
TARGET_SAMPLE_RATE = 24000
MIN_REF_SECONDS = 2.0
MAX_REF_SECONDS = float(os.environ.get("VOICE_AI_MAX_REF_SECONDS", 30.0))
MAX_UPLOAD_BYTES = int(os.environ.get("VOICE_AI_MAX_UPLOAD_BYTES", 25 * 1024 * 1024))

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".aac", ".opus", ".webm", ".mp4", ".json"}

# Generated audio is saved exactly as VieNeu emits it: no noise gate and no gap
# collapsing. VieNeu denoises the reference itself (``denoise=True``).

# --- Queue -------------------------------------------------------------

QUEUE_WORKERS = int(os.environ.get("VOICE_AI_QUEUE_WORKERS", "1"))
MAX_TEXT_CHARS = int(os.environ.get("VOICE_AI_MAX_TEXT_CHARS", 4000))
CHUNK_MAX_CHARS = 220
