"""OmniVoice synthesis engine — the only engine in this backend.

OmniVoice (https://github.com/k2-fsa/OmniVoice) is a 0.6B diffusion LM TTS.
This backend runs the Vietnamese fine-tune ``splendor1811/omnivoice-vietnamese``
on CPU in float32 (the machine has an AMD GPU with no CUDA support). Voices are
cloned from the uploaded reference clip via a ``VoiceClonePrompt``; Whisper ASR
auto-transcribes the clip when the prompt is built, and the prompt is cached on
disk next to the reference WAV so it survives restarts.

OmniVoice outputs 24 kHz natively, so no resampling is needed in practice.
There is no temperature/seed knob — diffusion decoding is deterministic — and
``speed`` maps onto ``generate(speed=...)``.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
import soundfile as sf

from .config import (
    DEFAULT_LANGUAGE,
    LANGUAGE_NAMES,
    OMNIVOICE_ASR_MODEL,
    OMNIVOICE_MODEL,
    OMNIVOICE_NUM_STEP,
    OMNIVOICE_PRESETS,
    SUPPORTED_LANGUAGES,
    TARGET_SAMPLE_RATE,
    VOICES_DIR,
)
from .device import DeviceInfo, resolve_device

logger = logging.getLogger("voice-ai.engine")

ProgressCb = Callable[[float], None]


class Engine:
    """OmniVoice model wrapper. One inference at a time (model is not reentrant)."""

    def __init__(self) -> None:
        self._model = None
        self._device_info: DeviceInfo | None = None
        # Reentrant: synthesize() holds the lock and load() re-acquires it.
        self._lock = threading.RLock()
        # Voice clone prompts cached in memory, keyed by voice id (or ref path).
        self._prompts: dict[str, object] = {}

    # --- Loading -------------------------------------------------------

    def load(self) -> None:
        """Download and load OmniVoice (float32 on CPU) plus the Whisper ASR.

        float16 is not used because CPU kernels are unsupported/slow there;
        float32 keeps the 0.6B model usable on this machine.
        """
        with self._lock:
            if self._model is not None:
                return
            logger.info("Loading OmniVoice %s on CPU (float32)", OMNIVOICE_MODEL)
            import torch
            from omnivoice import OmniVoice

            self._model = OmniVoice.from_pretrained(
                OMNIVOICE_MODEL,
                device_map="cpu",
                dtype=torch.float32,
                load_asr=True,
                asr_model_name=OMNIVOICE_ASR_MODEL,
                asr_device="cpu",
            )
            logger.info("OmniVoice ready (sample rate %s Hz)", self._model.sampling_rate)

    def prompt_path(self, voice_id: str) -> Path:
        """On-disk cache file for a voice's clone prompt."""
        return VOICES_DIR / f"{voice_id}.prompt.pt"

    def build_prompt(self, ref_wav: str, voice_id: str = "") -> object:
        """Build (and cache) a ``VoiceClonePrompt`` for ``ref_wav``.

        Whisper auto-transcribes the reference on first use, so the first job
        with a new voice pays the ASR cost; the prompt is saved next to the
        reference WAV and reloaded on later runs instead of re-transcribing.
        """
        self.load()
        model = self._model
        assert model is not None
        key = voice_id or ref_wav
        with self._lock:
            cached = self._prompts.get(key)
            if cached is not None:
                return cached
            path = self.prompt_path(key) if voice_id else None
            if path is not None and path.exists():
                from omnivoice import VoiceClonePrompt

                prompt = VoiceClonePrompt.load(str(path))
            else:
                prompt = model.create_voice_clone_prompt(ref_audio=ref_wav)
                if path is not None:
                    prompt.save(str(path))
            self._prompts[key] = prompt
            return prompt

    def warm_reference(self, ref_wav: str, voice_id: str = "") -> None:
        """Pre-build the clone prompt for a reference so the first job is fast."""
        self.build_prompt(ref_wav, voice_id)

    def drop_prompt(self, voice_id: str) -> None:
        """Forget a voice's cached prompt and delete its cache file."""
        self._prompts.pop(voice_id, None)
        self.prompt_path(voice_id).unlink(missing_ok=True)

    def preset_voices(self) -> list[dict]:
        """Built-in voice-design presets, editors' picks first.

        Returns ``[{name, description, gender, featured}]``. OmniVoice has no
        speaker presets; these map onto ``generate(instruct=...)`` descriptors.
        """
        ordered = sorted(
            OMNIVOICE_PRESETS.items(),
            key=lambda kv: (kv[1].get("featured") is None, kv[1].get("featured") or 0),
        )
        return [
            {
                "name": name,
                "description": v.get("description", ""),
                "gender": v.get("gender", ""),
                "featured": v.get("featured"),
            }
            for name, v in ordered
        ]

    def has_preset(self, name: str) -> bool:
        """True if ``name`` is a built-in voice-design preset."""
        return bool(name) and name in OMNIVOICE_PRESETS

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def engine_ready(self) -> bool:
        return self._model is not None

    @property
    def device_info(self) -> DeviceInfo:
        if self._device_info is None:
            self._device_info = resolve_device()
        return self._device_info

    @property
    def languages(self) -> list[str]:
        return list(SUPPORTED_LANGUAGES)

    # --- Internals -----------------------------------------------------

    def _resample_to_target(self, wav: np.ndarray, src_rate: int) -> np.ndarray:
        """OmniVoice emits 24 kHz natively; resample only if the tokenizer ever reports otherwise."""
        if src_rate == TARGET_SAMPLE_RATE or len(wav) == 0:
            return np.asarray(wav, dtype=np.float32)
        import soxr

        return np.asarray(soxr.resample(wav, src_rate, TARGET_SAMPLE_RATE), dtype=np.float32)

    def _generation_config(self):
        """Diffusion config for every request: upstream defaults with the configured step count."""
        from omnivoice import OmniVoiceGenerationConfig

        return OmniVoiceGenerationConfig(num_step=OMNIVOICE_NUM_STEP)

    def _language_name(self, language: str) -> str:
        """Map the API's short code onto the full name OmniVoice expects."""
        return LANGUAGE_NAMES.get(language, language)

    # --- Synthesis -----------------------------------------------------

    def synthesize(
        self,
        text: str,
        *,
        ref_wav: str,
        voice_id: str = "",
        language: str = DEFAULT_LANGUAGE,
        speed: float = 1.0,
        temperature: float = 0.75,
        seed: int | None = None,
        voice_preset: str | None = None,
        progress_cb: ProgressCb | None = None,
    ) -> np.ndarray:
        """Synthesize ``text`` cloned from ``ref_wav``. Returns float32 PCM at ``TARGET_SAMPLE_RATE``.

        ``voice_preset`` selects a built-in voice-design instruction instead of
        cloning ``ref_wav``. ``temperature``/``seed`` are accepted for API
        compatibility; diffusion decoding is deterministic and ignores them.
        """
        return self._synthesize_omnivoice(
            text,
            ref_wav=ref_wav,
            voice_id=voice_id,
            voice_preset=voice_preset,
            language=language,
            speed=speed,
            progress_cb=progress_cb,
        )

    def _synthesize_omnivoice(
        self,
        text: str,
        *,
        ref_wav: str,
        voice_id: str = "",
        voice_preset: str | None = None,
        language: str,
        speed: float,
        progress_cb: ProgressCb | None,
    ) -> np.ndarray:
        """OmniVoice path: voice cloning via a cached prompt, or voice design via instruct.

        With ``voice_preset``, ``generate(instruct=...)`` designs the voice and
        ``ref_wav`` is ignored; otherwise the reference clip's ``VoiceClonePrompt``
        is reused (built once, cached on disk).
        """
        self.load()
        model = self._model
        assert model is not None

        if voice_preset:
            instruct = OMNIVOICE_PRESETS[voice_preset]["instruct"]
            prompt = None
        else:
            instruct = None
            prompt = self.build_prompt(ref_wav, voice_id)

        with self._lock:
            wavs = model.generate(
                text=text,
                language=self._language_name(language),
                voice_clone_prompt=prompt,
                instruct=instruct,
                speed=speed,
                generation_config=self._generation_config(),
            )

        wav = np.asarray(wavs[0], dtype=np.float32) if wavs else np.zeros(0, dtype=np.float32)
        if progress_cb:
            progress_cb(1.0)
        return self._resample_to_target(wav, int(model.sampling_rate))

    def synthesize_stream(
        self,
        text: str,
        *,
        ref_wav: str,
        voice_id: str = "",
        voice_preset: str | None = None,
        language: str = DEFAULT_LANGUAGE,
        speed: float = 1.0,
        temperature: float = 0.75,
        seed: int | None = None,
    ) -> Iterator[np.ndarray]:
        """Yield audio chunks for SSE. OmniVoice has no incremental decoding, so
        the whole utterance arrives as a single chunk."""
        wav = self.synthesize(
            text,
            ref_wav=ref_wav,
            voice_id=voice_id,
            voice_preset=voice_preset,
            language=language,
            speed=speed,
        )
        if len(wav):
            yield wav

    def save_wav(self, wav: np.ndarray, path: str) -> dict:
        """Write float32 PCM to a WAV file at the target sample rate.

        The audio is saved exactly as OmniVoice produced it — no noise gate and
        no gap collapsing.
        """
        sr = TARGET_SAMPLE_RATE
        sf.write(path, wav, sr, subtype="PCM_16")
        return {"path": path, "sample_rate": sr, "duration_seconds": round(len(wav) / sr, 2)}


engine = Engine()
