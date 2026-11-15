"""VieNeu-TTS synthesis engine — the only engine in this backend.

VieNeu-TTS (https://github.com/pitvnga1/vieneu-tts) speaks Vietnamese and
clones voices from a reference WAV on a torch-free ONNX CPU runtime. Its
phonemizer is phoneme-level (sea-g2p), so Vietnamese compound words
("nhanh nhẹn") are read as one word instead of one syllable per token.

v3 Turbo outputs at 48 kHz and is downsampled to ``TARGET_SAMPLE_RATE`` before
saving; reference enrolment is cached inside VieNeu by clip content, so there
is no voice-file management.
"""

from __future__ import annotations

import logging
import threading
from typing import Callable, Iterator

import numpy as np
import soundfile as sf
import soxr

from .config import (
    DEFAULT_LANGUAGE,
    SUPPORTED_LANGUAGES,
    TARGET_SAMPLE_RATE,
    VIENEU_MODE,
    VIENEU_PRECISION,
    VIENEU_THREADS,
)
from .device import DeviceInfo, resolve_device

logger = logging.getLogger("voice-ai.engine")

ProgressCb = Callable[[float], None]


class Engine:
    """VieNeu model wrapper. One inference at a time (model is not reentrant)."""

    def __init__(self) -> None:
        self._vieneu = None
        self._device_info: DeviceInfo | None = None
        # Reentrant: synthesize() holds the lock and load_vieneu() re-acquires it.
        self._lock = threading.RLock()

    # --- Loading -------------------------------------------------------

    def load_vieneu(self) -> None:
        """Download and load VieNeu-TTS (torch-free ONNX on CPU)."""
        with self._lock:
            if self._vieneu is not None:
                return
            logger.info("Loading VieNeu-TTS %s (%s) on CPU", VIENEU_MODE, VIENEU_PRECISION)
            from vieneu import Vieneu

            if VIENEU_MODE == "v3nano":
                # v3 Nano is ONNX/CPU only and takes no device/backend/precision args.
                self._vieneu = Vieneu(mode="v3nano", threads=VIENEU_THREADS)
            else:
                # This machine has an AMD GPU with no CUDA support, so pin the
                # torch-free ONNX CPU runtime instead of letting it probe devices.
                self._vieneu = Vieneu(
                    mode=VIENEU_MODE,
                    device="cpu",
                    backend="onnx",
                    precision=VIENEU_PRECISION,
                    threads=VIENEU_THREADS,
                )
            logger.info("VieNeu model ready (sample rate %s Hz)", self._vieneu.sample_rate)

    def enrol_vieneu_reference(self, ref_wav: str) -> None:
        """Warm VieNeu's reference cache for ``ref_wav``.

        The speaker encoder is loaded lazily on the first clone and the clip is
        denoised/enrolled on every new clip (VieNeu memoises enrolment by clip
        content), so the first job with a voice pays ~13 s of extra work. Enroling
        the newest reference at startup removes that cost from the first job.
        """
        self.load_vieneu()
        model = self._vieneu
        assert model is not None
        with self._lock:
            model.encode_reference(ref_wav, denoise=True)

    def preset_voices(self) -> list[dict]:
        """Built-in VieNeu voices (Vietnamese presets), editors' picks first.

        Returns ``[{name, description, gender, featured}]``. VieNeu must be loaded;
        an empty list when the model exposes no presets.
        """
        self.load_vieneu()
        model = self._vieneu
        assert model is not None
        presets = getattr(model, "_preset_voices", {})
        # Editors' picks (featured 1..N) first in that order, then the rest in
        # insertion order — same ordering VieNeu's own dropdown uses.
        ordered = sorted(
            presets.items(),
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
        """True if ``name`` is a VieNeu preset voice (aliases resolved by the model)."""
        if not name:
            return False
        self.load_vieneu()
        model = self._vieneu
        assert model is not None
        resolve = getattr(model, "resolve_voice_name", None)
        if resolve is not None:
            return resolve(name) is not None
        return name in getattr(model, "_preset_voices", {})

    @property
    def loaded(self) -> bool:
        return self._vieneu is not None

    @property
    def vieneu_loaded(self) -> bool:
        return self._vieneu is not None

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
        """VieNeu v3 Turbo outputs 48 kHz; everything downstream assumes the target rate."""
        if src_rate == TARGET_SAMPLE_RATE or len(wav) == 0:
            return np.asarray(wav, dtype=np.float32)
        return np.asarray(soxr.resample(wav, src_rate, TARGET_SAMPLE_RATE), dtype=np.float32)

    def _vieneu_kwargs(self, *, temperature: float, speed: float, seed: int | None) -> dict:
        """Map the request knobs onto VieNeu's parameters.

        ``resemble-perth`` is not installed, so the watermark step is a no-op and
        is skipped rather than importing another dependency. v3 Turbo has no
        speed/seed knobs, so those are only forwarded to v3 Nano.
        """
        kwargs: dict = {"denoise": True, "apply_watermark": False}
        if VIENEU_MODE == "v3nano":
            kwargs["speed"] = speed
            if seed is not None:
                kwargs["seed"] = seed
        else:
            # VieNeu samples at temperature 0.8 by default; requests are normally
            # lower, so clamp into the range the model stays intelligible.
            kwargs["temperature"] = min(max(temperature, 0.3), 0.9)
        return kwargs

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

        ``voice_preset`` selects a built-in VieNeu voice instead of cloning
        ``ref_wav``. ``language`` is accepted for API compatibility; VieNeu
        speaks Vietnamese only and ignores it.
        """
        return self._synthesize_vieneu(
            text,
            ref_wav=ref_wav,
            voice_preset=voice_preset,
            speed=speed,
            temperature=temperature,
            seed=seed,
            progress_cb=progress_cb,
        )

    def _synthesize_vieneu(
        self,
        text: str,
        *,
        ref_wav: str,
        voice_preset: str | None = None,
        speed: float,
        temperature: float,
        seed: int | None,
        progress_cb: ProgressCb | None,
    ) -> np.ndarray:
        """VieNeu path: phoneme-level phonemizer + type-aware chunking.

        Reference enrolment is cached inside VieNeu by clip content, so there is
        no voice file to manage. VieNeu still leaves long silence between its
        chunks, so ``collapse_long_gaps`` caps them.

        With ``voice_preset``, VieNeu uses the built-in preset voice (speaker
        embedding + pre-encoded reference codes) instead of cloning ``ref_wav``.
        """
        self.load_vieneu()
        model = self._vieneu
        assert model is not None
        kwargs = self._vieneu_kwargs(temperature=temperature, speed=speed, seed=seed)

        with self._lock:
            if voice_preset:
                wav = model.infer(text, voice=voice_preset, **kwargs)
            else:
                wav = model.infer(text, ref_audio=ref_wav, **kwargs)

        if progress_cb:
            progress_cb(1.0)
        return self._resample_to_target(np.asarray(wav, dtype=np.float32), model.sample_rate)

    def _synthesize_vieneu_stream(
        self,
        text: str,
        *,
        ref_wav: str,
        voice_preset: str | None = None,
        speed: float,
        temperature: float,
        seed: int | None,
    ) -> Iterator[np.ndarray]:
        self.load_vieneu()
        model = self._vieneu
        assert model is not None
        kwargs = self._vieneu_kwargs(temperature=temperature, speed=speed, seed=seed)

        src_rate = model.sample_rate
        resampler = (
            soxr.ResampleStream(src_rate, TARGET_SAMPLE_RATE, 1, dtype="float32", quality="HQ")
            if src_rate != TARGET_SAMPLE_RATE
            else None
        )

        with self._lock:
            stream = (
                model.infer_stream(text, voice=voice_preset, **kwargs)
                if voice_preset
                else model.infer_stream(text, ref_audio=ref_wav, **kwargs)
            )
            for chunk in stream:
                if resampler is None:
                    yield np.asarray(chunk, dtype=np.float32)
                else:
                    out = resampler.resample_chunk(np.asarray(chunk, dtype=np.float32))
                    if len(out):
                        yield np.asarray(out, dtype=np.float32)
            if resampler is not None:
                tail = resampler.resample_chunk(np.zeros(0, dtype=np.float32), last=True)
                if len(tail):
                    yield np.asarray(tail, dtype=np.float32)

    def synthesize_stream(
        self,
        text: str,
        *,
        ref_wav: str,
        voice_preset: str | None = None,
        speed: float = 1.0,
        temperature: float = 0.75,
        seed: int | None = None,
    ) -> Iterator[np.ndarray]:
        """Yield audio chunks as they are generated (for SSE streaming)."""
        yield from self._synthesize_vieneu_stream(
            text, ref_wav=ref_wav, voice_preset=voice_preset, speed=speed, temperature=temperature, seed=seed
        )

    def save_wav(self, wav: np.ndarray, path: str) -> dict:
        """Write float32 PCM to a WAV file at the target sample rate.

        The audio is saved exactly as VieNeu produced it — no noise gate and no
        gap collapsing.
        """
        sr = TARGET_SAMPLE_RATE
        sf.write(path, wav, sr, subtype="PCM_16")
        return {"path": path, "sample_rate": sr, "duration_seconds": round(len(wav) / sr, 2)}


engine = Engine()
