"""XTTS v2 synthesis engine.

Loads the model once (lazily) on the best available device and exposes
blocking and streaming synthesis with voice cloning from a reference WAV.

Stock XTTS v2 speaks 17 languages and cannot speak Vietnamese, so Vietnamese
requests are routed to a separate Vietnamese XTTS finetune (``VI_MODEL_ID``)
that is loaded lazily on first use.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
import soundfile as sf
import torch

from .audio import collapse_long_gaps, collapse_long_gaps_stream
from .vietnamese import format_vietnamese
from .config import (
    DEFAULT_LANGUAGE,
    MODEL_NAME,
    VI_CONFIG_FILE,
    VI_LANGUAGE,
    VI_MODEL_ID,
    VI_VOICE_DIR,
)
from .device import DeviceInfo, resolve_device

logger = logging.getLogger("voice-ai.engine")

ProgressCb = Callable[[float], None]


class Engine:
    """Lazy-loaded XTTS model wrapper. One inference at a time (model is not reentrant)."""

    def __init__(self) -> None:
        self._tts = None
        self._vi = None
        self._device_info: DeviceInfo | None = None
        self._sample_rate: int | None = None
        self._languages: list[str] | None = None
        # Reentrant: synthesize() holds the lock and load() re-acquires it via _synth().
        self._lock = threading.RLock()

    # --- Loading -------------------------------------------------------

    def _ensure_ffmpeg_dlls(self) -> None:
        """torchcodec loads FFmpeg shared DLLs; on Windows they must be registered
        via os.add_dll_directory (PATH alone is not enough)."""
        from .audio import ffmpeg_dll_dir

        dll_dir = ffmpeg_dll_dir()
        if dll_dir:
            try:
                os.add_dll_directory(dll_dir)
            except OSError:
                pass

    def load(self) -> None:
        """Download (first run) and load the model on the selected device."""
        with self._lock:
            if self._tts is not None:
                return
            info = resolve_device()
            logger.info("Loading %s on %s", MODEL_NAME, info.description)
            # XTTS ships under the Coqui CPML; TTS prompts interactively for
            # agreement on first load unless this env var is set.
            os.environ.setdefault("COQUI_TOS_AGREED", "1")
            self._ensure_ffmpeg_dlls()
            from TTS.api import TTS

            tts = TTS(model_name=MODEL_NAME, progress_bar=True, gpu=False)
            if info.is_gpu:
                tts.to(info.device)
            self._tts = tts
            self._device_info = info
            self._sample_rate = tts.synthesizer.output_sample_rate
            langs = sorted(tts.languages or [DEFAULT_LANGUAGE])
            if VI_MODEL_ID and VI_LANGUAGE not in langs:
                langs = [VI_LANGUAGE, *langs]
            self._languages = langs
            logger.info("Model ready (sample rate %s Hz, device %s)", self._sample_rate, info.device)

    def load_vietnamese(self) -> None:
        """Download and load the Vietnamese XTTS finetune (only needed for ``vi``)."""
        with self._lock:
            if self._vi is not None:
                return
            if not VI_MODEL_ID:
                raise RuntimeError("Vietnamese is disabled (VOICE_AI_VI_MODEL is empty)")
            info = resolve_device()
            logger.info("Loading Vietnamese XTTS finetune %s on %s", VI_MODEL_ID, info.description)
            os.environ.setdefault("COQUI_TOS_AGREED", "1")
            self._ensure_ffmpeg_dlls()
            from huggingface_hub import snapshot_download
            from TTS.utils.synthesizer import Synthesizer

            # The repo ships model.pth + vocab.json + config2.json; the rest of
            # the snapshot (training configs, READMEs) is not needed.
            snapshot_dir = Path(
                snapshot_download(
                    VI_MODEL_ID,
                    allow_patterns=["model.pth", "*.pth", "vocab.json", VI_CONFIG_FILE],
                )
            )
            VI_VOICE_DIR.mkdir(parents=True, exist_ok=True)
            synth = Synthesizer(
                tts_checkpoint=snapshot_dir,
                tts_config_path=str(snapshot_dir / VI_CONFIG_FILE),
                voice_dir=VI_VOICE_DIR,
                use_cuda=info.is_gpu,
            )
            # The finetune's tokenizer knows the [vi] BPE token but the shared
            # XTTS tokenizer table has no Vietnamese entry, so the character
            # limit and the text normaliser have to be added on the instance.
            tokenizer = synth.tts_model.tokenizer
            tokenizer.char_limits.setdefault(VI_LANGUAGE, 250)
            from TTS.tts.layers.xtts.tokenizer import collapse_whitespace, lowercase

            tokenizer.preprocess_text = lambda txt, lang: collapse_whitespace(lowercase(txt))
            self._vi = synth
            logger.info(
                "Vietnamese model ready (sample rate %s Hz, device %s)",
                synth.output_sample_rate,
                info.device,
            )

    @property
    def loaded(self) -> bool:
        return self._tts is not None

    @property
    def vietnamese_loaded(self) -> bool:
        return self._vi is not None

    @property
    def device_info(self) -> DeviceInfo:
        if self._device_info is None:
            self._device_info = resolve_device()
        return self._device_info

    @property
    def sample_rate(self) -> int:
        self.load()
        assert self._sample_rate is not None
        return self._sample_rate

    @property
    def languages(self) -> list[str]:
        self.load()
        assert self._languages is not None
        return self._languages

    # --- Internals -----------------------------------------------------

    def _synth(self):
        self.load()
        assert self._tts is not None
        return self._tts.synthesizer

    def _synth_for(self, language: str):
        """Return the synthesizer that can speak ``language``. Vietnamese needs
        its own finetune, which is loaded on first use."""
        if language == VI_LANGUAGE:
            self.load_vietnamese()
            assert self._vi is not None
            return self._vi
        return self._synth()

    def _voice_dir_for(self, language: str) -> str:
        """Voice latents are model-specific, so each model keeps its own cache."""
        if language == VI_LANGUAGE:
            return str(VI_VOICE_DIR)
        return str(self._synth().voice_dir)

    def _clone(self, model, ref_wav: str, voice_id: str, voice_dir: str):
        """Return (latents, embedding), reusing the cached voice file when present."""
        try:
            voice = model.load_voice_file(voice_id, voice_dir)
        except FileNotFoundError:
            voice = model.clone_voice([ref_wav], voice_id, voice_dir)
        return voice["gpt_conditioning_latents"], voice["speaker_embedding"]

    def _settings(self, *, speed: float, temperature: float) -> dict:
        return {
            "speed": speed,
            "temperature": temperature,
            "repetition_penalty": 2.0,
            "top_k": 50,
            "top_p": 0.85,
            "length_penalty": 1.0,
            # XTTS hard-caps ~400 tokens per call, so let it split long sentences.
            "enable_text_splitting": True,
        }

    def _apply_seed(self, seed: int | None) -> None:
        """Seed sampling so the same seed reproduces the same audio.

        ``deterministic_seed`` is not accepted by the GPT decoder (transformers
        rejects unused model kwargs), so seeding the RNG is the supported way.
        """
        if seed is None:
            return
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # --- Synthesis -----------------------------------------------------

    def synthesize(
        self,
        text: str,
        *,
        ref_wav: str,
        voice_id: str,
        language: str = DEFAULT_LANGUAGE,
        speed: float = 1.0,
        temperature: float = 0.75,
        seed: int | None = None,
        progress_cb: ProgressCb | None = None,
    ) -> np.ndarray:
        """Synthesize ``text`` cloned from ``ref_wav``. Returns float32 PCM at the model rate."""
        synth = self._synth_for(language)
        model = synth.tts_model
        if language == VI_LANGUAGE:
            text = format_vietnamese(text)
        sentences = synth.split_into_sentences(text, language)
        settings = self._settings(speed=speed, temperature=temperature)

        with self._lock:
            self._apply_seed(seed)
            cond_latent, embedding = self._clone(model, ref_wav, voice_id, self._voice_dir_for(language))
            chunks: list[np.ndarray] = []
            for i, sentence in enumerate(sentences):
                out = model.inference(sentence, language, cond_latent, embedding, **settings)
                wav = out["wav"]
                if isinstance(wav, torch.Tensor):
                    wav = wav.cpu().numpy()
                chunks.append(np.asarray(wav, dtype=np.float32))
                if progress_cb:
                    progress_cb((i + 1) / len(sentences))

        if not chunks:
            return np.zeros(0, dtype=np.float32)
        return collapse_long_gaps(np.concatenate(chunks))

    def synthesize_stream(
        self,
        text: str,
        *,
        ref_wav: str,
        voice_id: str,
        language: str = DEFAULT_LANGUAGE,
        speed: float = 1.0,
        temperature: float = 0.75,
        seed: int | None = None,
        chunk_size: int = 20,
    ) -> Iterator[np.ndarray]:
        """Yield audio chunks as they are generated (for SSE streaming)."""
        synth = self._synth_for(language)
        model = synth.tts_model
        settings = self._settings(speed=speed, temperature=temperature)
        if language == VI_LANGUAGE:
            text = format_vietnamese(text)
        sentences = synth.split_into_sentences(text, language)

        with self._lock:
            self._apply_seed(seed)
            cond_latent, embedding = self._clone(model, ref_wav, voice_id, self._voice_dir_for(language))

            def raw_chunks() -> Iterator[np.ndarray]:
                for sentence in sentences:
                    for chunk in model.inference_stream(
                        sentence,
                        language,
                        cond_latent,
                        embedding,
                        stream_chunk_size=chunk_size,
                        **settings,
                    ):
                        if isinstance(chunk, torch.Tensor):
                            chunk = chunk.cpu().numpy()
                        yield np.asarray(chunk, dtype=np.float32)

            yield from collapse_long_gaps_stream(raw_chunks())

    def save_wav(self, wav: np.ndarray, path: str) -> dict:
        """Write float32 PCM to a WAV file at the model sample rate."""
        sr = self.sample_rate
        sf.write(path, wav, sr, subtype="PCM_16")
        return {"path": path, "sample_rate": sr, "duration_seconds": round(len(wav) / sr, 2)}


engine = Engine()
