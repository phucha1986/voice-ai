"""FastAPI backend for the Voice AI app (VieNeu-TTS engine).

Endpoints:
  GET  /health                     server + device status
  GET  /languages                  supported languages (Vietnamese only)
  GET  /presets                    built-in VieNeu voices (Vietnamese presets)
  POST /voices                     upload reference audio -> preprocess -> store
  GET  /voices                     list voices
  GET  /voices/{id}                voice metadata
  GET  /voices/{id}/audio          reference audio (preview)
  GET  /preview                    short audition sample for a voice or preset
  DELETE /voices/{id}
  POST /generate                   enqueue a synthesis job
  GET  /jobs                       list jobs
  GET  /jobs/{id}                  job status
  GET  /jobs/{id}/stream           SSE job status stream
  GET  /jobs/{id}/audio            generated audio (inline playback)
  GET  /jobs/{id}/download         generated audio as attachment
  POST /jobs/{id}/cancel
  DELETE /jobs/{id}
"""

from __future__ import annotations

import hashlib
import logging
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

from .config import (
    ALLOWED_EXTENSIONS,
    DEFAULT_LANGUAGE,
    MAX_TEXT_CHARS,
    MAX_UPLOAD_BYTES,
    PREVIEW_DIR,
    PREVIEW_TEXT,
    SUPPORTED_LANGUAGES,
    VIENEU_MODE,
    VIENEU_PRELOAD,
    TMP_DIR,
    VOICES_DIR,
)
from .audio import ffmpeg_path, preprocess_reference
from .engine import engine
from .models import GenerateRequest, HealthOut, JobOut, LanguagesOut, PresetVoiceOut, VoiceOut
from .queue import generation_queue
from .store import store

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("voice-ai.api")

@asynccontextmanager
async def lifespan(_app: FastAPI):
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    generation_queue.start()
    n = generation_queue.requeue_pending()
    if n:
        logger.info("Requeued %d pending job(s)", n)
    if VIENEU_PRELOAD:
        # VieNeu is CPU-only and takes ~12 s to load plus ~13 s to enrol a
        # reference, so warm it in the background instead of on the first job.
        _spawn_preload(_preload_vieneu)
    yield
    generation_queue.stop()


def _spawn_preload(target: Callable[[], None]) -> None:
    """Run ``target`` on a daemon thread so startup is not blocked."""
    threading.Thread(target=target, daemon=True).start()


def _preload_vieneu() -> None:
    """Load VieNeu and enrol the newest voice reference, off the event loop."""
    try:
        voices = store.list_voices()
        if voices:
            engine.enrol_vieneu_reference(voices[0]["ref_path"])
        else:
            engine.load_vieneu()
        logger.info("VieNeu preloaded")
    except Exception as exc:  # noqa: BLE001
        logger.warning("VieNeu preload failed: %s", exc)


def _enrol_new_voice(ref_path: str) -> None:
    try:
        engine.enrol_vieneu_reference(ref_path)
        logger.info("VieNeu enrolled new reference %s", ref_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("VieNeu enrolment failed for %s: %s", ref_path, exc)


app = FastAPI(title="Voice AI", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Status ------------------------------------------------------------


@app.get("/health", response_model=HealthOut)
def health() -> HealthOut:
    info = engine.device_info
    return HealthOut(
        ok=True,
        model=f"vieneu-{VIENEU_MODE}",
        device=info.device,
        backend=info.backend,
        device_description=info.description,
        gpu=info.is_gpu,
        ffmpeg=ffmpeg_path() is not None,
        model_loaded=engine.vieneu_loaded,
        languages=list(SUPPORTED_LANGUAGES),
        queued=store.queued_count(),
        vieneu_ready=engine.vieneu_loaded,
    )


@app.get("/languages", response_model=LanguagesOut)
def languages() -> LanguagesOut:
    return LanguagesOut(languages=list(SUPPORTED_LANGUAGES), default=DEFAULT_LANGUAGE)


@app.get("/presets", response_model=list[PresetVoiceOut])
def presets() -> list[PresetVoiceOut]:
    """Built-in VieNeu voices. Loading VieNeu takes ~12 s on first call; the
    startup preload thread normally warms it before the first request."""
    try:
        return [PresetVoiceOut(**p) for p in engine.preset_voices()]
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(503, f"VieNeu chưa sẵn sàng: {exc}") from exc


# --- Voices ------------------------------------------------------------


@app.post("/voices", response_model=VoiceOut)
async def create_voice(file: UploadFile = File(...), name: str | None = Form(None)) -> VoiceOut:
    suffix = Path(file.filename or "reference").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS or suffix == ".json":
        raise HTTPException(400, f"Không hỗ trợ định dạng '{suffix}'. Dùng wav/mp3/ogg/flac/m4a.")

    tmp = TMP_DIR / f"upload-{uuid.uuid4().hex}{suffix}"
    size = 0
    with open(tmp, "wb") as fh:
        while chunk := await file.read(1 << 20):
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                fh.close()
                tmp.unlink(missing_ok=True)
                raise HTTPException(413, f"File vượt quá {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
            fh.write(chunk)

    try:
        voice_id = uuid.uuid4().hex[:12]
        ref_path = VOICES_DIR / f"{voice_id}.wav"
        meta = preprocess_reference(tmp, ref_path)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    finally:
        tmp.unlink(missing_ok=True)

    record = store.create_voice(
        name=name or Path(file.filename or "voice").stem,
        ref_path=str(ref_path),
        meta={
            "duration_seconds": meta["duration_seconds"],
            "sample_rate": meta["sample_rate"],
            "source_filename": file.filename,
        },
    )
    logger.info("Voice %s created (%.1fs reference)", record["id"], meta["duration_seconds"])
    if VIENEU_PRELOAD:
        # Enrol the new reference in the background: VieNeu denoises + encodes
        # each new clip on first use (~13 s), which would otherwise land on the
        # first job generated from this voice.
        _spawn_preload(lambda: _enrol_new_voice(str(ref_path)))
    return VoiceOut(
        id=record["id"],
        name=record["name"],
        created_at=record["created_at"],
        duration_seconds=meta["duration_seconds"],
        sample_rate=meta["sample_rate"],
        source_filename=file.filename,
    )


@app.get("/voices", response_model=list[VoiceOut])
def list_voices() -> list[VoiceOut]:
    return [
        VoiceOut(
            id=v["id"],
            name=v["name"],
            created_at=v["created_at"],
            duration_seconds=v.get("duration_seconds", 0.0),
            sample_rate=v.get("sample_rate", 24000),
            source_filename=v.get("source_filename"),
        )
        for v in store.list_voices()
    ]


def _voice_or_404(voice_id: str) -> dict:
    voice = store.get_voice(voice_id)
    if voice is None:
        raise HTTPException(404, "Không tìm thấy giọng nói")
    return voice


@app.get("/voices/{voice_id}", response_model=VoiceOut)
def get_voice(voice_id: str) -> VoiceOut:
    v = _voice_or_404(voice_id)
    return VoiceOut(
        id=v["id"],
        name=v["name"],
        created_at=v["created_at"],
        duration_seconds=v.get("duration_seconds", 0.0),
        sample_rate=v.get("sample_rate", 24000),
        source_filename=v.get("source_filename"),
    )


@app.get("/voices/{voice_id}/audio")
def voice_audio(voice_id: str) -> FileResponse:
    v = _voice_or_404(voice_id)
    path = Path(v["ref_path"])
    if not path.exists():
        raise HTTPException(404, "Thiếu audio tham chiếu")
    return FileResponse(path, media_type="audio/wav", filename=f"{v['name']}-reference.wav")


@app.get("/preview")
def preview(voice_id: str | None = None, voice_preset: str | None = None) -> FileResponse:
    """Short audition sample ("nghe thử") for one voice or preset, cached on disk.

    Exactly one of ``voice_id`` / ``voice_preset`` must be given. The sample is
    synthesised once and reused; an uploaded voice is regenerated when its
    reference file changed after the cached sample was written.
    """
    preset = (voice_preset or "").strip() or None
    vid = (voice_id or "").strip() or None
    if preset and vid:
        raise HTTPException(422, "Chỉ chọn một giọng nghe thử: voice_id hoặc voice_preset")
    if preset:
        if not engine.has_preset(preset):
            raise HTTPException(404, f"Không tìm thấy giọng dựng sẵn '{preset}'")
        ref_path = ""
        cache_key = f"preset:{preset}"
    elif vid:
        v = _voice_or_404(vid)
        ref_path = v["ref_path"]
        cache_key = f"voice:{vid}"
    else:
        raise HTTPException(422, "Hãy chọn giọng nghe thử (voice_id hoặc voice_preset)")

    cache_key = f"{cache_key}|{PREVIEW_TEXT}"
    out = PREVIEW_DIR / f"{hashlib.sha256(cache_key.encode('utf-8')).hexdigest()[:16]}.wav"
    ref = Path(ref_path) if ref_path else None
    stale = True
    if out.exists():
        if ref is None or not ref.exists() or out.stat().st_mtime >= ref.stat().st_mtime:
            stale = False
    if stale:
        wav = engine.synthesize(
            PREVIEW_TEXT,
            ref_wav=ref_path,
            voice_id=vid or "",
            language=DEFAULT_LANGUAGE,
            voice_preset=preset,
            temperature=0.4,
            seed=42,
        )
        PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
        engine.save_wav(wav, str(out))
    return FileResponse(out, media_type="audio/wav", filename="voice-preview.wav")


@app.delete("/voices/{voice_id}")
def delete_voice(voice_id: str) -> dict:
    v = _voice_or_404(voice_id)
    Path(v["ref_path"]).unlink(missing_ok=True)
    # VieNeu caches reference enrolment internally by clip content, so there is
    # no voice-latent file to clean up here.
    store.delete_voice(voice_id)
    return {"deleted": voice_id}


# --- Generation --------------------------------------------------------


@app.post("/generate", response_model=JobOut)
def generate(req: GenerateRequest) -> JobOut:
    preset = (req.voice_preset or "").strip() or None
    if req.language != DEFAULT_LANGUAGE:
        raise HTTPException(
            400,
            f"Ngôn ngữ '{req.language}' không được hỗ trợ. VieNeu chỉ nói "
            f"tiếng Việt ('{DEFAULT_LANGUAGE}').",
        )
    if preset:
        # Built-in VieNeu voice.
        if not engine.has_preset(preset):
            raise HTTPException(404, f"Không tìm thấy giọng dựng sẵn '{preset}'")
        voice_id = ""
    else:
        if not req.voice_id:
            raise HTTPException(422, "Hãy chọn giọng nói (tải lên hoặc giọng dựng sẵn)")
        _voice_or_404(req.voice_id)
        voice_id = req.voice_id

    text = req.text.strip()
    if not text:
        raise HTTPException(422, "Văn bản trống")
    if len(text) > MAX_TEXT_CHARS:
        raise HTTPException(413, f"Văn bản vượt quá {MAX_TEXT_CHARS} ký tự")

    job = store.create_job(
        text=text,
        voice_id=voice_id,
        voice_preset=preset,
        language=req.language,
        speed=req.speed,
        temperature=req.temperature,
        seed=req.seed,
    )
    generation_queue.enqueue(job["id"])
    return JobOut(**job)


@app.get("/jobs", response_model=list[JobOut])
def list_jobs(limit: int = 50) -> list[JobOut]:
    return [JobOut(**j) for j in store.list_jobs(limit=limit)]


def _job_or_404(job_id: str) -> dict:
    job = store.get_job(job_id)
    if job is None:
        raise HTTPException(404, "Không tìm thấy công việc")
    return job


@app.get("/jobs/{job_id}", response_model=JobOut)
def get_job(job_id: str) -> JobOut:
    return JobOut(**_job_or_404(job_id))


@app.get("/jobs/{job_id}/stream")
async def job_stream(job_id: str) -> StreamingResponse:
    """SSE: pushes job status until it reaches a terminal state."""
    import asyncio
    import json

    async def events():
        last = None
        while True:
            job = store.get_job(job_id)
            if job is None:
                yield f"data: {json.dumps({'status': 'gone'})}\n\n"
                return
            if job["status"] != last:
                last = job["status"]
                payload = {k: job.get(k) for k in ("id", "status", "progress", "duration_seconds", "error")}
                yield f"data: {json.dumps(payload)}\n\n"
            if job["status"] in ("done", "failed", "cancelled"):
                return
            await asyncio.sleep(0.4)

    return StreamingResponse(events(), media_type="text/event-stream")


@app.get("/jobs/{job_id}/audio")
def job_audio(job_id: str) -> FileResponse:
    job = _job_or_404(job_id)
    if job["status"] != "done" or not job.get("audio_path"):
        raise HTTPException(409, f"Công việc đang {job['status']} — chưa có audio")
    path = Path(job["audio_path"])
    if not path.exists():
        raise HTTPException(404, "Thiếu file audio")
    return FileResponse(path, media_type="audio/wav")


@app.get("/jobs/{job_id}/download")
def job_download(job_id: str) -> FileResponse:
    job = _job_or_404(job_id)
    if job["status"] != "done" or not job.get("audio_path"):
        raise HTTPException(409, f"Công việc đang {job['status']} — chưa có audio")
    path = Path(job["audio_path"])
    if not path.exists():
        raise HTTPException(404, "Thiếu file audio")
    return FileResponse(path, media_type="audio/wav", filename=f"voice-ai-{job_id}.wav")


@app.post("/jobs/{job_id}/cancel", response_model=JobOut)
def cancel_job(job_id: str) -> JobOut:
    job = _job_or_404(job_id)
    if job["status"] in ("done", "failed", "cancelled"):
        return JobOut(**job)
    updated = store.update_job(job_id, status="cancelled", progress=0.0, finished_at=_now_iso())
    return JobOut(**updated)


@app.delete("/jobs/{job_id}")
def delete_job(job_id: str) -> dict:
    job = _job_or_404(job_id)
    if job.get("audio_path"):
        Path(job["audio_path"]).unlink(missing_ok=True)
    store.delete_job(job_id)
    return {"deleted": job_id}


def _now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()
