"""FastAPI backend for the Voice AI app.

Endpoints:
  GET  /health                     server + device status
  GET  /languages                  supported XTTS languages
  POST /voices                     upload reference audio -> preprocess -> store
  GET  /voices                     list voices
  GET  /voices/{id}                voice metadata
  GET  /voices/{id}/audio          reference audio (preview)
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

import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

from .config import (
    ALLOWED_EXTENSIONS,
    AUDIO_DIR,
    DEFAULT_LANGUAGE,
    MAX_TEXT_CHARS,
    MAX_UPLOAD_BYTES,
    MODEL_NAME,
    SUPPORTED_LANGUAGES,
    VIENEU_ENABLED,
    VIENEU_MODE,
    TMP_DIR,
    VI_VOICE_DIR,
    VOICES_DIR,
)
from .audio import ffmpeg_path, preprocess_reference
from .engine import engine
from .models import GenerateRequest, HealthOut, JobOut, LanguagesOut, VoiceOut
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
    if engine.device_info.is_gpu:
        # Warm the model eagerly on GPU so the first job is not slow.
        try:
            engine.load()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Model preload failed: %s", exc)
    yield
    generation_queue.stop()


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
        model=MODEL_NAME,
        device=info.device,
        backend=info.backend,
        device_description=info.description,
        gpu=info.is_gpu,
        ffmpeg=ffmpeg_path() is not None,
        model_loaded=engine.loaded,
        languages=engine.languages if engine.loaded else SUPPORTED_LANGUAGES,
        queued=store.queued_count(),
        vietnamese_engine=VIENEU_MODE if VIENEU_ENABLED else "xtts-finetune",
    )


@app.get("/languages", response_model=LanguagesOut)
def languages() -> LanguagesOut:
    return LanguagesOut(languages=engine.languages if engine.loaded else SUPPORTED_LANGUAGES, default=DEFAULT_LANGUAGE)


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


@app.delete("/voices/{voice_id}")
def delete_voice(voice_id: str) -> dict:
    v = _voice_or_404(voice_id)
    Path(v["ref_path"]).unlink(missing_ok=True)
    # Drop the cached voice latents too (each model caches separately).
    if engine.loaded:
        voice_dir = engine._synth().voice_dir  # noqa: SLF001
        (Path(voice_dir) / f"{voice_id}.pth").unlink(missing_ok=True)
    if engine.vietnamese_loaded:
        (VI_VOICE_DIR / f"{voice_id}.pth").unlink(missing_ok=True)
    store.delete_voice(voice_id)
    return {"deleted": voice_id}


# --- Generation --------------------------------------------------------


@app.post("/generate", response_model=JobOut)
def generate(req: GenerateRequest) -> JobOut:
    _voice_or_404(req.voice_id)
    text = req.text.strip()
    if not text:
        raise HTTPException(422, "Văn bản trống")
    if len(text) > MAX_TEXT_CHARS:
        raise HTTPException(413, f"Văn bản vượt quá {MAX_TEXT_CHARS} ký tự")
    langs = engine.languages if engine.loaded else SUPPORTED_LANGUAGES
    if req.language not in langs:
        raise HTTPException(400, f"Ngôn ngữ '{req.language}' không được hỗ trợ. thử: {', '.join(langs)}")

    job = store.create_job(
        text=text,
        voice_id=req.voice_id,
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
