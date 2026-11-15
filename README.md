# Voice AI — VieNeu-TTS voice cloning

A local voice-cloning app: upload a reference audio clip of a speaker, type text, and get speech in that voice. **VieNeu-TTS is the only engine** — Vietnamese voice cloning on a torch-free ONNX CPU runtime.

- **Frontend**: Next.js 15 (React 19) — upload, voice list with reference player, built-in preset voices, text + speed/temperature controls, live job queue (SSE progress), audio player, WAV download.
- **Backend**: FastAPI (Python) — upload validation, audio decoding (FFmpeg decode → 2–30 s fit), generation queue with worker threads, WAV storage, SSE status stream, download. Generated audio is saved raw, exactly as VieNeu emits it.
- **Model**: [VieNeu-TTS](https://github.com/pitvnga1/vieneu-tts) (`vieneu` package, `v3turbo` by default). Output: 24 kHz mono WAV (v3 Turbo renders at 48 kHz and is downsampled with `soxr`).
- **Device**: CPU only (ONNX Runtime). VieNeu is torch-free; there is no GPU path.

## Layout

```
voice-ai/
  backend/            FastAPI + VieNeu engine + queue
    app/              config, device, audio, store, models, engine, queue, main
    tests/            pytest suite
    requirements.txt
    data/             voices/, audio/, tmp/, previews/, voices.json, jobs.json (created at runtime)
  frontend/           Next.js app
    src/app/          layout, page, globals.css
    src/lib/api.ts    typed API client
```

## Prerequisites

- **Python 3.11+** (the venv used here is Python 3.x with `vieneu==3.8.4`).
- **FFmpeg** on PATH for mp3/m4a/ogg upload decoding. On Windows, `winget install Gyan.FFmpeg` (a static build is enough — the backend calls it as a subprocess). The backend also probes a known winget install location as a fallback.
- **Node.js 18+**.
- VieNeu model weights are cached on first load (~a few hundred MB), not gigabytes.

## Setup

```bash
# Backend
cd backend
python -m venv .venv
.venv\Scripts\activate            # Windows; source .venv/bin/activate on Linux/macOS
pip install -r requirements.txt

# Frontend
cd ../frontend
npm install
```

## Run

```bash
# Backend (terminal 1) — first run downloads VieNeu (~12 s)
cd backend
.venv\Scripts\activate
uvicorn app.main:app --host 127.0.0.1 --port 8000

# Frontend (terminal 2)
cd frontend
npm run dev        # http://localhost:3000
```

Open http://localhost:3000: upload a 2–30 s clip of one speaker (WAV/MP3/OGG/FLAC/M4A/AAC/OPUS, ≤ 25 MB), pick a voice (uploaded or a built-in VieNeu preset), type Vietnamese text, click **Tạo giọng nói**. Progress streams live; when the job is done, play or download the WAV. The UI is in Vietnamese and the only language is `vi`.

For Vietnamese output, upload a **Vietnamese** reference clip: cloning transfers the speaker's timbre, so a Vietnamese sample yields a Vietnamese-sounding voice.

An optional **Seed** makes output reproducible on `v3nano` (which accepts `seed`); on `v3turbo` it is ignored.

## VieNeu engine

VieNeu-TTS speaks Vietnamese with voice cloning on a torch-free ONNX CPU runtime. Its phonemizer is phoneme-level (`sea-g2p`), so Vietnamese compound words (`nhanh nhẹn`, `hoạt động`) are read as one word instead of one syllable per token — no text preprocessing is needed.

Two modes, both ONNX on CPU (the app pins `device="cpu"`, `backend="onnx"`):

| Mode | Sample rate | Notes |
|---|---|---|
| `v3turbo` (default) | 48 kHz | Downsampled to 24 kHz with `soxr` (blocking and streaming). `fp32` is the default: `int8` is ~1.45x faster but drops tone marks on compound words (`nhanh nhẹn` → `nhanh nhẹ`) and needs a VNNI-capable CPU. `speed`/`seed` are ignored — `temperature` is mapped (clamped 0.3–0.9). |
| `v3nano` | 24 kHz | No resampling needed. Smaller/faster per step but slower overall on this CPU; supports `speed` and `seed`. |

Measured on an i5-11400F (6 cores), cloning the bundled Vietnamese reference, blocking `infer`, RTF = synthesis seconds / audio seconds:

| Setting | RTF | Notes |
|---|---|---|
| `v3turbo` `fp32`, `THREADS=0` | 0.60–0.63 | Default and fastest measured setting. |
| `v3turbo` `fp32`, `THREADS=3` | 0.70–0.81 | ~25% slower; ONNX already caps intra-op threads at half the cores. |
| `v3turbo` `fp32`, `THREADS=12` | 0.61–0.63 | No gain over `0`. |
| `v3turbo` `int8`, `THREADS=0` | 0.38–0.43 | ~1.45x faster, but Whisper reads `nhanh nhẹn` as `nhanh nhẹ` (tone mark lost), so it is opt-in. |
| `v3nano` | 0.96–2.78 | Slower than `v3turbo` except for long text. |

Startup is warm, not lazy: `VIENEU_PRELOAD` (default on) loads VieNeu and enrols the newest voice reference on a background thread, so the first job does not pay the ~12 s model load plus the ~13 s first-enrolment (denoise + speaker encoder). `POST /voices` enrols the new clip the same way. `GET /health` reports `vieneu_ready`. Reference enrolment is memoised by clip content inside VieNeu, so repeat jobs with the same voice pay no enrolment cost.

## Raw output

There is **no post-processing** of generated audio: `Engine.synthesize()` returns exactly what VieNeu produced (resampled to 24 kHz when the mode renders at 48 kHz), and `Engine.save_wav()` writes it straight to disk. Reference clips are likewise only decoded and length-fitted — no silence trimming, no peak normalizing, no noise gate.

The only cleanup is inside VieNeu itself: `denoise=True` on `infer()` and `encode_reference()`, which denoises the reference at enrolment.

Consequence: VieNeu's own chunk-boundary artifacts are preserved — long silence between chunks and the reference's room tone re-emitted at boundaries, which can sound like a pause or a murmur repeating between phrases. That is the engine's raw behaviour, deliberately left untouched.

## Reference audio

Cloning copies the speaker's **timbre**, not the language, so the language of the *reference* clip decides what accent the cloned voice carries. To get Vietnamese speech, upload a Vietnamese sample.

## Environment variables (backend)

| Variable | Default | Purpose |
|---|---|---|
| `VOICE_AI_DATA_DIR` | `backend/data` | Storage root (voices, audio, tmp, previews, JSON stores) |
| `VOICE_AI_LANGUAGE` | `vi` | Default (and only) language |
| `VOICE_AI_DEVICE` | `cpu` | VieNeu is CPU-only; only `cpu` is accepted |
| `VOICE_AI_QUEUE_WORKERS` | `1` | Parallel synthesis workers |
| `VOICE_AI_MAX_UPLOAD_BYTES` | `25 MB` | Upload size limit |
| `VOICE_AI_MAX_REF_SECONDS` | `30` | Reference clip length cap |
| `VOICE_AI_MAX_TEXT_CHARS` | `4000` | Text length cap |
| `VOICE_AI_VIENEU_MODE` | `v3turbo` | VieNeu mode: `v3turbo` (48 kHz, resampled) or `v3nano` (24 kHz) |
| `VOICE_AI_VIENEU_PRECISION` | `fp32` | VieNeu ONNX precision (`fp32` or `int8`, `v3turbo` only) |
| `VOICE_AI_VIENEU_THREADS` | `0` | ONNX inference threads (`0` = auto) |
| `VOICE_AI_VIENEU_PRELOAD` | `1` | Set `0` to skip the startup warm-up |

Frontend: `NEXT_PUBLIC_API_BASE` (default `http://localhost:8000`).

## API

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Device, ffmpeg, model-loaded, `vieneu_ready`, queue depth |
| `GET` | `/languages` | Supported languages (`vi` only) |
| `GET` | `/presets` | Built-in VieNeu voices |
| `POST` | `/voices` | Multipart upload (`file`, optional `name`) → preprocessed voice |
| `GET` | `/voices` / `/voices/{id}` | List / get voices |
| `GET` | `/voices/{id}/audio` | Reference WAV |
| `GET` | `/preview` | Short audition sample for a voice or preset (cached) |
| `DELETE` | `/voices/{id}` | Remove voice |
| `POST` | `/generate` | JSON `{text, voice_id, voice_preset, language, speed, temperature, seed}` → queued job |
| `GET` | `/jobs` / `/jobs/{id}` | List / get jobs |
| `GET` | `/jobs/{id}/stream` | SSE status/progress until terminal state |
| `GET` | `/jobs/{id}/audio` | Generated WAV (409 until done) |
| `GET` | `/jobs/{id}/download` | Same, as `voice-ai-<job>.wav` attachment |
| `POST` | `/jobs/{id}/cancel` | Cancel queued/running job |
| `DELETE` | `/jobs/{id}` | Delete job |

`/generate` rejects any `language` other than `vi` with `400`.

## Tests

```bash
cd backend
.venv\Scripts\activate
pytest            # audio decoding/duration fit, VieNeu routing/resampling/raw output, API (fake engine)
```

## License note

VieNeu-TTS is Apache-2.0.
