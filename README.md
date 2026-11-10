# Voice AI — XTTS v2 voice cloning

A local voice-cloning app: upload a reference audio clip of a speaker, type text, and get speech in that voice.

- **Frontend**: Next.js 15 (React 19) — upload, voice list with reference player, text + language/speed/temperature controls, live job queue (SSE progress), audio player, WAV download.
- **Backend**: FastAPI (Python 3.11) — upload validation, audio preprocessing (FFmpeg decode → silence trim → peak normalize → 2–30 s fit), voice latent caching, generation queue with worker threads, WAV storage, SSE status stream, download.
- **Model**: `tts_models/multilingual/multi-dataset/xtts_v2` (Coqui XTTS v2, ~1.77 GB, cached on first load) for 17 languages, plus the Vietnamese finetune `Nhat1106/xtts-vietnamese` (~5.7 GB, cached on first `vi` request). Output: 24 kHz mono WAV.
- **GPU**: CUDA / ROCm via PyTorch when available; optional DirectML; CPU fallback.

## Layout

```
voice-ai/
  backend/            FastAPI + XTTS engine + queue
    app/              config, device, audio, store, models, engine, queue, main
    tests/            pytest suite (28 tests)
    requirements.txt
    data/             voices/, voices-vi/, audio/, tmp/, voices.json, jobs.json (created at runtime)
  frontend/           Next.js app
    src/app/          layout, page, globals.css
    src/lib/api.ts    typed API client
```

## Prerequisites

- **Python 3.11** (3.13 is too new for Coqui TTS; 3.11 is verified).
- **FFmpeg**:
  - On Windows, `torchcodec` (used by coqui-tts on torch ≥ 2.9) needs a **shared** FFmpeg build with DLLs, e.g. `winget install BtbN.FFmpeg.GPL.Shared`. The backend auto-discovers it under `%LOCALAPPDATA%\Microsoft\WinGet\Packages\BtbN.FFmpeg*` and calls `os.add_dll_directory()`.
  - A static build (e.g. `Gyan.FFmpeg`) on PATH is enough for upload preprocessing (subprocess), but generation will fail without shared DLLs for torchcodec.
- **Node.js 18+** (verified on 26).
- ~2 GB free disk for the XTTS v2 cache (`%LOCALAPPDATA%\tts\`), plus ~5.7 GB for the Vietnamese finetune cache (`%USERPROFILE%\.cache\huggingface\`) if you use `vi`.

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
# Backend (terminal 1) — first run downloads the XTTS model (~1.77 GB)
cd backend
.venv\Scripts\activate
uvicorn app.main:app --host 127.0.0.1 --port 8000

# Frontend (terminal 2)
cd frontend
npm run dev        # http://localhost:3000
```

Open http://localhost:3000: upload a 2–30 s clip of one speaker (WAV/MP3/OGG/FLAC/M4A/AAC/OPUS, ≤ 25 MB), pick it, type text, click **Tạo giọng nói**. Progress streams live; when the job is done, play or download the WAV. The UI is in Vietnamese and the default language is `vi`.

For Vietnamese output, upload a **Vietnamese** reference clip: cloning transfers the speaker's timbre, so a Vietnamese sample yields a Vietnamese-sounding voice. The bundled voice `Giọng nói tiếng Việt` (`data/voices/`) is a 12 s Vietnamese clip; the earlier English `ref_voice.wav` sample has been removed.

An optional **Seed** makes output reproducible: the same seed with the same text, voice and settings returns byte-identical audio. Leave it empty for varied output.

## Pauses between sentences

XTTS emits 200–380 ms of silence at the end (and sometimes the start) of each sentence chunk, so concatenating chunks produced an audible pause mid-speech. `collapse_long_gaps()` in `backend/app/audio.py` scans the output in 20 ms windows, classifies a window as silence when its RMS is below `max(0.02, 5 % of peak)`, and shortens any silence run longer than `MAX_GAP_MS` (default 150 ms, override with `VOICE_AI_MAX_GAP_MS`) down to that length, keeping the middle of the run so no speech is lost. Silence shorter than the cap is a natural word boundary and is left untouched, so prosody stays natural. `Engine.synthesize()` applies it after concatenation and `Engine.synthesize_stream()` applies the streaming variant, which holds back the buffer tail so a gap spanning two chunks is capped once.

Measured on a two-sentence Vietnamese job (seed 1234): raw concatenation had gaps of 380/280/240 ms; the collapsed output has a 160 ms maximum and keeps the same speech samples. All stored jobs were regenerated with the fix.

## Languages

The app serves 18 languages via `GET /languages` and rejects anything else with `400`:

`vi, en, es, fr, de, it, pt, pl, tr, ru, nl, cs, ar, zh-cn, ja, hu, ko, hi`

The 17 non-Vietnamese languages come from stock XTTS v2, whose language set is fixed. **Vietnamese (`vi`) is not part of stock XTTS v2** — it has no Vietnamese phonemizer or speaker set, so `language="vi"` would raise `KeyError: 'vi'` inside Coqui's sentence segmenter.

Vietnamese speech is served by a second model: the community XTTS v2 finetune **`Nhat1106/xtts-vietnamese`** (~5.7 GB, downloaded from Hugging Face on first use of `vi`). `Engine.load_vietnamese()` builds a separate `Synthesizer` for it, and `Engine._synth_for(language)` routes `vi` requests to that model while everything else stays on stock XTTS v2. Consequences:

- Vietnamese voices are cloned into their own cache (`data/voices-vi/`) so the two models never share latents. `DELETE /voices/{id}` clears both caches.
- The first `vi` job pays the download + load cost (~5.7 GB, ~20 s to load on CPU), then latents are cached.
- Set `VOICE_AI_VI_MODEL=` (empty) to disable Vietnamese entirely — `vi` is then removed from `/languages` and rejected with `400`.
- A Vietnamese reference clip still clones *timbre* cross-lingually for the other 17 languages, as before.

## Reference audio

Cloning copies the speaker's **timbre**, not the language, so the language of the *reference* clip decides what accent the cloned voice carries. To get Vietnamese speech, upload a Vietnamese sample. The bundled voice ships with a 12 s clip taken from the Hugging Face dataset [`Speech-data/Vietnamese-Speech-Dataset`](https://huggingface.co/datasets/Speech-data/Vietnamese-Speech-Dataset) (`vietnamese.mp3`, trimmed to the first 12 s, then decoded to mono 24 kHz, silence-stripped and normalized by `preprocess_reference()`). That dataset is licensed **CC BY-NC-ND 4.0** (non-commercial, no derivatives), so replace it with your own recording before any commercial use. Verified with `faster-whisper`: the reference transcribes as `vi` (probability 1.00), and jobs cloned from it transcribe as `vi` (0.98–0.99).

## GPU

The backend picks the best device at startup and reports it in `GET /health`:

- CUDA: `torch.cuda.is_available()`
- ROCm: PyTorch ROCm build (`torch.version.hip`)
- DirectML (AMD on Windows): install `torch-directml` into the venv; used if present and no CUDA/ROCm
- CPU fallback otherwise

Force a device with `VOICE_AI_DEVICE=cpu` (force CPU even when a GPU exists) or `VOICE_AI_DEVICE=cuda` (require CUDA/ROCm, error if absent). DirectML is auto-detected when installed. On CPU, expect roughly 1–2 minutes for a short sentence.

## Environment variables (backend)

| Variable | Default | Purpose |
|---|---|---|
| `VOICE_AI_DATA_DIR` | `backend/data` | Storage root (voices, audio, tmp, JSON stores) |
| `VOICE_AI_MODEL` | `tts_models/multilingual/multi-dataset/xtts_v2` | Coqui model registry name |
| `VOICE_AI_VI_MODEL` | `Nhat1106/xtts-vietnamese` | Hugging Face repo serving `vi`; set empty to disable Vietnamese |
| `VOICE_AI_VI_CONFIG` | `config2.json` | Inference config file inside the Vietnamese snapshot |
| `VOICE_AI_LANGUAGE` | `vi` | Default language |
| `VOICE_AI_DEVICE` | auto | Force `cpu`, or require `cuda` (DirectML auto-detected) |
| `VOICE_AI_QUEUE_WORKERS` | `1` | Parallel synthesis workers |
| `VOICE_AI_MAX_UPLOAD_BYTES` | `25 MB` | Upload size limit |
| `VOICE_AI_MAX_REF_SECONDS` | `30` | Reference clip length cap |
| `VOICE_AI_MAX_TEXT_CHARS` | `4000` | Text length cap |
| `COQUI_TOS_AGREED` | set to `1` by the app | Skip the CPML license interactive prompt |
| `HF_TOKEN` | unset | Optional Hugging Face token for model download |

Frontend: `NEXT_PUBLIC_API_BASE` (default `http://localhost:8000`).

## API

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Device, backend, GPU flag, ffmpeg, model-loaded, queue depth |
| `GET` | `/languages` | Supported languages (XTTS v2's 17 plus `vi` — see "Languages") |
| `POST` | `/voices` | Multipart upload (`file`, optional `name`) → preprocessed voice |
| `GET` | `/voices` / `/voices/{id}` | List / get voices |
| `GET` | `/voices/{id}/audio` | Reference WAV |
| `DELETE` | `/voices/{id}` | Remove voice |
| `POST` | `/generate` | JSON `{text, voice_id, language, speed, temperature, seed}` → queued job |
| `GET` | `/jobs` / `/jobs/{id}` | List / get jobs |
| `GET` | `/jobs/{id}/stream` | SSE status/progress until terminal state |
| `GET` | `/jobs/{id}/audio` | Generated WAV (409 until done) |
| `GET` | `/jobs/{id}/download` | Same, as `voice-ai-<job>.wav` attachment |
| `POST` | `/jobs/{id}/cancel` | Cancel queued/running job |
| `DELETE` | `/jobs/{id}` | Delete job |

## Tests

```bash
cd backend
.venv\Scripts\activate
pytest            # 28 tests: audio preprocessing, gap collapsing, API (fake engine), engine settings/seed, language list + Vietnamese routing
```

## License note

XTTS v2 weights are under the Coqui Public Model License (CPML): free for non-commercial use; commercial use requires a license. The app sets `COQUI_TOS_AGREED=1` so the prompt is skipped — accept the CPML terms before using the model.
