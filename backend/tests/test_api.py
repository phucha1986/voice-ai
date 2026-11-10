"""API tests with the synthesis engine mocked (no model download)."""

from __future__ import annotations

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from app.main import app
from app.queue import generation_queue
from app.store import store


class FakeEngine:
    """Stand-in for Engine: returns a deterministic tone and writes a WAV."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    @property
    def loaded(self) -> bool:
        return True

    @property
    def device_info(self):
        from app.device import DeviceInfo

        return DeviceInfo("cpu", "cpu", "CPU (test)")

    @property
    def languages(self) -> list[str]:
        return ["en", "vi"]

    def synthesize(self, text: str, *, ref_wav: str, voice_id: str, language: str = "en", **kw) -> np.ndarray:
        self.calls.append({"text": text, "ref_wav": ref_wav, "voice_id": voice_id, "language": language, **kw})
        sr = 24000
        n = int(sr * min(2.0, max(0.5, len(text) / 40)))
        t = np.arange(n) / sr
        return (0.3 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)

    def save_wav(self, wav: np.ndarray, path: str) -> dict:
        sf.write(path, wav, 24000, subtype="PCM_16")
        return {"path": path, "sample_rate": 24000, "duration_seconds": round(len(wav) / 24000, 2)}


def _wait_for_job(client: TestClient, job_id: str, timeout: float = 10.0) -> dict:
    import time

    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/jobs/{job_id}").json()
        if job["status"] in ("done", "failed", "cancelled"):
            return job
        time.sleep(0.15)
    raise AssertionError(f"job {job_id} never finished")


def test_health(client) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["device"] == "cpu"
    assert body["ffmpeg"] is True


def test_upload_voice_and_reference(client) -> None:
    wav = _make_wav(seconds=4.0)
    r = client.post("/voices", files={"file": ("ref.wav", wav, "audio/wav")}, data={"name": "Test Voice"})
    assert r.status_code == 200
    voice = r.json()
    assert voice["name"] == "Test Voice"
    assert voice["duration_seconds"] >= 2.0

    voices = client.get("/voices").json()
    assert any(v["id"] == voice["id"] for v in voices)

    audio = client.get(f"/voices/{voice['id']}/audio")
    assert audio.status_code == 200
    assert audio.headers["content-type"].startswith("audio")


def test_upload_rejects_bad_type_and_short_clip(client) -> None:
    r = client.post("/voices", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 400

    short = _make_wav(seconds=1.0)
    r = client.post("/voices", files={"file": ("short.wav", short, "audio/wav")})
    assert r.status_code == 422
    assert "quá ngắn" in r.json()["detail"]


def test_generate_flow_with_queue(client, monkeypatch) -> None:
    fake = FakeEngine()
    monkeypatch.setattr("app.queue.engine", fake)

    voice = client.post(
        "/voices", files={"file": ("ref.wav", _make_wav(seconds=3.0), "audio/wav")}, data={"name": "V"}
    ).json()

    r = client.post("/generate", json={"text": "Hello from the queue.", "voice_id": voice["id"], "language": "en"})
    assert r.status_code == 200
    job = r.json()
    assert job["status"] == "queued"

    done = _wait_for_job(client, job["id"])
    assert done["status"] == "done"
    assert done["duration_seconds"] > 0
    assert fake.calls and fake.calls[0]["text"] == "Hello from the queue."

    audio = client.get(f"/jobs/{job['id']}/audio")
    assert audio.status_code == 200
    import io

    data, sr = sf.read(io.BytesIO(audio.content))
    assert sr == 24000 and len(data) > 0

    dl = client.get(f"/jobs/{job['id']}/download")
    assert dl.status_code == 200
    assert "attachment" in dl.headers.get("content-disposition", "")


def test_generate_validation(client) -> None:
    voice = client.post(
        "/voices", files={"file": ("ref.wav", _make_wav(seconds=3.0), "audio/wav")}, data={"name": "V2"}
    ).json()

    assert client.post("/generate", json={"text": "", "voice_id": voice["id"]}).status_code == 422
    assert client.post("/generate", json={"text": "hi", "voice_id": "missing"}).status_code == 404
    assert client.post("/generate", json={"text": "hi", "voice_id": voice["id"], "language": "xx"}).status_code == 400


def test_cancel_and_delete(client, monkeypatch) -> None:
    fake = FakeEngine()
    monkeypatch.setattr("app.queue.engine", fake)

    voice = client.post(
        "/voices", files={"file": ("ref.wav", _make_wav(seconds=3.0), "audio/wav")}, data={"name": "V3"}
    ).json()
    job = client.post("/generate", json={"text": "Cancel me.", "voice_id": voice["id"]}).json()

    # Cancel before the worker picks it up (queue may already be running; either outcome is terminal).
    r = client.post(f"/jobs/{job['id']}/cancel")
    assert r.status_code == 200
    assert r.json()["status"] in ("cancelled", "done", "running")

    d = client.delete(f"/jobs/{job['id']}")
    assert d.status_code == 200
    assert client.get(f"/jobs/{job['id']}").status_code == 404


def test_delete_voice_removes_files(client) -> None:
    voice = client.post(
        "/voices", files={"file": ("ref.wav", _make_wav(seconds=3.0), "audio/wav")}, data={"name": "V4"}
    ).json()
    r = client.delete(f"/voices/{voice['id']}")
    assert r.status_code == 200
    assert client.get(f"/voices/{voice['id']}").status_code == 404
    assert store.get_voice(voice["id"]) is None


def _make_wav(seconds: float) -> bytes:
    import io

    sr = 24000
    t = np.arange(int(seconds * sr)) / sr
    audio = (0.4 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()
