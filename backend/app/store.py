"""Persistent JSON store for voices and synthesis jobs.

Keeps state on disk (``data/voices.json``, ``data/jobs.json``) so the server can
restart without losing uploaded voices or generated audio metadata.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import AUDIO_DIR, DATA_DIR, VOICES_DIR


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    os.replace(tmp, path)


class Store:
    """Thread-safe JSON index for voices and jobs."""

    def __init__(self, data_dir: Path = DATA_DIR) -> None:
        self.data_dir = data_dir
        self.voices_path = data_dir / "voices.json"
        self.jobs_path = data_dir / "jobs.json"
        self._lock = threading.Lock()
        self._voices: dict[str, dict[str, Any]] = self._load(self.voices_path)
        self._jobs: dict[str, dict[str, Any]] = self._load(self.jobs_path)
        VOICES_DIR.mkdir(parents=True, exist_ok=True)
        AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _load(path: Path) -> dict[str, dict[str, Any]]:
        if not path.exists():
            return {}
        try:
            with open(path, encoding="utf-8") as fh:
                return json.load(fh)
        except (json.JSONDecodeError, OSError):
            return {}

    def _save_voices(self) -> None:
        _atomic_write(self.voices_path, self._voices)

    def _save_jobs(self) -> None:
        _atomic_write(self.jobs_path, self._jobs)

    # --- Voices --------------------------------------------------------

    def create_voice(self, *, name: str, ref_path: str, meta: dict) -> dict:
        voice_id = uuid.uuid4().hex[:12]
        record = {
            "id": voice_id,
            "name": name,
            "ref_path": ref_path,
            "created_at": _now(),
            **meta,
        }
        with self._lock:
            self._voices[voice_id] = record
            self._save_voices()
        return record

    def list_voices(self) -> list[dict]:
        with self._lock:
            return sorted(self._voices.values(), key=lambda v: v["created_at"], reverse=True)

    def get_voice(self, voice_id: str) -> dict | None:
        with self._lock:
            return self._voices.get(voice_id)

    def delete_voice(self, voice_id: str) -> bool:
        with self._lock:
            record = self._voices.pop(voice_id, None)
            if record is None:
                return False
            self._save_voices()
        return True

    # --- Jobs ----------------------------------------------------------

    def create_job(
        self,
        *,
        text: str,
        voice_id: str,
        language: str,
        speed: float = 1.0,
        temperature: float = 0.75,
        seed: int | None = None,
        voice_preset: str | None = None,
    ) -> dict:
        job_id = uuid.uuid4().hex[:12]
        record = {
            "id": job_id,
            "text": text,
            "voice_id": voice_id,
            "voice_preset": voice_preset,
            "language": language,
            "speed": speed,
            "temperature": temperature,
            "seed": seed,
            "status": "queued",
            "progress": 0.0,
            "created_at": _now(),
            "started_at": None,
            "finished_at": None,
            "audio_path": None,
            "duration_seconds": None,
            "error": None,
        }
        with self._lock:
            self._jobs[job_id] = record
            self._save_jobs()
        return record

    def update_job(self, job_id: str, **changes: Any) -> dict:
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                raise KeyError(job_id)
            record.update(changes)
            self._save_jobs()
            return dict(record)

    def get_job(self, job_id: str) -> dict | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list_jobs(self, limit: int = 50) -> list[dict]:
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda j: j["created_at"], reverse=True)
        return jobs[:limit]

    def delete_job(self, job_id: str) -> bool:
        with self._lock:
            record = self._jobs.pop(job_id, None)
            if record is None:
                return False
            self._save_jobs()
        return True

    def queued_count(self) -> int:
        with self._lock:
            return sum(1 for j in self._jobs.values() if j["status"] in ("queued", "running"))


store = Store()
