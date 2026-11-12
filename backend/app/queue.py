"""Background generation queue.

Jobs are enqueued by the API and drained by a small pool of worker threads.
XTTS is not reentrant, so the engine lock serialises actual inference; extra
workers mainly keep the queue draining while a job is waiting on the lock.
"""

from __future__ import annotations

import logging
import queue
import threading
import time

from .config import AUDIO_DIR, QUEUE_WORKERS
from .engine import engine
from .store import store

logger = logging.getLogger("voice-ai.queue")


class Queue:
    def __init__(self, workers: int = QUEUE_WORKERS) -> None:
        self._q: queue.Queue[str] = queue.Queue()
        self._workers: list[threading.Thread] = []
        self._stop = threading.Event()
        self._workers_n = workers

    def start(self) -> None:
        if self._workers:
            return
        self._stop.clear()
        for i in range(self._workers_n):
            t = threading.Thread(target=self._run, name=f"tts-worker-{i}", daemon=True)
            t.start()
            self._workers.append(t)
        logger.info("Queue started with %d worker(s)", self._workers_n)

    def stop(self) -> None:
        self._stop.set()
        self._q.put("")
        for t in self._workers:
            t.join(timeout=2)
        self._workers = []

    def enqueue(self, job_id: str) -> None:
        self._q.put(job_id)

    def requeue_pending(self) -> int:
        """Re-enqueue jobs left in queued/running state after a restart."""
        n = 0
        for job in store.list_jobs(limit=1000):
            if job["status"] in ("queued", "running"):
                store.update_job(job["id"], status="queued", progress=0.0, error="Requeued after server restart")
                self.enqueue(job["id"])
                n += 1
        return n

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                job_id = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            if not job_id:
                continue
            try:
                self._process(job_id)
            except Exception as exc:  # noqa: BLE001 - any failure must mark the job failed
                logger.exception("Job %s failed", job_id)
                try:
                    store.update_job(job_id, status="failed", error=str(exc)[:500], finished_at=_now())
                except KeyError:
                    pass  # job was deleted while it was running

    def _process(self, job_id: str) -> None:
        job = store.get_job(job_id)
        if job is None or job["status"] == "cancelled":
            return
        preset = job.get("voice_preset")
        if preset:
            # Built-in VieNeu voice: no uploaded reference to look up.
            ref_path = ""
        else:
            voice = store.get_voice(job["voice_id"])
            if voice is None:
                store.update_job(job_id, status="failed", error="Không tìm thấy giọng nói", finished_at=_now())
                return
            ref_path = voice["ref_path"]

        store.update_job(job_id, status="running", started_at=_now(), progress=0.0, error=None)
        out_path = AUDIO_DIR / f"{job_id}.wav"
        started = time.time()

        def on_progress(frac: float) -> None:
            current = store.get_job(job_id)
            if current is None or current["status"] == "cancelled":
                raise RuntimeError("Job cancelled")
            store.update_job(job_id, progress=round(min(frac, 0.99), 3))

        try:
            wav = engine.synthesize(
                job["text"],
                ref_wav=ref_path,
                voice_id=job["voice_id"],
                voice_preset=preset,
                language=job["language"],
                speed=job.get("speed", 1.0),
                temperature=job.get("temperature", 0.75),
                seed=job.get("seed"),
                progress_cb=on_progress,
            )
        except RuntimeError as exc:
            if "cancelled" in str(exc):
                store.update_job(job_id, status="cancelled", finished_at=_now(), progress=0.0)
                return
            raise

        meta = engine.save_wav(wav, str(out_path))
        store.update_job(
            job_id,
            status="done",
            progress=1.0,
            finished_at=_now(),
            audio_path=str(out_path),
            duration_seconds=meta["duration_seconds"],
            sample_rate=meta["sample_rate"],
            synth_seconds=round(time.time() - started, 2),
        )
        logger.info("Job %s done (%.2fs audio in %.2fs)", job_id, meta["duration_seconds"], time.time() - started)


def _now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


generation_queue = Queue()
