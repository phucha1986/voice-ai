"""Pydantic schemas shared by the API and the frontend."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

JobStatus = Literal["queued", "running", "done", "failed", "cancelled"]


class VoiceOut(BaseModel):
    id: str
    name: str
    created_at: str
    duration_seconds: float
    sample_rate: int
    source_filename: str | None = None


class JobOut(BaseModel):
    id: str
    text: str
    voice_id: str
    language: str
    status: JobStatus
    progress: float = Field(ge=0, le=1)
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    duration_seconds: float | None = None
    error: str | None = None


class GenerateRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4000)
    voice_id: str
    language: str = "en"
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    temperature: float = Field(default=0.75, ge=0.0, le=1.0)
    seed: int | None = None


class HealthOut(BaseModel):
    ok: bool
    model: str
    device: str
    backend: str
    device_description: str
    gpu: bool
    ffmpeg: bool
    model_loaded: bool
    languages: list[str]
    queued: int
    vietnamese_engine: str = "xtts-finetune"
    vieneu_ready: bool = False


class LanguagesOut(BaseModel):
    languages: list[str]
    default: str
