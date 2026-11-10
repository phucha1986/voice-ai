export const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

export type JobStatus = "queued" | "running" | "done" | "failed" | "cancelled";

export interface Voice {
  id: string;
  name: string;
  created_at: string;
  duration_seconds: number;
  sample_rate: number;
  source_filename: string | null;
}

export interface Job {
  id: string;
  text: string;
  voice_id: string;
  language: string;
  status: JobStatus;
  progress: number;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  duration_seconds: number | null;
  error: string | null;
}

export interface Health {
  ok: boolean;
  model: string;
  device: string;
  backend: string;
  device_description: string;
  gpu: boolean;
  ffmpeg: boolean;
  model_loaded: boolean;
  languages: string[];
  queued: number;
}

export interface GenerateRequest {
  text: string;
  voice_id: string;
  language: string;
  speed: number;
  temperature: number;
  seed: number | null;
}

async function http<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${url}`, init);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail ?? body);
    } catch {
      /* non-JSON error body */
    }
    throw new Error(`${res.status}: ${detail}`);
  }
  return res.json() as Promise<T>;
}

export function uploadVoice(file: File, name: string | null): Promise<Voice> {
  const form = new FormData();
  form.append("file", file);
  if (name) form.append("name", name);
  return http<Voice>("/voices", { method: "POST", body: form });
}

export function listVoices(): Promise<Voice[]> {
  return http<Voice[]>("/voices");
}

export function deleteVoice(id: string): Promise<void> {
  return http<void>(`/voices/${id}`, { method: "DELETE" });
}

export function generate(req: GenerateRequest): Promise<Job> {
  return http<Job>("/generate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
}

export function listJobs(): Promise<Job[]> {
  return http<Job[]>("/jobs");
}

export function cancelJob(id: string): Promise<Job> {
  return http<Job>(`/jobs/${id}/cancel`, { method: "POST" });
}

export function deleteJob(id: string): Promise<void> {
  return http<void>(`/jobs/${id}`, { method: "DELETE" });
}

export function getHealth(): Promise<Health> {
  return http<Health>("/health");
}

export function voiceAudioUrl(id: string): string {
  return `${API_BASE}/voices/${id}/audio`;
}

export function jobAudioUrl(id: string): string {
  return `${API_BASE}/jobs/${id}/audio`;
}

export function jobDownloadUrl(id: string): string {
  return `${API_BASE}/jobs/${id}/download`;
}

export function jobStreamUrl(id: string): string {
  return `${API_BASE}/jobs/${id}/stream`;
}
