"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  cancelJob,
  deleteJob,
  deleteVoice,
  generate,
  getHealth,
  jobAudioUrl,
  jobDownloadUrl,
  jobStreamUrl,
  listJobs,
  listPresets,
  listVoices,
  previewUrl,
  uploadVoice,
  voiceAudioUrl,
  type Health,
  type Job,
  type PresetVoice,
  type Voice,
} from "@/lib/api";

const STATUS_LABELS: Record<string, string> = {
  queued: "trong hàng đợi",
  running: "đang tạo",
  done: "hoàn tất",
  failed: "lỗi",
  cancelled: "đã hủy",
};

// OmniVoice Vietnamese speaks Vietnamese only; job metadata is shown in Vietnamese.
const LANGUAGE_LABEL = "Tiếng Việt";

// Backend preset metadata uses English gender codes.
const GENDER_LABELS: Record<string, string> = {
  female: "Nữ",
  male: "Nam",
};

export default function Home() {
  const [health, setHealth] = useState<Health | null>(null);
  const [voices, setVoices] = useState<Voice[]>([]);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [notice, setNotice] = useState<string | null>(null);

  const [file, setFile] = useState<File | null>(null);
  const [voiceName, setVoiceName] = useState("");
  const [uploading, setUploading] = useState(false);

  const [text, setText] = useState(
    "Xin chào! Đây là giọng nói được tạo bằng OmniVoice trên máy của tôi.",
  );
  const [voiceId, setVoiceId] = useState("");
  const [presets, setPresets] = useState<PresetVoice[]>([]);
  const [presetsLoading, setPresetsLoading] = useState(true);
  const [voicePreset, setVoicePreset] = useState("");
  const [speed, setSpeed] = useState(1.0);
  const [temperature, setTemperature] = useState(0.75);
  const [seed, setSeed] = useState<string>("");
  const [generating, setGenerating] = useState(false);

  const fileInputRef = useRef<HTMLInputElement | null>(null);

  // "Nghe thử": one audition clip at a time; only one voice can play.
  const previewAudioRef = useRef<HTMLAudioElement | null>(null);
  const [previewingKey, setPreviewingKey] = useState<string | null>(null);
  const [previewPlayingKey, setPreviewPlayingKey] = useState<string | null>(null);

  const stopPreview = useCallback(() => {
    previewAudioRef.current?.pause();
    previewAudioRef.current = null;
    setPreviewingKey(null);
    setPreviewPlayingKey(null);
  }, []);

  async function onPreview(key: string, voiceId: string, preset: string) {
    stopPreview();
    setPreviewingKey(key);
    const el = new Audio(previewUrl(voiceId || undefined, preset || undefined));
    previewAudioRef.current = el;
    el.addEventListener("playing", () => setPreviewPlayingKey(key));
    el.addEventListener("ended", () => {
      if (previewAudioRef.current === el) {
        previewAudioRef.current = null;
        setPreviewingKey(null);
        setPreviewPlayingKey(null);
      }
    });
    el.addEventListener("error", () => {
      if (previewAudioRef.current === el) {
        previewAudioRef.current = null;
        setPreviewingKey(null);
        setPreviewPlayingKey(null);
        setNotice("Không nghe thử được giọng này — backend có thể chưa sẵn sàng.");
      }
    });
    try {
      await el.play();
    } catch {
      /* load/play failure is reported by the error listener */
    }
  }

  const refreshVoices = useCallback(async () => {
    try {
      const vs = await listVoices();
      setVoices(vs);
      setVoiceId((cur) => {
        if (cur && vs.some((v) => v.id === cur)) return cur;
        return vs[0]?.id ?? "";
      });
    } catch (e) {
      setNotice(`Không tải được danh sách giọng nói: ${e instanceof Error ? e.message : e}`);
    }
  }, []);

  // Giọng dựng sẵn của OmniVoice — chỉ dùng cho tiếng Việt.
  const refreshPresets = useCallback(async () => {
    setPresetsLoading(true);
    try {
      const ps = await listPresets();
      setPresets(ps);
      setVoicePreset((cur) => (cur && ps.some((p) => p.name === cur) ? cur : (ps[0]?.name ?? "")));
    } catch (e) {
      setNotice(`Không tải được danh sách giọng dựng sẵn: ${e instanceof Error ? e.message : e}`);
    } finally {
      setPresetsLoading(false);
    }
  }, []);

  const refreshJobs = useCallback(async () => {
    try {
      setJobs(await listJobs());
    } catch (e) {
      setNotice(`Không tải được danh sách công việc: ${e instanceof Error ? e.message : e}`);
    }
  }, []);

  useEffect(() => {
    getHealth()
      .then(setHealth)
      .catch((e) => setNotice(`Không kết nối được backend: ${e instanceof Error ? e.message : e}`));
    refreshVoices();
    refreshPresets();
    refreshJobs();
  }, [refreshVoices, refreshPresets, refreshJobs]);

  // OmniVoice khởi động nền sau khi server bật; polling cho tới khi sẵn sàng.
  useEffect(() => {
    if (health === null || health.engine_ready) {
      return;
    }
    const timer = setInterval(() => {
      getHealth()
        .then(setHealth)
        .catch(() => {});
    }, 3000);
    return () => clearInterval(timer);
  }, [health]);

  // SSE: subscribe to progress for every active job.
  const streamsRef = useRef<Map<string, EventSource>>(new Map());
  useEffect(() => {
    const streams = streamsRef.current;
    for (const job of jobs) {
      if ((job.status === "queued" || job.status === "running") && !streams.has(job.id)) {
        const es = new EventSource(jobStreamUrl(job.id));
        es.onmessage = (ev) => {
          try {
            const patch = JSON.parse(ev.data) as {
              id?: string;
              status?: string;
              progress?: number;
              duration_seconds?: number | null;
              error?: string | null;
            };
            if (patch.status === "gone") {
              es.close();
              streams.delete(job.id);
              refreshJobs();
              return;
            }
            setJobs((prev) =>
              prev.map((j) =>
                j.id === job.id
                  ? {
                      ...j,
                      progress: patch.progress ?? j.progress,
                      duration_seconds: patch.duration_seconds ?? j.duration_seconds,
                      error: patch.error ?? j.error,
                      status: patch.status as Job["status"],
                    }
                  : j,
              ),
            );
            if (patch.status === "done" || patch.status === "failed" || patch.status === "cancelled") {
              es.close();
              streams.delete(job.id);
            }
          } catch {
            /* ignore malformed SSE frames */
          }
        };
        es.onerror = () => {
          es.close();
          streams.delete(job.id);
        };
        streams.set(job.id, es);
      }
    }
    for (const [id, es] of streams) {
      const job = jobs.find((j) => j.id === id);
      if (!job || job.status === "done" || job.status === "failed" || job.status === "cancelled") {
        es.close();
        streams.delete(id);
      }
    }
  }, [jobs, refreshJobs]);

  useEffect(() => {
    const streams = streamsRef.current;
    return () => {
      for (const es of streams.values()) es.close();
      streams.clear();
      previewAudioRef.current?.pause();
      previewAudioRef.current = null;
    };
  }, []);

  async function onUpload() {
    if (!file) {
      setNotice("Hãy chọn file âm thanh trước (WAV/MP3/OGG/FLAC, 2–30 giây giọng nói).");
      return;
    }
    setUploading(true);
    setNotice(null);
    try {
      const v = await uploadVoice(file, voiceName.trim() || null);
      setFile(null);
      setVoiceName("");
      if (fileInputRef.current) fileInputRef.current.value = "";
      await refreshVoices();
      setVoiceId(v.id);
      setVoicePreset("");
    } catch (e) {
      setNotice(`Tải lên thất bại: ${e instanceof Error ? e.message : e}`);
    } finally {
      setUploading(false);
    }
  }

  function selectPreset(name: string) {
    setVoicePreset(name);
    setVoiceId("");
  }

  function selectUploaded(id: string) {
    setVoiceId(id);
    setVoicePreset("");
  }

  async function onDeleteVoice(id: string) {
    setNotice(null);
    try {
      await deleteVoice(id);
      await refreshVoices();
    } catch (e) {
      setNotice(`Xóa thất bại: ${e instanceof Error ? e.message : e}`);
    }
  }

  async function onGenerate() {
    if (!text.trim()) {
      setNotice("Hãy nhập văn bản cần tạo giọng nói.");
      return;
    }
    if (!voicePreset && !voiceId) {
      setNotice("Hãy chọn giọng nói — giọng dựng sẵn hoặc giọng tham chiếu đã tải lên.");
      return;
    }
    setGenerating(true);
    setNotice(null);
    try {
      const job = await generate({
        text: text.trim(),
        voice_id: voicePreset ? "" : voiceId,
        voice_preset: voicePreset || null,
        language: "vi",
        speed,
        temperature,
        seed: seed.trim() ? Number(seed) : null,
      });
      setJobs((prev) => [job, ...prev]);
    } catch (e) {
      setNotice(`Tạo giọng nói thất bại: ${e instanceof Error ? e.message : e}`);
    } finally {
      setGenerating(false);
    }
  }

  async function onCancel(id: string) {
    try {
      const job = await cancelJob(id);
      setJobs((prev) => prev.map((j) => (j.id === id ? job : j)));
    } catch (e) {
      setNotice(`Hủy thất bại: ${e instanceof Error ? e.message : e}`);
    }
  }

  async function onDeleteJob(id: string) {
    try {
      await deleteJob(id);
      setJobs((prev) => prev.filter((j) => j.id !== id));
    } catch (e) {
      setNotice(`Xóa công việc thất bại: ${e instanceof Error ? e.message : e}`);
    }
  }

  return (
    <main className="container">
      <div className="header">
        <div>
          <h1>Voice AI</h1>
          <p className="tagline">Tổng hợp giọng nói tiếng Việt với OmniVoice — tải lên audio tham chiếu, nhập văn bản, nhận giọng nói.</p>
        </div>
        <div className="health" data-pw="health">
          {health === null ? (
            <span className="pill">đang kiểm tra backend…</span>
          ) : (
            <>
              <span className={health.ok ? "pill ok" : "pill bad"} data-pw="health-backend">
                backend {health.ok ? "sẵn sàng" : "không hoạt động"}
              </span>
              <span className={health.gpu ? "pill ok" : "pill"} data-pw="health-device">
                {health.device} ({health.backend})
              </span>
              <span className={health.model_loaded ? "pill ok" : "pill"} data-pw="health-model">
                model {health.model_loaded ? "đã nạp" : "chưa nạp"}
              </span>
              <span className={health.engine_ready ? "pill ok" : "pill"} data-pw="health-vi-engine">
                OmniVoice {health.engine_ready ? "sẵn sàng" : "đang khởi động…"}
              </span>
              <span className={health.ffmpeg ? "pill ok" : "pill bad"} data-pw="health-ffmpeg">
                ffmpeg {health.ffmpeg ? "sẵn sàng" : "thiếu"}
              </span>
            </>
          )}
        </div>
      </div>

      {notice !== null && <p className="notice" data-pw="notice">{notice}</p>}

      <section className="panel">
        <h2>1. Giọng nói tham chiếu</h2>
        <p className="sub">Tải lên đoạn giọng nói sạch của một người, dài 2–30 giây. Audio được khử nhiễu, cắt và chuẩn hóa ở server. Muốn giọng nói tiếng Việt, hãy dùng mẫu tham chiếu nói tiếng Việt.</p>
        <div className="row">
          <label className="file-label" data-pw="file-label">
            <input
              ref={fileInputRef}
              type="file"
              accept=".wav,.mp3,.ogg,.flac,.m4a,.aac,.opus,audio/*"
              data-pw="file-input"
              onChange={(e) => setFile(e.target.files?.[0] ?? null)}
            />
            {file ? file.name : "Chọn file âm thanh"}
          </label>
          <input
            type="text"
            placeholder="Tên giọng nói (không bắt buộc)"
            value={voiceName}
            onChange={(e) => setVoiceName(e.target.value)}
            data-pw="voice-name"
          />
          <button onClick={onUpload} disabled={uploading || !file} data-pw="upload-btn">
            {uploading ? "Đang tải lên…" : "Tải lên & nhân bản"}
          </button>
        </div>

        <div className="voice-list" data-pw="voice-list">
          {voices.length === 0 && <p className="empty">Chưa có giọng nói — hãy tải lên audio tham chiếu.</p>}
          {voices.map((v) => (
            <div className="voice-item" key={v.id} data-pw={`voice-${v.id}`}>
              <div>
                <div className="name">{v.name}</div>
                <div className="meta">
                  {v.duration_seconds.toFixed(1)} giây · {v.sample_rate} Hz · {v.source_filename ?? "đã tải lên"}
                </div>
                <audio controls src={voiceAudioUrl(v.id)} preload="none" data-pw={`voice-audio-${v.id}`} />
              </div>
              <div className="row">
                {voiceId === v.id && <span className="pill ok">đang chọn</span>}
                <button
                  className="secondary"
                  onClick={() => onPreview(`upload:${v.id}`, v.id, "")}
                  disabled={previewingKey === `upload:${v.id}`}
                  data-pw={`voice-preview-${v.id}`}
                >
                  {previewingKey === `upload:${v.id}`
                    ? previewPlayingKey === `upload:${v.id}`
                      ? "Đang phát…"
                      : "Đang tạo mẫu…"
                    : "Nghe thử"}
                </button>
                <button className="danger" onClick={() => onDeleteVoice(v.id)} data-pw={`voice-delete-${v.id}`}>
                  Xóa
                </button>
              </div>
            </div>
          ))}
        </div>
      </section>

      <section className="panel">
        <h2>2. Tạo giọng nói</h2>
        <p className="sub">Công việc được xếp hàng ở worker FastAPI; tiến độ hiển thị trực tiếp qua SSE.</p>
        <textarea
          value={text}
          maxLength={4000}
          onChange={(e) => setText(e.target.value)}
          placeholder="Nhập văn bản cần đọc…"
          data-pw="text-input"
        />
        <div className="voice-menu" data-pw="voice-menu">
          <div className="voice-menu-title">Giọng nói</div>
          {presetsLoading ? (
            <p className="empty">Đang tải danh sách giọng dựng sẵn…</p>
          ) : presets.length === 0 ? (
            <p className="empty">Chưa có giọng dựng sẵn (OmniVoice chưa sẵn sàng).</p>
          ) : (
            presets.map((p) => (
              <label className="voice-radio" key={p.name} data-pw={`voice-radio-${p.name}`}>
                <input
                  type="radio"
                  name="voice-choice"
                  checked={voicePreset === p.name}
                  onChange={() => selectPreset(p.name)}
                />
                <span className="voice-radio-name">
                  {p.featured !== null ? "⭐ " : ""}
                  {p.name}
                </span>
                <span className="voice-radio-desc">
                  {p.description
                    ? p.description
                    : GENDER_LABELS[p.gender] ?? p.gender}
                </span>
                <button
                  className="secondary voice-preview-btn"
                  onClick={(e) => {
                    e.preventDefault();
                    void onPreview(`preset:${p.name}`, "", p.name);
                  }}
                  disabled={previewingKey === `preset:${p.name}`}
                  data-pw={`voice-preview-${p.name}`}
                >
                  {previewingKey === `preset:${p.name}`
                    ? previewPlayingKey === `preset:${p.name}`
                      ? "Đang phát…"
                      : "Đang tạo mẫu…"
                    : "Nghe thử"}
                </button>
              </label>
            ))
          )}
          {voices.length > 0 && (
            <div className="voice-radio-uploaded" data-pw="voice-radio-uploaded">
              {voices.map((v) => (
                <label className="voice-radio" key={v.id} data-pw={`voice-radio-upload-${v.id}`}>
                  <input
                    type="radio"
                    name="voice-choice"
                    checked={!voicePreset && voiceId === v.id}
                    onChange={() => selectUploaded(v.id)}
                  />
                  <span className="voice-radio-name">{v.name}</span>
                  <span className="voice-radio-desc">giọng tải lên · {v.duration_seconds.toFixed(1)} giây</span>
                  <button
                    className="secondary voice-preview-btn"
                    onClick={(e) => {
                      e.preventDefault();
                      void onPreview(`upload:${v.id}`, v.id, "");
                    }}
                    disabled={previewingKey === `upload:${v.id}`}
                    data-pw={`voice-preview-radio-${v.id}`}
                  >
                    {previewingKey === `upload:${v.id}`
                      ? previewPlayingKey === `upload:${v.id}`
                        ? "Đang phát…"
                        : "Đang tạo mẫu…"
                      : "Nghe thử"}
                  </button>
                </label>
              ))}
            </div>
          )}
        </div>
        <div className="row" style={{ marginTop: 12 }}>
          <div className="slider-group">
            <span>Tốc độ <span className="val">{speed.toFixed(2)}</span></span>
            <input type="range" min={0.5} max={2} step={0.05} value={speed}
              onChange={(e) => setSpeed(Number(e.target.value))} data-pw="speed-slider" />
          </div>
          <div className="slider-group">
            <span>Nhiệt độ <span className="val">{temperature.toFixed(2)}</span></span>
            <input type="range" min={0} max={1} step={0.05} value={temperature}
              onChange={(e) => setTemperature(Number(e.target.value))} data-pw="temperature-slider" />
          </div>
          <input type="text" placeholder="Seed (không bắt buộc)" value={seed}
            onChange={(e) => setSeed(e.target.value)} data-pw="seed-input" style={{ width: 120 }} />
          <button onClick={onGenerate} disabled={generating} data-pw="generate-btn">
            {generating ? "Đang xếp hàng…" : "Tạo giọng nói"}
          </button>
        </div>
      </section>

      <section className="panel">
        <h2>3. Hàng đợi tạo giọng nói</h2>
        <div className="job-list" data-pw="job-list">
          {jobs.length === 0 && <p className="empty">Chưa có công việc nào.</p>}
          {jobs.map((j) => (
            <div className="job-item" key={j.id} data-pw={`job-${j.id}`}>
              <div className="top">
                <span className={`status ${j.status}`} data-pw={`job-status-${j.id}`}>
                  {STATUS_LABELS[j.status] ?? j.status}
                </span>
                <div className="row">
                  {j.status === "done" && (
                    <a className="download" href={jobDownloadUrl(j.id)} data-pw={`job-download-${j.id}`}>
                      Tải WAV
                    </a>
                  )}
                  {(j.status === "queued" || j.status === "running") && (
                    <button className="secondary" onClick={() => onCancel(j.id)} data-pw={`job-cancel-${j.id}`}>
                      Hủy
                    </button>
                  )}
                  <button className="danger" onClick={() => onDeleteJob(j.id)} data-pw={`job-delete-${j.id}`}>
                    Xóa
                  </button>
                </div>
              </div>
              <p className="text">{j.text}</p>
              {j.status === "running" && (
                <div className="progress" data-pw={`job-progress-${j.id}`}>
                  <div className="bar" style={{ width: `${Math.round(j.progress * 100)}%` }} />
                </div>
              )}
              {j.status === "done" && (
                <>
                  <audio controls src={jobAudioUrl(j.id)} preload="none" data-pw={`job-audio-${j.id}`} />
                  {j.duration_seconds !== null && (
                    <div className="meta" style={{ color: "var(--muted)", fontSize: 12, marginTop: 6 }}>
                      {j.duration_seconds.toFixed(2)} giây · {LANGUAGE_LABEL}
                    </div>
                  )}
                </>
              )}
              {j.status === "failed" && j.error && <p className="error" data-pw={`job-error-${j.id}`}>{j.error}</p>}
            </div>
          ))}
        </div>
      </section>
    </main>
  );
}
