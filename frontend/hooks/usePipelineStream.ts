"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type {
  LatencyItem,
  PipelineError,
  PipelineStageData,
  RetrievalCandidate,
  StageStatus,
} from "@/components/types";

export type PipelinePhase = "idle" | "uploading" | "streaming" | "done" | "error";

type StageMeta = {
  id: string;
  name: string;
  description: string;
  provider: string;
  testId: string;
};

const STAGE_META: StageMeta[] = [
  {
    id: "stt",
    name: "1. Speech-to-Text",
    description: "Mengubah ujaran pengguna menjadi teks mentah untuk analisis berikutnya.",
    provider: "OpenAI Whisper",
    testId: "stage-stt",
  },
  {
    id: "normalize",
    name: "2. Normalisasi Query",
    description: "Membersihkan filler, memperbaiki istilah, dan menyusun query formal bahasa Indonesia.",
    provider: "LoRA LLM",
    testId: "stage-normalize",
  },
  {
    id: "embed",
    name: "3. Embedding",
    description: "Menyusun vektor dense dari query yang telah dinormalisasi.",
    provider: "BGE-M3",
    testId: "stage-embed",
  },
  {
    id: "retrieve",
    name: "4. Retrieval Kandidat",
    description: "Mengambil kandidat FAQ/dokumen menggunakan skor similarity dan pencocokan kata kunci.",
    provider: "pgvector",
    testId: "stage-retrieve",
  },
  {
    id: "baseline_rerank",
    name: "5. Reranking Baseline",
    description: "Mengurutkan ulang kandidat berdasarkan rerank skor hybrid sebelum LLM dipilih.",
    provider: "Hybrid scorer",
    testId: "stage-baseline-rerank",
  },
  {
    id: "select_and_verbalize",
    name: "6. Seleksi & Verbalization",
    description: "Memilih jawaban final lalu mengubahnya menjadi tuturan yang ringkas dan natural.",
    provider: "LoRA LLM",
    testId: "stage-select-verbalize",
  },
  {
    id: "tts",
    name: "7. Text-to-Speech",
    description: "Menghasilkan audio jawaban akhir untuk diputar ke pengguna.",
    provider: "Supertonic / OpenAI",
    testId: "stage-tts",
  },
];

const STAGE_META_BY_ID = Object.fromEntries(STAGE_META.map((stage) => [stage.id, stage])) as Record<string, StageMeta>;

const LATENCY_LABELS: Record<string, string> = {
  stt: "STT",
  normalization: "Normalize",
  embedding: "Embed",
  retrieval: "Retrieve",
  llm_selection: "Select",
  tts: "TTS",
};

type SseEventData = Record<string, unknown>;

type SseEvent = {
  event: string;
  data: SseEventData;
};

type ProviderInfo = {
  provider?: string;
  model?: string;
  embedding?: string;
  embeddingDim?: number | null;
};

type BaselineRerankData = {
  question?: string;
  answer?: string;
  similarity?: number;
  rerank_score?: number;
};

type LLMSelectionData = {
  provider?: string;
  selected_rank?: number | null;
  selected_question?: string;
  selected_answer?: string;
  spoken_answer?: string;
  reason?: string;
  latency_ms?: number;
  fallback_used?: boolean;
  refused?: boolean;
  refusal_reason?: string;
};

type TTSData = {
  provider?: string;
  fallback_used?: boolean;
  audio_url?: string;
  latency_ms?: number;
};

type FinalResponse = {
  request_id?: string;
  transcript?: string;
  normalized_query?: string;
  normalizer?: { provider?: string; latency_ms?: number } | null;
  retrieval?: {
    top_k?: number;
    similarity_threshold?: number;
    rerank_keyword_weight?: number;
    candidates?: Array<{
      question: string;
      answer: string;
      similarity: number;
      keyword_score: number;
      rerank_score: number;
    }>;
    baseline_rerank_selected?: BaselineRerankData | null;
    answered?: boolean;
  };
  answer?: string;
  spoken_answer?: string;
  llm_selection?: LLMSelectionData | null;
  tts?: TTSData | null;
  timing?: {
    stt_ms?: number;
    normalization_ms?: number;
    embedding_ms?: number;
    retrieval_ms?: number;
    llm_selection_ms?: number;
    tts_ms?: number;
    total_ms?: number;
  };
  errors?: Array<{ stage: string; message: string; detail?: string }>;
};

export type PipelineState = {
  phase: PipelinePhase;
  requestId: string | null;
  stages: PipelineStageData[];
  transcript: string;
  normalizedQuery: string;
  providerInfo: ProviderInfo;
  candidates: RetrievalCandidate[];
  baselineSelected: BaselineRerankData | null;
  llmSelection: LLMSelectionData | null;
  finalAnswer: string;
  spokenAnswer: string;
  latencyItems: LatencyItem[];
  errors: PipelineError[];
  audioUrl: string;
  ttsProvider: string;
  ttsFallbackUsed: boolean;
  ttsLatencyMs: number | null;
  thresholdGateSkipped: boolean;
};

type PipelineActionResult = {
  error: string | null;
};

export type UsePipelineStreamReturn = {
  state: PipelineState;
  submitAudio: (audioBlob: Blob, mimeType: string) => Promise<PipelineActionResult>;
  reset: () => void;
};

const initialState: PipelineState = {
  phase: "idle",
  requestId: null,
  stages: STAGE_META.map((meta) => ({
    id: meta.id,
    name: meta.name,
    description: meta.description,
    status: "pending",
    testId: meta.testId,
    provider: meta.provider,
    detail: "Menunggu event pipeline.",
  })),
  transcript: "",
  normalizedQuery: "",
  providerInfo: {},
  candidates: [],
  baselineSelected: null,
  llmSelection: null,
  finalAnswer: "",
  spokenAnswer: "",
  latencyItems: [],
  errors: [],
  audioUrl: "",
  ttsProvider: "-",
  ttsFallbackUsed: false,
  ttsLatencyMs: null,
  thresholdGateSkipped: false,
};

function parseSseEvent(buffer: string): SseEvent[] {
  const events: SseEvent[] = [];
  const blocks = buffer.split("\n\n");
  for (const block of blocks) {
    if (!block.trim()) continue;
    let eventName = "message";
    let data = "";
    for (const rawLine of block.split("\n")) {
      const line = rawLine.trim();
      if (!line) continue;
      if (line.startsWith("event:")) {
        eventName = line.slice(6).trim();
      } else if (line.startsWith("data:")) {
        data += line.slice(5).trim();
      }
    }
    if (!data) continue;
    try {
      events.push({ event: eventName, data: JSON.parse(data) });
    } catch {
      events.push({ event: eventName, data: { raw: data } });
    }
  }
  return events;
}

function getBackendUrl(): string {
  const raw = process.env.NEXT_PUBLIC_BACKEND_URL;
  if (raw && raw.trim()) return raw.replace(/\/$/, "");
  return "http://localhost:8000";
}

function buildAudioUrl(mimeType: string): string {
  const params = new URLSearchParams();
  params.set("mime_type", mimeType || "audio/webm");
  return `${getBackendUrl()}/api/pipeline/audio-query/stream?${params.toString()}`;
}

function applyStageUpdate(
  state: PipelineState,
  stageId: string,
  patch: Partial<PipelineStageData>,
): PipelineState {
  const stages = state.stages.map((stage) =>
    stage.id === stageId ? { ...stage, ...patch } : stage,
  );
  return { ...state, stages };
}

function applyPipelineStart(state: PipelineState, data: SseEventData): PipelineState {
  const stages = STAGE_META.map((meta) => ({
    id: meta.id,
    name: meta.name,
    description: meta.description,
    status: "pending" as StageStatus,
      testId: meta.testId,
      provider: meta.provider,
      detail: "Menunggu event pipeline.",
    }));
  return {
    ...initialState,
    ...state,
    phase: "streaming",
    requestId: typeof data.request_id === "string" ? data.request_id : state.requestId,
    stages,
    errors: [],
    latencyItems: [],
    candidates: [],
    baselineSelected: null,
    llmSelection: null,
    finalAnswer: "",
    spokenAnswer: "",
    audioUrl: "",
    ttsProvider: "-",
    ttsFallbackUsed: false,
    ttsLatencyMs: null,
    thresholdGateSkipped: false,
  };
}

function applyStageStart(state: PipelineState, data: SseEventData): PipelineState {
  const stageId = typeof data.stage === "string" ? data.stage : "";
  if (!stageId) return state;
  const meta = STAGE_META_BY_ID[stageId];
  return applyStageUpdate(state, stageId, {
    status: "active",
    detail: meta
      ? `Tahap ${meta.name.toLowerCase()} sedang diproses oleh backend.`
      : `Tahap ${stageId} sedang diproses oleh backend.`,
  });
}

function applyStageComplete(state: PipelineState, data: SseEventData): PipelineState {
  const stageId = typeof data.stage === "string" ? data.stage : "";
  if (!stageId) return state;
  const payload = (data.data ?? {}) as Record<string, unknown>;
  const stageLatencyMs = typeof payload.latency_ms === "number" ? payload.latency_ms : undefined;
  let nextState = state;

  switch (stageId) {
    case "stt": {
      const transcript = typeof payload.transcript === "string" ? payload.transcript : "";
      const language = typeof payload.language === "string" ? payload.language : null;
      nextState = {
        ...nextState,
        transcript,
        providerInfo: {
          ...nextState.providerInfo,
          provider: "openai_whisper",
        },
      };
      nextState = applyStageUpdate(nextState, stageId, {
        status: "complete",
        provider: "openai_whisper",
        latencyMs: stageLatencyMs,
        detail: transcript
          ? `Transkrip: ${transcript}${language ? ` (${language})` : ""}`
          : "Transkrip kosong diterima dari STT.",
      });
      break;
    }
    case "normalize": {
      const normalizedQuery = typeof payload.normalized_query === "string" ? payload.normalized_query : "";
      const provider =
        typeof payload.provider === "string"
          ? payload.provider
          : nextState.providerInfo.provider ?? STAGE_META_BY_ID[stageId]?.provider;
      nextState = {
        ...nextState,
        normalizedQuery,
        providerInfo: {
          ...nextState.providerInfo,
          provider,
          model: provider,
        },
      };
      nextState = applyStageUpdate(nextState, stageId, {
        status: "complete",
        provider,
        latencyMs: stageLatencyMs,
        detail: normalizedQuery ? `Query: ${normalizedQuery}` : "Normalisasi gagal, menggunakan transkrip mentah.",
      });
      break;
    }
    case "embed": {
      const embeddingDim = typeof payload.embedding_dim === "number" ? payload.embedding_dim : null;
      nextState = {
        ...nextState,
        providerInfo: {
          ...nextState.providerInfo,
          embedding: "bge-m3",
          embeddingDim,
        },
      };
      nextState = applyStageUpdate(nextState, stageId, {
        status: "complete",
        provider: embeddingDim ? `bge-m3 / ${embeddingDim}d` : "bge-m3",
        latencyMs: stageLatencyMs,
        detail: embeddingDim ? `Embedding dimensi: ${embeddingDim}` : "Embedding selesai tanpa metadata dimensi.",
      });
      break;
    }
    case "retrieve": {
      const candidates = Array.isArray(payload.candidates)
        ? (payload.candidates as Array<Record<string, any>>).map((candidate, index) => ({
            rank: index + 1,
            question: String(candidate.question ?? ""),
            answer: String(candidate.answer ?? ""),
            similarity: Number(candidate.similarity ?? 0),
            keyword_score: Number(candidate.keyword_score ?? 0),
            rerank_score: Number(candidate.rerank_score ?? 0),
          }))
        : [];
      nextState = { ...nextState, candidates };
      nextState = applyStageUpdate(nextState, stageId, {
        status: "complete",
        provider: "pgvector + keyword",
        latencyMs: stageLatencyMs,
        detail: candidates.length > 0
          ? `Top-${candidates.length} kandidat ditemukan dengan skor rerank hybrid.`
          : "Tidak ada kandidat yang dikembalikan oleh retrieval.",
      });
      break;
    }
    case "baseline_rerank": {
      if (payload.threshold_gate === "SKIPPED") {
        nextState = { ...nextState, thresholdGateSkipped: true };
        nextState = applyStageUpdate(nextState, stageId, {
          status: "complete",
          provider: "threshold gate",
          latencyMs: stageLatencyMs,
          detail: "Threshold gate: semua kandidat di bawah ambang similarity, LLM dilewati.",
        });
      } else {
        const selected = payload.selected as BaselineRerankData | null | undefined;
        nextState = { ...nextState, baselineSelected: selected ?? null };
        nextState = applyStageUpdate(nextState, stageId, {
          status: "complete",
          provider: "hybrid rerank",
          latencyMs: stageLatencyMs,
          detail: selected
            ? `Baseline rerank memilih kandidat dengan rerank_score=${
                typeof selected.rerank_score === "number" ? selected.rerank_score.toFixed(3) : "?"
              }.`
            : "Baseline rerank selesai tanpa kandidat terpilih.",
        });
      }
      break;
    }
    case "select_and_verbalize": {
      if (payload.skipped === true) {
        nextState = {
          ...nextState,
          llmSelection: {
            provider: "-",
            selected_rank: null,
            reason: "Threshold gate dilewati karena tidak ada kandidat yang lolos ambang similarity.",
            fallback_used: true,
            refused: true,
          },
        };
        nextState = applyStageUpdate(nextState, stageId, {
          status: "complete",
          provider: "threshold gate",
          latencyMs: stageLatencyMs,
          detail: "LLM seleksi dilewati karena threshold gate.",
        });
      } else {
        const llmSelection: LLMSelectionData = {
          provider: typeof payload.provider === "string" ? payload.provider : undefined,
          selected_rank: typeof payload.selected_rank === "number" ? payload.selected_rank : null,
          selected_question: typeof payload.selected_question === "string" ? payload.selected_question : undefined,
          selected_answer: typeof payload.selected_answer === "string" ? payload.selected_answer : undefined,
          spoken_answer: typeof payload.spoken_answer === "string" ? payload.spoken_answer : undefined,
          reason: typeof payload.reason === "string" ? payload.reason : undefined,
          latency_ms: typeof payload.latency_ms === "number" ? payload.latency_ms : undefined,
          fallback_used: typeof payload.fallback_used === "boolean" ? payload.fallback_used : undefined,
          refused: typeof payload.refused === "boolean" ? payload.refused : undefined,
          refusal_reason: typeof payload.refusal_reason === "string" ? payload.refusal_reason : undefined,
        };
        nextState = { ...nextState, llmSelection };
        nextState = applyStageUpdate(nextState, stageId, {
          status: "complete",
          provider: llmSelection.provider ?? STAGE_META_BY_ID[stageId]?.provider,
          latencyMs: stageLatencyMs ?? llmSelection.latency_ms,
          detail: llmSelection.refused
            ? `LLM menolak memilih: ${llmSelection.refusal_reason || "tanpa alasan"}`
            : `LLM memilih rank #${llmSelection.selected_rank ?? "?"} dengan alasan seleksi tersedia.`,
        });
      }
      break;
    }
    case "tts": {
      const tts: TTSData = {
        provider: typeof payload.provider === "string" ? payload.provider : undefined,
        fallback_used: typeof payload.fallback_used === "boolean" ? payload.fallback_used : undefined,
        audio_url: typeof payload.audio_url === "string" ? payload.audio_url : undefined,
        latency_ms: typeof payload.latency_ms === "number" ? payload.latency_ms : undefined,
      };
      const audioUrl = tts.audio_url && tts.audio_url.length > 0
        ? `${getBackendUrl()}${tts.audio_url.startsWith("/") ? "" : "/"}${tts.audio_url}`
        : "";
      nextState = {
        ...nextState,
        audioUrl,
        ttsProvider: tts.provider ?? "-",
        ttsFallbackUsed: tts.fallback_used ?? false,
        ttsLatencyMs: typeof tts.latency_ms === "number" ? tts.latency_ms : null,
      };
      nextState = applyStageUpdate(nextState, stageId, {
        status: "complete",
        provider: tts.provider ?? STAGE_META_BY_ID[stageId]?.provider,
        latencyMs: stageLatencyMs ?? tts.latency_ms,
        detail: audioUrl
          ? `Audio TTS tersedia dari provider ${tts.provider ?? "-"}.`
          : "TTS selesai tanpa URL audio; jawaban teks tetap tersedia.",
      });
      break;
    }
    default:
      nextState = applyStageUpdate(nextState, stageId, {
        status: "complete",
        provider: STAGE_META_BY_ID[stageId]?.provider,
        latencyMs: stageLatencyMs,
        detail: JSON.stringify(payload),
      });
      break;
  }

  return nextState;
}

function applyStageError(state: PipelineState, data: SseEventData): PipelineState {
  const stageId = typeof data.stage === "string" ? data.stage : "pipeline";
  const recoverable = data.recoverable !== false;
  const error: PipelineError = {
    stage: stageId,
    message: typeof data.message === "string" ? data.message : "STAGE_ERROR",
    detail: typeof data.detail === "string" ? data.detail : undefined,
    recoverable,
  };
  let nextState = applyStageUpdate(state, stageId, {
    status: "error",
    detail: error.detail || error.message,
  });
  nextState = {
    ...nextState,
    errors: [...nextState.errors, error],
    phase: recoverable ? nextState.phase : "error",
  };
  return nextState;
}

function applyPipelineComplete(state: PipelineState, data: SseEventData): PipelineState {
  const response = (data.response ?? data.final_response ?? {}) as FinalResponse;
  const timing = response.timing ?? {};
  const latencyItems: LatencyItem[] = [
    { label: LATENCY_LABELS.stt, ms: Number(timing.stt_ms ?? 0) },
    { label: LATENCY_LABELS.normalization, ms: Number(timing.normalization_ms ?? 0) },
    { label: LATENCY_LABELS.embedding, ms: Number(timing.embedding_ms ?? 0) },
    { label: LATENCY_LABELS.retrieval, ms: Number(timing.retrieval_ms ?? 0) },
    { label: LATENCY_LABELS.llm_selection, ms: Number(timing.llm_selection_ms ?? 0) },
    { label: LATENCY_LABELS.tts, ms: Number(timing.tts_ms ?? 0) },
  ];
  const stageLatencyById: Record<string, number | undefined> = {
    stt: typeof timing.stt_ms === "number" ? timing.stt_ms : undefined,
    normalize: typeof timing.normalization_ms === "number" ? timing.normalization_ms : undefined,
    embed: typeof timing.embedding_ms === "number" ? timing.embedding_ms : undefined,
    retrieve: typeof timing.retrieval_ms === "number" ? timing.retrieval_ms : undefined,
    select_and_verbalize: typeof timing.llm_selection_ms === "number" ? timing.llm_selection_ms : undefined,
    tts: typeof timing.tts_ms === "number" ? timing.tts_ms : undefined,
  };

  let providerInfo = state.providerInfo;
  if (response.normalizer) {
    providerInfo = {
      ...providerInfo,
      provider: response.normalizer.provider ?? providerInfo.provider,
      model: response.normalizer.provider ?? providerInfo.model,
    };
  }

  const candidates = Array.isArray(response.retrieval?.candidates)
    ? (response.retrieval?.candidates ?? []).map((candidate, index) => ({
        rank: index + 1,
        question: candidate.question,
        answer: candidate.answer,
        similarity: candidate.similarity,
        keyword_score: candidate.keyword_score,
        rerank_score: candidate.rerank_score,
      }))
    : state.candidates;

  const llmSelection: LLMSelectionData | null = response.llm_selection ?? state.llmSelection;
  const tts: TTSData | null = response.tts ?? null;
  const audioUrl = tts?.audio_url
    ? `${getBackendUrl()}${tts.audio_url.startsWith("/") ? "" : "/"}${tts.audio_url}`
    : state.audioUrl;

  const finalErrors: PipelineError[] = Array.isArray(response.errors)
    ? response.errors.map((err) => ({
        stage: err.stage,
        message: err.message,
        detail: err.detail,
      }))
    : state.errors;

  return {
    ...state,
    phase: "done",
    stages: state.stages.map((stage) => ({
      ...stage,
      latencyMs: stageLatencyById[stage.id] ?? stage.latencyMs,
    })),
    transcript: response.transcript ?? state.transcript,
    normalizedQuery: response.normalized_query ?? state.normalizedQuery,
    providerInfo,
    candidates,
    baselineSelected: response.retrieval?.baseline_rerank_selected ?? state.baselineSelected,
    llmSelection,
    finalAnswer: response.answer ?? state.finalAnswer,
    spokenAnswer: response.spoken_answer ?? state.spokenAnswer,
    audioUrl,
    ttsProvider: tts?.provider ?? state.ttsProvider,
    ttsFallbackUsed: tts?.fallback_used ?? state.ttsFallbackUsed,
    ttsLatencyMs: typeof tts?.latency_ms === "number" ? tts.latency_ms : state.ttsLatencyMs,
    latencyItems,
    errors: finalErrors,
  };
}

function applySseEvent(state: PipelineState, event: SseEvent): PipelineState {
  switch (event.event) {
    case "pipeline_start":
      return applyPipelineStart(state, event.data);
    case "stage_start":
      return applyStageStart(state, event.data);
    case "stage_complete":
      return applyStageComplete(state, event.data);
    case "stage_error":
      return applyStageError(state, event.data);
    case "pipeline_complete":
      return applyPipelineComplete(state, event.data);
    default:
      return state;
  }
}

export function usePipelineStream(): UsePipelineStreamReturn {
  const [state, setState] = useState<PipelineState>(initialState);
  const abortControllerRef = useRef<AbortController | null>(null);
  const readerRef = useRef<ReadableStreamDefaultReader<Uint8Array> | null>(null);
  const reset = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    if (readerRef.current) {
      readerRef.current.cancel().catch(() => undefined);
      readerRef.current = null;
    }
    setState(initialState);
  }, []);

  const consumeStream = useCallback(async (reader: ReadableStreamDefaultReader<Uint8Array>) => {
    const decoder = new TextDecoder("utf-8");
    let buffer = "";
    try {
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        if (!value) continue;
        buffer += decoder.decode(value, { stream: true });
        const events = parseSseEvent(buffer);
        if (events.length > 0) {
          setState((current) => events.reduce(applySseEvent, current));
          const lastEventIndex = buffer.lastIndexOf("\n\n");
          buffer = lastEventIndex === -1 ? "" : buffer.slice(lastEventIndex + 2);
        }
      }
      const trailingEvents = parseSseEvent(buffer);
      if (trailingEvents.length > 0) {
        setState((current) => trailingEvents.reduce(applySseEvent, current));
      }
      setState((current) => (current.phase === "streaming" ? { ...current, phase: "done" } : current));
    } catch (error) {
      const message = error instanceof Error ? error.message : "Stream gagal dibaca.";
      setState((current) => ({
        ...current,
        phase: "error",
        errors: [
          ...current.errors,
          { stage: "pipeline", message: "STREAM_READ_ERROR", detail: message },
        ],
      }));
    } finally {
      reader.releaseLock();
      readerRef.current = null;
    }
  }, []);

  const submitAudio = useCallback(
    async (audioBlob: Blob, mimeType: string): Promise<PipelineActionResult> => {
      if (!audioBlob) {
        return { error: "Audio kosong, tidak dapat dikirim." };
      }
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
      if (readerRef.current) {
        await readerRef.current.cancel().catch(() => undefined);
        readerRef.current = null;
      }

      const controller = new AbortController();
      abortControllerRef.current = controller;

      const form = new FormData();
      form.append("audio", audioBlob, `recording.${(mimeType || "audio/webm").split("/").pop() || "webm"}`);
      form.append("mime_type", mimeType || "audio/webm");

      setState((current) => ({
        ...current,
        phase: "uploading",
        errors: [],
      }));

      let response: Response;
      try {
        response = await fetch(buildAudioUrl(mimeType), {
          method: "POST",
          body: form,
          signal: controller.signal,
          headers: {
            Accept: "text/event-stream",
          },
        });
      } catch (error) {
        const message = error instanceof Error ? error.message : "Gagal menghubungi backend.";
        setState((current) => ({
          ...current,
          phase: "error",
          errors: [
            ...current.errors,
            { stage: "pipeline", message: "BACKEND_UNREACHABLE", detail: message },
          ],
        }));
        return { error: message };
      }

      if (!response.ok || !response.body) {
        let detail = `HTTP ${response.status}`;
        try {
          const text = await response.text();
          if (text) detail = text;
        } catch {
          detail = `HTTP ${response.status}`;
        }
        setState((current) => ({
          ...current,
          phase: "error",
          errors: [
            ...current.errors,
            { stage: "pipeline", message: "BACKEND_ERROR", detail },
          ],
        }));
        return { error: detail };
      }

      const reader = response.body.getReader();
      readerRef.current = reader;
      setState((current) => ({ ...current, phase: "streaming" }));
      void consumeStream(reader);
      return { error: null };
    },
    [consumeStream],
  );

  useEffect(() => {
    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
        abortControllerRef.current = null;
      }
      if (readerRef.current) {
        readerRef.current.cancel().catch(() => undefined);
        readerRef.current = null;
      }
    };
  }, []);

  return useMemo(
    () => ({
      state,
      submitAudio,
      reset,
    }),
    [state, submitAudio, reset],
  );
}
