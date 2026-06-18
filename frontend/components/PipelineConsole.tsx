"use client";

import { useCallback, useMemo } from "react";
import { AudioTtsPanel } from "@/components/AudioTtsPanel";
import { ErrorPanel } from "@/components/ErrorPanel";
import { LatencyTimeline } from "@/components/LatencyTimeline";
import { LLMSelection } from "@/components/LLMSelection";
import { PipelineStage } from "@/components/PipelineStage";
import { RecordControls } from "@/components/RecordControls";
import { RetrievalCandidates } from "@/components/RetrievalCandidates";
import { TranscriptPanel } from "@/components/TranscriptPanel";
import { usePipelineStream } from "@/hooks/usePipelineStream";

export function PipelineConsole() {
  const { state, submitAudio, reset } = usePipelineStream();

  const handleRecordingComplete = useCallback(
    async (blob: Blob, mimeType: string) => {
      reset();
      await submitAudio(blob, mimeType);
    },
    [reset, submitAudio],
  );

  const recordStatus = useMemo(() => {
    switch (state.phase) {
      case "idle":
        return "siap merekam pertanyaan bahasa Indonesia";
      case "uploading":
        return "mengunggah audio ke backend pipeline";
      case "streaming":
        return `streaming SSE (${state.requestId ?? "request belum tersedia"})`;
      case "done":
        return `pipeline selesai (${state.requestId ?? "request tidak diketahui"})`;
      case "error":
        return "pipeline berhenti karena error";
      default:
        return "siap merekam";
    }
  }, [state.phase, state.requestId]);

  const recordError = useMemo(() => {
    if (state.errors.length === 0) return undefined;
    const lastError = state.errors[state.errors.length - 1];
    return `${lastError.stage}: ${lastError.message}`;
  }, [state.errors]);

  const providerInfo = useMemo(() => {
    const parts: string[] = [];
    if (state.providerInfo.provider) parts.push(`provider=${state.providerInfo.provider}`);
    if (state.providerInfo.model) parts.push(`model=${state.providerInfo.model}`);
    if (state.providerInfo.embedding) {
      parts.push(
        `embedding=${state.providerInfo.embedding}${
          state.providerInfo.embeddingDim ? `(${state.providerInfo.embeddingDim})` : ""
        }`,
      );
    }
    return parts.length > 0 ? parts.join(" | ") : "Menunggu metadata provider dari pipeline.";
  }, [state.providerInfo]);

  const llmSelectionRank = state.llmSelection?.selected_rank ?? 0;
  const llmSelectionReason = state.llmSelection?.reason ?? "Menunggu hasil seleksi LLM dari pipeline.";
  const spokenAnswer = state.spokenAnswer || state.finalAnswer || "Menunggu spoken_answer dari pipeline.";
  const finalAnswer = state.finalAnswer || "Menunggu jawaban final dari pipeline.";

  const baselineDetail = useMemo(() => {
    if (state.thresholdGateSkipped) {
      return "Threshold gate dilewati; baseline rerank tidak memilih kandidat karena similarity di bawah ambang.";
    }
    if (state.baselineSelected) {
      return `Baseline rerank memilih kandidat dengan rerank_score=${
        typeof state.baselineSelected.rerank_score === "number"
          ? state.baselineSelected.rerank_score.toFixed(3)
          : "?"
      }.`;
    }
    return "Menunggu hasil baseline rerank dari pipeline.";
  }, [state.baselineSelected, state.thresholdGateSkipped]);

  const stageDetailsOverride = useMemo(() => {
    const overrides: Record<string, string> = {};
    if (state.transcript) {
      overrides.stt = `Transkrip: ${state.transcript}`;
    }
    if (state.normalizedQuery) {
      overrides.normalize = `Query: ${state.normalizedQuery}`;
    }
    if (state.candidates.length > 0) {
      overrides.retrieve = `Top-${state.candidates.length} kandidat ditemukan dengan skor rerank hybrid.`;
    }
    overrides.baseline_rerank = baselineDetail;
    return overrides;
  }, [state.transcript, state.normalizedQuery, state.candidates.length, baselineDetail]);

  const displayStages = useMemo(
    () =>
      state.stages.map((stage) => ({
        ...stage,
        detail: stageDetailsOverride[stage.id] || stage.detail,
      })),
    [state.stages, stageDetailsOverride],
  );

  const hasTranscript = Boolean(state.transcript || state.normalizedQuery);
  const hasCandidates = state.candidates.length > 0;
  const hasLlmSelection = Boolean(state.llmSelection);

  return (
    <>
      <RecordControls
        onRecordingCompleteAction={handleRecordingComplete}
        status={recordStatus}
        error={recordError}
        isUploading={state.phase === "uploading"}
      />

      <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_22rem]">
        <div className="space-y-6">
          <section className="space-y-4">
            <div className="flex flex-wrap items-end justify-between gap-3">
              <div>
                <p className="font-mono text-xs uppercase tracking-[0.28em] text-accent-strong">
                  Vertical pipeline flow
                </p>
                <h2 className="mt-2 text-2xl font-semibold tracking-[-0.04em] text-ink">
                  Tahapan pemrosesan suara
                </h2>
              </div>
              <span className="rounded-full border border-line bg-surface px-4 py-2 font-mono text-xs text-ink-muted">
                {state.phase === "idle"
                  ? "idle / menunggu audio"
                  : state.phase === "done"
                    ? "pipeline_complete"
                    : "live SSE stream"}
              </span>
            </div>
            <div className="space-y-4">
              {displayStages.map((stage) => (
                <PipelineStage key={stage.id} stage={stage} />
              ))}
            </div>
          </section>

          {hasTranscript ? (
            <TranscriptPanel
              transcript={state.transcript || "—"}
              normalizedQuery={state.normalizedQuery || state.transcript || "—"}
              providerInfo={providerInfo}
            />
          ) : null}

          {hasCandidates ? <RetrievalCandidates candidates={state.candidates} /> : null}

          {hasLlmSelection ? (
            <LLMSelection
              selectedRank={llmSelectionRank}
              reason={llmSelectionReason}
              spokenAnswer={spokenAnswer}
              finalAnswer={finalAnswer}
            />
          ) : null}

          <AudioTtsPanel
            audioUrl={state.audioUrl}
            provider={state.ttsProvider}
            fallbackUsed={state.ttsFallbackUsed}
            latencyMs={state.ttsLatencyMs ?? undefined}
            status={
              state.audioUrl
                ? "Audio jawaban tersedia dari backend."
                : state.phase === "done"
                  ? "TTS selesai tanpa URL audio; jawaban teks tetap ditampilkan."
                  : "Menunggu event TTS dari pipeline."
            }
          />
        </div>

        <div className="space-y-6 xl:sticky xl:top-6 xl:self-start">
          <LatencyTimeline items={state.latencyItems} />
          <ErrorPanel errors={state.errors} />
        </div>
      </div>
    </>
  );
}
