"use client";

import { TimelineStage } from "@/components/TimelineStage";
import type {
  LatencyItem,
  PipelineError,
  PipelineStageData,
  RetrievalCandidate,
} from "@/components/types";
import type { PipelinePhase } from "@/hooks/usePipelineStream";
import { useMemo, useState } from "react";

type PipelineTimelineProps = {
  stages: PipelineStageData[];
  latencyItems: LatencyItem[];
  errors: PipelineError[];
  candidates: RetrievalCandidate[];
  llmSelection: {
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
  } | null;
  phase: PipelinePhase;
  requestId: string | null;
};

export function PipelineTimeline({
  stages,
  latencyItems,
  errors,
  candidates,
  llmSelection,
  phase,
  requestId,
}: PipelineTimelineProps) {
  const [isCollapsed, setIsCollapsed] = useState(false);

  const totalMs = useMemo(
    () => latencyItems.reduce((sum, item) => sum + item.ms, 0),
    [latencyItems],
  );

  const phaseLabel = useMemo(() => {
    switch (phase) {
      case "idle":
        return "Menunggu input";
      case "uploading":
        return "Mengunggah audio";
      case "streaming":
        return "Pipeline aktif";
      case "done":
        return "Selesai";
      case "error":
        return "Error";
      default:
        return "—";
    }
  }, [phase]);

  const phaseColor = useMemo(() => {
    switch (phase) {
      case "streaming":
        return "text-its-blue";
      case "done":
        return "text-success";
      case "error":
        return "text-error";
      default:
        return "text-text-muted";
    }
  }, [phase]);

  return (
    <aside className="raisa-card overflow-hidden flex flex-col h-full">
      {/* Header */}
      <button
        onClick={() => setIsCollapsed(!isCollapsed)}
        className="flex items-center justify-between w-full px-4 py-3 border-b border-surface-3 text-left hover:bg-surface-1 transition-colors"
      >
        <div className="flex items-center gap-2">
          <svg
            width="14"
            height="14"
            viewBox="0 0 24 24"
            fill="none"
            stroke="var(--its-blue)"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <circle cx="12" cy="12" r="10" />
            <polyline points="12 6 12 12 16 14" />
          </svg>
          <span className="text-sm font-display text-its-cover">Pipeline</span>
        </div>
        <div className="flex items-center gap-2">
          <span className={`text-[11px] ${phaseColor}`}>{phaseLabel}</span>
          <svg
            width="14"
            height="14"
            viewBox="0 0 24 24"
            fill="none"
            stroke="var(--text-muted)"
            strokeWidth="2"
            strokeLinecap="round"
            className={`transition-transform duration-200 ${isCollapsed ? "-rotate-90" : "rotate-0"}`}
          >
            <polyline points="6 9 12 15 18 9" />
          </svg>
        </div>
      </button>

      {/* Collapsible content */}
      {!isCollapsed && (
        <div className="flex-1 overflow-y-auto scrollable">
          {/* Request ID */}
          {requestId && (
            <div className="px-4 pt-3 pb-1">
              <span className="text-[10px] text-text-muted">{requestId}</span>
            </div>
          )}

          {/* Timeline stages */}
          <div className="px-4 py-3">
            {stages.map((stage, i) => (
              <TimelineStage
                key={stage.id}
                name={stage.name}
                status={stage.status}
                detail={stage.detail}
                latencyMs={stage.latencyMs}
                provider={stage.provider}
                isLast={i === stages.length - 1}
                index={i}
              />
            ))}
          </div>

          {/* Retrieval candidates (expandable) */}
          {candidates.length > 0 && (
            <CandidatesSection candidates={candidates} />
          )}

          {llmSelection && <LLMSelectionSection selection={llmSelection} />}

          {/* Latency summary */}
          {latencyItems.some((item) => item.ms > 0) && (
            <div className="px-4 py-3 border-t border-surface-3">
              <p className="text-[11px] uppercase tracking-widest text-text-muted mb-2">
                Latency
              </p>
              <div className="space-y-1">
                {latencyItems
                  .filter((item) => item.ms > 0)
                  .map((item) => (
                    <div key={item.label} className="flex items-center gap-2">
                      <span className="text-xs text-text-secondary flex-1">
                        {item.label}
                      </span>
                      <div className="flex-1 h-1.5 bg-surface-2 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-its-blue rounded-full transition-all duration-500"
                          style={{
                            width: `${Math.min((item.ms / Math.max(totalMs, 1)) * 100, 100)}%`,
                          }}
                        />
                      </div>
                      <span className="text-[11px] text-text-muted w-14 text-right">
                        {item.ms >= 1000
                          ? `${(item.ms / 1000).toFixed(1)}s`
                          : `${Math.round(item.ms)}ms`}
                      </span>
                    </div>
                  ))}
                <div className="flex items-center justify-between pt-1 mt-1 border-t border-surface-3">
                  <span className="text-xs font-medium text-text-primary">
                    Total
                  </span>
                  <span className="text-[11px] text-its-blue font-medium">
                    {totalMs >= 1000
                      ? `${(totalMs / 1000).toFixed(1)}s`
                      : `${Math.round(totalMs)}ms`}
                  </span>
                </div>
              </div>
            </div>
          )}

          {/* Errors */}
          {errors.length > 0 && (
            <div className="px-4 py-3 border-t border-surface-3">
              <p className="text-[11px] uppercase tracking-widest text-error mb-2">
                Errors
              </p>
              <div className="space-y-1.5">
                {errors.map((err, i) => (
                  <div
                    key={i}
                    className="text-xs text-error bg-error-soft rounded px-2 py-1.5"
                  >
                    <span className="font-medium">{err.stage}</span>
                    <span className="text-text-secondary mx-1">→</span>
                    <span>{err.message}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </aside>
  );
}

/* ── LLM selection sub-section ───────────────────── */

function LLMSelectionSection({
  selection,
}: {
  selection: NonNullable<PipelineTimelineProps["llmSelection"]>;
}) {
  const [expanded, setExpanded] = useState(true);
  const selectedRank = selection.selected_rank
    ? `#${selection.selected_rank}`
    : "—";

  return (
    <div className="px-4 py-3 border-t border-surface-3">
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex items-center justify-between w-full text-left"
      >
        <div>
          <p className="text-[11px] uppercase tracking-widest text-text-muted">
            LLM Selection
          </p>
          <p className="text-xs text-text-secondary mt-0.5">
            selected_rank {selectedRank}
          </p>
        </div>
        <svg
          width="12"
          height="12"
          viewBox="0 0 24 24"
          fill="none"
          stroke="var(--text-muted)"
          strokeWidth="2"
          strokeLinecap="round"
          className={`transition-transform duration-200 ${expanded ? "rotate-180" : "rotate-0"}`}
        >
          <polyline points="6 9 12 15 18 9" />
        </svg>
      </button>
      {expanded && (
        <div
          className="mt-2 space-y-2 animate-fade-in text-xs"
          data-testid="llm-selection-sidebar"
        >
          {selection.selected_question ? (
            <p className="rounded bg-surface-1 border border-surface-3 p-2 text-text-secondary leading-relaxed">
              <span className="font-medium text-text-primary">Q:</span>{" "}
              {selection.selected_question}
            </p>
          ) : null}
          {selection.selected_answer ? (
            <p className="rounded bg-success-soft border border-success/30 p-2 text-text-secondary leading-relaxed">
              <span className="font-medium text-success">A:</span>{" "}
              {selection.selected_answer}
            </p>
          ) : null}
          <p className="text-text-muted leading-relaxed">
            {selection.reason ||
              selection.refusal_reason ||
              "LLM selection metadata diterima."}
          </p>
        </div>
      )}
    </div>
  );
}

/* ── Candidates sub-section ─────────────────────── */

function CandidatesSection({
  candidates,
}: {
  candidates: RetrievalCandidate[];
}) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="px-4 py-3 border-t border-surface-3">
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex items-center justify-between w-full text-left"
      >
        <p className="text-[11px] uppercase tracking-widest text-text-muted">
          Retrieval ({candidates.length} kandidat)
        </p>
        <svg
          width="12"
          height="12"
          viewBox="0 0 24 24"
          fill="none"
          stroke="var(--text-muted)"
          strokeWidth="2"
          strokeLinecap="round"
          className={`transition-transform duration-200 ${expanded ? "rotate-180" : "rotate-0"}`}
        >
          <polyline points="6 9 12 15 18 9" />
        </svg>
      </button>
      {expanded && (
        <div
          className="mt-2 space-y-2 animate-fade-in"
          data-testid="retrieval-candidates"
        >
          {candidates.map((c) => (
            <div
              key={c.rank}
              className="rounded bg-surface-1 border border-surface-3 p-2.5 text-xs"
            >
              <div className="flex items-center gap-2 mb-1">
                <span className="text-its-blue font-medium">#{c.rank}</span>
                <span className="text-text-muted">
                  sim={c.similarity.toFixed(3)} rr={c.rerank_score.toFixed(3)}
                </span>
              </div>
              <p className="text-text-secondary leading-relaxed">
                <span className="font-medium text-text-primary">Q:</span>{" "}
                {c.question}
              </p>
              <p className="text-text-muted leading-relaxed mt-0.5">
                <span className="font-medium text-text-secondary">A:</span>{" "}
                {c.answer}
              </p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
