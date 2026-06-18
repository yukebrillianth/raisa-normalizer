"use client";

import type { StageStatus } from "@/components/types";

type TimelineStageProps = {
  name: string;
  status: StageStatus;
  detail: string;
  latencyMs?: number;
  provider?: string;
  isLast: boolean;
  index: number;
};

function StageIcon({ status }: { status: StageStatus }) {
  switch (status) {
    case "complete":
      return (
        <span className="inline-flex items-center justify-center h-6 w-6 rounded-full bg-success">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
            <polyline points="20 6 9 17 4 12" />
          </svg>
        </span>
      );
    case "active":
      return (
        <span className="inline-flex items-center justify-center h-6 w-6 rounded-full bg-its-blue animate-pulse-ring">
          <span className="h-2.5 w-2.5 rounded-full bg-white" />
        </span>
      );
    case "error":
      return (
        <span className="inline-flex items-center justify-center h-6 w-6 rounded-full bg-error">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
            <line x1="18" y1="6" x2="6" y2="18" />
            <line x1="6" y1="6" x2="18" y2="18" />
          </svg>
        </span>
      );
    default: // pending
      return (
        <span className="inline-flex items-center justify-center h-6 w-6 rounded-full border-2 border-surface-3 bg-surface-0">
          <span className="h-2 w-2 rounded-full bg-surface-3" />
        </span>
      );
  }
}

function ConnectorLine({ status, isLast }: { status: StageStatus; isLast: boolean }) {
  if (isLast) return null;

  const lineColor =
    status === "complete"
      ? "bg-success"
      : status === "active"
        ? "bg-its-blue"
        : status === "error"
          ? "bg-error"
          : "bg-surface-3";

  return (
    <div
      className={`absolute left-[11px] top-[28px] bottom-[-8px] w-0.5 ${lineColor} transition-colors duration-300`}
    />
  );
}

export function TimelineStage({
  name,
  status,
  detail,
  latencyMs,
  provider,
  isLast,
  index,
}: TimelineStageProps) {
  const isActive = status === "active" || status === "complete" || status === "error";

  return (
    <div
      className={`relative pl-9 pb-5 ${isActive ? "animate-stage-pop" : ""}`}
      style={isActive ? { animationDelay: `${index * 40}ms` } : undefined}
    >
      {/* Node icon */}
      <div className="absolute left-0 top-0">
        <StageIcon status={status} />
      </div>

      {/* Connector line */}
      <ConnectorLine status={status} isLast={isLast} />

      {/* Content */}
      <div className={`transition-opacity duration-200 ${status === "pending" ? "opacity-40" : "opacity-100"}`}>
        <div className="flex items-baseline gap-2 flex-wrap">
          <span className={`text-sm font-medium ${status === "pending" ? "text-text-muted" : "text-text-primary"}`}>
            {name}
          </span>
          {latencyMs !== undefined && (
            <span className="text-[11px] font-mono text-text-muted">
              {latencyMs >= 1000 ? `${(latencyMs / 1000).toFixed(1)}s` : `${Math.round(latencyMs)}ms`}
            </span>
          )}
          {status === "active" && (
            <span className="inline-flex items-center gap-1 text-[11px] font-mono text-its-blue">
              <svg width="10" height="10" viewBox="0 0 24 24" fill="none" className="animate-spin-slow">
                <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" opacity="0.3" />
                <path d="M12 2a10 10 0 0 1 10 10" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
              </svg>
              processing
            </span>
          )}
        </div>

        {/* Detail text — shown when complete or error */}
        {(status === "complete" || status === "error") && detail && (
          <p className={`mt-1 text-xs leading-relaxed ${status === "error" ? "text-error" : "text-text-muted"}`}>
            {detail}
          </p>
        )}

        {/* Provider badge */}
        {provider && status === "complete" && (
          <span className="inline-block mt-1.5 text-[10px] font-mono text-text-muted bg-surface-2 rounded px-1.5 py-0.5">
            {provider}
          </span>
        )}
      </div>
    </div>
  );
}
