import type { PipelineStageData, StageStatus } from "@/components/types";

const statusStyles: Record<StageStatus, string> = {
  pending: "border-line bg-surface-muted text-ink-muted",
  active: "border-info bg-info-soft text-info",
  complete: "border-success bg-success-soft text-success",
  error: "border-error bg-error-soft text-error",
};

const statusLabel: Record<StageStatus, string> = {
  pending: "Menunggu",
  active: "Diproses",
  complete: "Selesai",
  error: "Galat",
};

type PipelineStageProps = {
  stage: PipelineStageData;
};

export function PipelineStage({ stage }: PipelineStageProps) {
  return (
    <article
      data-testid={stage.testId}
      className="group relative overflow-hidden rounded-[var(--radius-card)] border border-line bg-surface p-5 shadow-[var(--shadow-card)] transition-transform duration-300 hover:-translate-y-0.5"
    >
      <div className="absolute -right-10 -top-10 h-24 w-24 rounded-full bg-accent-soft/40 blur-2xl transition-opacity duration-300 group-hover:opacity-80" />
      <div className="relative flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-3">
            <h3 className="text-lg font-semibold tracking-[-0.02em] text-ink">
              {stage.name}
            </h3>
            <span
              className={`rounded-full border px-3 py-1 font-mono text-[0.68rem] uppercase tracking-[0.18em] ${statusStyles[stage.status]}`}
            >
              {statusLabel[stage.status]}
            </span>
          </div>
          <p className="mt-2 text-sm leading-6 text-ink-muted">{stage.description}</p>
          <p className="mt-4 break-words rounded-2xl border border-line bg-background/70 p-4 font-mono text-xs leading-6 text-ink">
            {stage.detail}
          </p>
          {stage.status === "active" ? (
            <div className="mt-4 flex items-center gap-2 font-mono text-xs uppercase tracking-[0.18em] text-info">
              <span className="h-2 w-2 animate-pulse rounded-full bg-info" />
              thinking...
            </div>
          ) : null}
        </div>
        {typeof stage.latencyMs === "number" ? (
          <span className="shrink-0 rounded-full border border-line-strong bg-surface-strong px-3 py-1 font-mono text-xs font-semibold text-ink">
            {stage.latencyMs} ms
          </span>
        ) : null}
      </div>
    </article>
  );
}
