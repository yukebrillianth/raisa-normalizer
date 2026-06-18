import type { PipelineStageData, StageStatus } from "@/components/types";

const statusStyles: Record<StageStatus, string> = {
  pending: "border-line bg-surface-muted text-ink-muted",
  active: "border-info bg-info-soft text-info shadow-[0_0_0_4px_color-mix(in_srgb,var(--info-soft)_55%,transparent)]",
  complete: "border-success bg-success-soft text-success",
  error: "border-error bg-error-soft text-error",
};

const articleStyles: Record<StageStatus, string> = {
  pending: "border-line bg-surface opacity-80",
  active: "border-info bg-info-soft/35 shadow-[0_18px_50px_color-mix(in_srgb,var(--info)_14%,transparent)]",
  complete: "border-success bg-surface shadow-[0_14px_36px_color-mix(in_srgb,var(--success)_10%,transparent)]",
  error: "border-error bg-error-soft/40 shadow-[0_18px_50px_color-mix(in_srgb,var(--error)_13%,transparent)]",
};

const railStyles: Record<StageStatus, string> = {
  pending: "bg-line",
  active: "bg-info",
  complete: "bg-success",
  error: "bg-error",
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
      className={`stage-fade-in group relative overflow-hidden rounded-[var(--radius-card)] border p-5 transition-all duration-300 hover:-translate-y-0.5 ${articleStyles[stage.status]}`}
    >
      <div className={`absolute left-0 top-0 h-full w-1.5 ${railStyles[stage.status]}`} />
      <div className="absolute -right-10 -top-10 h-24 w-24 rounded-full bg-accent-soft/40 blur-2xl transition-opacity duration-300 group-hover:opacity-80" />
      <div className="relative flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-3">
            <h3 className="text-lg font-semibold tracking-[-0.02em] text-ink">
              {stage.name}
            </h3>
            <span
              className={`rounded-full border px-3 py-1 text-[0.68rem] uppercase tracking-[0.18em] ${statusStyles[stage.status]}`}
            >
              {statusLabel[stage.status]}
            </span>
            <span className="rounded-full border border-line bg-background/80 px-3 py-1 text-[0.68rem] uppercase tracking-[0.18em] text-ink-muted">
              {stage.provider || "provider menunggu"}
            </span>
          </div>
          <p className="mt-2 text-sm leading-6 text-ink-muted">{stage.description}</p>
          <p className="scrollable-panel mt-4 max-h-40 overflow-auto break-words rounded-2xl border border-line bg-background/70 p-4 text-xs leading-6 text-ink transition-colors duration-300">
            {stage.detail}
          </p>
          {stage.status === "active" ? (
            <div className="mt-4 flex items-center gap-2 text-xs uppercase tracking-[0.18em] text-info">
              <span className="h-2 w-2 animate-pulse rounded-full bg-info" />
              sedang diproses...
            </div>
          ) : null}
        </div>
        <span className="shrink-0 rounded-full border border-line-strong bg-surface-strong px-3 py-1 text-xs font-semibold text-ink">
          {typeof stage.latencyMs === "number" ? `${stage.latencyMs} ms` : "latensi —"}
        </span>
      </div>
    </article>
  );
}
