type LatencyItem = {
  label: string;
  ms: number;
};

type LatencyTimelineProps = {
  items: LatencyItem[];
};

export function LatencyTimeline({ items }: LatencyTimelineProps) {
  const max = Math.max(...items.map((item) => item.ms));

  return (
    <aside
      data-testid="latency-timeline"
      className="paper-panel rounded-[var(--radius-panel)] p-5"
    >
      <p className="font-mono text-xs uppercase tracking-[0.28em] text-accent-strong">
        Timeline latensi
      </p>
      <h2 className="mt-2 text-xl font-semibold tracking-[-0.03em] text-ink">
        Observasi performa
      </h2>
      <div className="mt-5 space-y-4">
        {items.map((item) => (
          <div key={item.label}>
            <div className="mb-2 flex items-center justify-between gap-3 text-sm">
              <span className="font-medium text-ink">{item.label}</span>
              <span className="font-mono text-xs text-ink-muted">{item.ms} ms</span>
            </div>
            <div className="h-2 overflow-hidden rounded-full bg-surface-muted">
              <div
                className="h-full rounded-full bg-accent"
                style={{ width: `${Math.max(8, (item.ms / max) * 100)}%` }}
              />
            </div>
          </div>
        ))}
      </div>
    </aside>
  );
}
