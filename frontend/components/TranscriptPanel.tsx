import { SectionCard } from "@/components/SectionCard";

type TranscriptPanelProps = {
  transcript: string;
  normalizedQuery: string;
  providerInfo: string;
};

export function TranscriptPanel({
  transcript,
  normalizedQuery,
  providerInfo,
}: TranscriptPanelProps) {
  return (
    <SectionCard title="Jejak Bahasa" eyebrow="STT + Normalisasi">
      <div className="grid gap-4 lg:grid-cols-3">
        <div className="lg:col-span-2">
          <p className="mb-2 text-xs font-semibold uppercase tracking-[0.18em] text-ink-muted">
            Transkrip mentah
          </p>
          <p
            data-testid="transcript"
            className="min-h-28 break-words rounded-2xl border border-line bg-background/80 p-4 text-base leading-8 text-ink"
          >
            {transcript}
          </p>
        </div>
        <div className="space-y-4">
          <div>
            <p className="mb-2 text-xs font-semibold uppercase tracking-[0.18em] text-ink-muted">
              Query dinormalisasi
            </p>
            <p
              data-testid="normalized-query"
              className="break-words rounded-2xl border border-line bg-surface-strong p-4 text-sm leading-6 text-ink"
            >
              {normalizedQuery}
            </p>
          </div>
          <div>
            <p className="mb-2 text-xs font-semibold uppercase tracking-[0.18em] text-ink-muted">
              Provider model
            </p>
            <p
              data-testid="provider-info"
              className="rounded-2xl border border-info bg-info-soft p-4 text-sm text-info"
            >
              {providerInfo}
            </p>
          </div>
        </div>
      </div>
    </SectionCard>
  );
}
