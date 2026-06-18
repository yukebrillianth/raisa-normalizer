import { SectionCard } from "@/components/SectionCard";

type LLMSelectionProps = {
  selectedRank: number;
  reason: string;
  spokenAnswer: string;
  finalAnswer: string;
};

export function LLMSelection({
  selectedRank,
  reason,
  spokenAnswer,
  finalAnswer,
}: LLMSelectionProps) {
  return (
    <SectionCard title="Seleksi & Verbalization" eyebrow="LLM decision">
      <div data-testid="llm-selection" className="grid gap-4 lg:grid-cols-[0.55fr_1.45fr]">
        <div className="rounded-2xl border border-success bg-success-soft p-5">
          <p className="text-xs uppercase tracking-[0.18em] text-success">
            selected_rank
          </p>
          <p className="mt-3 text-5xl font-black tracking-[-0.08em] text-success">
            #{selectedRank}
          </p>
        </div>
        <div className="space-y-4">
          <div className="rounded-2xl border border-line bg-background/70 p-4">
            <p className="mb-2 text-xs font-semibold uppercase tracking-[0.18em] text-ink-muted">
              Reason
            </p>
            <p className="break-words text-sm leading-6 text-ink">{reason}</p>
          </div>
          <div className="rounded-2xl border border-line bg-background/70 p-4">
            <p className="mb-2 text-xs font-semibold uppercase tracking-[0.18em] text-ink-muted">
              spoken_answer
            </p>
            <p className="break-words text-sm leading-6 text-ink">{spokenAnswer}</p>
          </div>
        </div>
      </div>
      <div className="mt-4 rounded-[var(--radius-card)] border border-accent bg-accent-soft p-5">
        <p className="mb-2 text-xs font-semibold uppercase tracking-[0.18em] text-accent-strong">
          Final answer
        </p>
        <p data-testid="final-answer" className="break-words text-lg font-semibold leading-8 text-ink">
          {finalAnswer}
        </p>
      </div>
    </SectionCard>
  );
}
