import { SectionCard } from "@/components/SectionCard";
import type { RetrievalCandidate } from "@/components/types";

type RetrievalCandidatesProps = {
  candidates: RetrievalCandidate[];
};

export function RetrievalCandidates({ candidates }: RetrievalCandidatesProps) {
  return (
    <SectionCard
      title="Kandidat Retrieval"
      eyebrow="Vector + Keyword + Rerank"
      testId="retrieval-candidates"
    >
      <div className="scrollable-panel max-h-[34rem] overflow-auto rounded-2xl border border-line bg-background/70">
        <table className="w-full min-w-[920px] table-fixed border-collapse text-left text-sm">
          <thead className="bg-surface-strong text-xs uppercase tracking-[0.18em] text-ink-muted">
            <tr>
              <th className="w-20 p-4 font-semibold">Rank</th>
              <th className="w-64 p-4 font-semibold">Pertanyaan kandidat</th>
              <th className="w-80 p-4 font-semibold">Jawaban kandidat</th>
              <th className="w-32 p-4 font-semibold">Similarity</th>
              <th className="w-32 p-4 font-semibold">Keyword</th>
              <th className="w-32 p-4 font-semibold">Rerank</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-line">
            {candidates.map((candidate) => (
              <tr key={candidate.rank} className="align-top hover:bg-surface/70">
                <td className="p-4 font-semibold text-accent-strong">
                  #{candidate.rank}
                </td>
                <td className="break-words p-4 leading-6 text-ink">
                  {candidate.question}
                </td>
                <td className="break-words p-4 leading-6 text-ink-muted">
                  {candidate.answer}
                </td>
                <td className="p-4 text-ink">{candidate.similarity.toFixed(3)}</td>
                <td className="p-4 text-ink">{candidate.keyword_score.toFixed(3)}</td>
                <td className="p-4 font-semibold text-success">
                  {candidate.rerank_score.toFixed(3)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </SectionCard>
  );
}
