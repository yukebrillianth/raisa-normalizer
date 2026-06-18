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
      <div className="overflow-x-auto rounded-2xl border border-line bg-background/70">
        <table className="min-w-[920px] w-full border-collapse text-left text-sm">
          <thead className="bg-surface-strong text-xs uppercase tracking-[0.18em] text-ink-muted">
            <tr>
              <th className="p-4 font-semibold">Rank</th>
              <th className="p-4 font-semibold">Pertanyaan</th>
              <th className="p-4 font-semibold">Jawaban</th>
              <th className="p-4 font-semibold">Similarity</th>
              <th className="p-4 font-semibold">Keyword</th>
              <th className="p-4 font-semibold">Rerank</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-line">
            {candidates.map((candidate) => (
              <tr key={candidate.rank} className="align-top hover:bg-surface/70">
                <td className="p-4 font-mono font-semibold text-accent-strong">
                  #{candidate.rank}
                </td>
                <td className="max-w-[18rem] break-words p-4 leading-6 text-ink">
                  {candidate.question}
                </td>
                <td className="max-w-[24rem] break-words p-4 leading-6 text-ink-muted">
                  {candidate.answer}
                </td>
                <td className="p-4 font-mono text-ink">{candidate.similarity.toFixed(3)}</td>
                <td className="p-4 font-mono text-ink">{candidate.keyword_score.toFixed(3)}</td>
                <td className="p-4 font-mono font-semibold text-success">
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
