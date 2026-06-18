export type StageStatus = "pending" | "active" | "complete" | "error";

export type PipelineStageData = {
  id: string;
  name: string;
  description: string;
  status: StageStatus;
  latencyMs?: number;
  testId: string;
  detail: string;
};

export type RetrievalCandidate = {
  rank: number;
  question: string;
  answer: string;
  similarity: number;
  keyword_score: number;
  rerank_score: number;
};
