"use client";

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

export type PipelineError = {
  stage: string;
  message: string;
  detail?: string;
  recoverable?: boolean;
};

export type LatencyItem = {
  label: string;
  ms: number;
};
