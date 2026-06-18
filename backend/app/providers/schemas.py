"""Pydantic v2 models for the pipeline response and SSE streaming events.

The schema mirrors the final-response JSON defined in the plan
(lines 207-246 of ``.sisyphus/plans/voice-assistant-thesis.md``).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ProviderMeta(BaseModel):
    """Metadata about the provider that executed a stage."""

    provider: str = Field(..., examples=["alpaca"])
    latency_ms: float = Field(..., examples=[1234.0])
    fallback_used: bool = Field(default=False)


class RetrievalCandidate(BaseModel):
    """A single candidate retrieved from the vector datastore."""

    question: str
    answer: str
    similarity: float = Field(..., ge=0.0, le=1.0, examples=[0.91])
    keyword_score: float = Field(default=0.0, ge=0.0, le=1.0)
    rerank_score: float = Field(default=0.0, ge=0.0, le=1.0, examples=[0.99])


class BaselineRerankSelected(BaseModel):
    """The top candidate selected by the baseline reranker."""

    question: str
    answer: str
    similarity: float = Field(..., ge=0.0, le=1.0)
    rerank_score: float = Field(..., ge=0.0, le=1.0)


class RetrievalResult(BaseModel):
    """Complete retrieval sub-response."""

    top_k: int = Field(default=3)
    similarity_threshold: float = Field(default=0.75)
    rerank_keyword_weight: float = Field(default=0.2)
    candidates: list[RetrievalCandidate] = Field(default_factory=list)
    baseline_rerank_selected: BaselineRerankSelected | None = None
    answered: bool = Field(default=False)


class LLMSelectionResult(BaseModel):
    """LLM-based selection / verbalization sub-response."""

    provider: str = Field(..., examples=["alpaca"])
    selected_rank: int = Field(..., examples=[1])
    selected_question: str = ""
    selected_answer: str = ""
    reason: str = ""
    latency_ms: float = Field(default=0.0)
    fallback_used: bool = Field(default=False)
    refused: bool = Field(default=False)
    refusal_reason: str = ""


class TTSResult(BaseModel):
    """Text-to-speech sub-response."""

    provider: str = Field(..., examples=["supertonic"])
    fallback_used: bool = Field(default=False)
    audio_url: str = ""
    latency_ms: float = Field(default=0.0)


class TimingResult(BaseModel):
    """Per-stage latency summary (milliseconds)."""

    stt_ms: float = 0.0
    normalization_ms: float = 0.0
    embedding_ms: float = 0.0
    retrieval_ms: float = 0.0
    llm_selection_ms: float = 0.0
    tts_ms: float = 0.0
    total_ms: float = 0.0


class StageError(BaseModel):
    """Error recorded during a specific pipeline stage."""

    stage: str = Field(..., examples=["stt"])
    message: str = Field(..., examples=["Whisper API timeout"])
    detail: str = Field(default="")


class PipelineResponse(BaseModel):
    """Top-level response emitted in the ``pipeline_complete`` SSE event.

    Includes partial results from stages that completed successfully
    (partial errors are captured in ``errors`` and do not prevent
    serialisation).
    """

    model_config = {"extra": "forbid"}

    request_id: str = Field(..., examples=["abc123def456"])
    transcript: str = ""
    normalized_query: str = ""
    normalizer: ProviderMeta | None = None
    retrieval: RetrievalResult = Field(default_factory=RetrievalResult)
    answer: str = ""
    spoken_answer: str = ""
    llm_selection: LLMSelectionResult | None = None
    tts: TTSResult | None = None
    timing: TimingResult = Field(default_factory=TimingResult)
    errors: list[StageError] = Field(default_factory=list)


class PipelineStartEvent(BaseModel):
    """Emitted when the pipeline begins processing a request."""

    event: str = Field(default="pipeline_start", frozen=True)
    request_id: str
    timestamp: float = Field(default_factory=lambda: __import__("time").time())


class StageStartEvent(BaseModel):
    """Emitted when a pipeline stage begins execution."""

    event: str = Field(default="stage_start", frozen=True)
    request_id: str
    stage: str = Field(..., examples=["stt", "normalization", "embedding"])
    timestamp: float = Field(default_factory=lambda: __import__("time").time())


class StageCompleteEvent(BaseModel):
    """Emitted when a pipeline stage completes successfully.

    The ``data`` payload carries stage-specific partial results.
    """

    event: str = Field(default="stage_complete", frozen=True)
    request_id: str
    stage: str
    timestamp: float = Field(default_factory=lambda: __import__("time").time())
    data: dict[str, Any] = Field(default_factory=dict)


class PipelineCompleteEvent(BaseModel):
    """Emitted when the entire pipeline finishes.

    The ``response`` field is a fully hydrated ``PipelineResponse``.
    """

    event: str = Field(default="pipeline_complete", frozen=True)
    request_id: str
    timestamp: float = Field(default_factory=lambda: __import__("time").time())
    response: PipelineResponse


class StageErrorEvent(BaseModel):
    """Emitted when a pipeline stage encounters an error."""

    event: str = Field(default="stage_error", frozen=True)
    request_id: str
    stage: str
    timestamp: float = Field(default_factory=lambda: __import__("time").time())
    message: str
    detail: str = ""
    recoverable: bool = Field(default=True)