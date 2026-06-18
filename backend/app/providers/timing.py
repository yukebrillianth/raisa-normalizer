"""Per-stage timing and error collection utilities for the pipeline."""

from __future__ import annotations

import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field


def generate_request_id() -> str:
    """Generate a unique request identifier (UUID4 without dashes)."""
    return uuid.uuid4().hex


@dataclass
class StageTiming:
    """Timing result for a single pipeline stage."""

    stage: str
    latency_ms: float = 0.0
    started_at: float | None = None
    finished_at: float | None = None


@dataclass
class TimingContext:
    """Manages per-stage latency tracking and error collection across the pipeline.

    Usage::

        tc = TimingContext()
        with tc.stage("stt"):
            result = await stt_provider.process(audio)
        # tc.stages["stt"].latency_ms is now populated

        # At the end, export to PipelineResponse:
        timing_dict = tc.to_dict()
        errors = tc.errors
    """

    stages: dict[str, StageTiming] = field(default_factory=dict)
    errors: list[dict[str, object]] = field(default_factory=list)
    started_at: float = field(default_factory=time.perf_counter)

    @contextmanager
    def stage(self, name: str) -> Iterator[StageTiming]:
        """Context manager that times a pipeline stage.

        Records latency in milliseconds using ``time.perf_counter()``.
        Preserves the existing timing entry if an error occurs mid-stage
        (partial results are kept, and the error is appended to
        ``self.errors``).

        Yields:
            StageTiming: the timing tracker for this stage.
        """
        st = StageTiming(stage=name, started_at=time.perf_counter())
        self.stages[name] = st
        try:
            yield st
        except Exception:
            # Record partial timing before propagating
            st.finished_at = time.perf_counter()
            st.latency_ms = (st.finished_at - (st.started_at or st.finished_at)) * 1000
            raise
        finally:
            if st.finished_at is None:
                st.finished_at = time.perf_counter()
            st.latency_ms = (st.finished_at - (st.started_at or st.finished_at)) * 1000

    def record_error(self, stage: str, message: str, detail: str = "") -> None:
        """Append an error entry for a specific stage.

        Args:
            stage: Pipeline stage name (e.g. ``"stt"``, ``"retrieval"``).
            message: Human-readable error summary.
            detail: Optional traceback or extra debug information.
        """
        self.errors.append({
            "stage": stage,
            "message": message,
            "detail": detail,
        })

    def to_dict(self) -> dict[str, float]:
        """Export timing as a flat dict matching ``PipelineResponse.timing``.

        Keys: ``stt_ms``, ``normalization_ms``, ``embedding_ms``,
        ``retrieval_ms``, ``llm_selection_ms``, ``tts_ms``.

        Missing stages default to ``0.0``.
        """
        mapping: dict[str, str] = {
            "stt": "stt_ms",
            "normalization": "normalization_ms",
            "embedding": "embedding_ms",
            "retrieval": "retrieval_ms",
            "llm_selection": "llm_selection_ms",
            "tts": "tts_ms",
        }
        result: dict[str, float] = {v: 0.0 for v in mapping.values()}
        for stage_name, key in mapping.items():
            st = self.stages.get(stage_name)
            if st is not None:
                result[key] = round(st.latency_ms, 2)
        return result

    @property
    def total_ms(self) -> float:
        """Total elapsed time from context creation to now."""
        return (time.perf_counter() - self.started_at) * 1000