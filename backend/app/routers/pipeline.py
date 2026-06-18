"""SSE Pipeline Router — end-to-end voice assistant orchestration.

Streams every stage of the pipeline as Server-Sent Events:

  pipeline_start → stage_start → stage_complete → pipeline_complete

Stages:
  stt → normalize → embed → retrieve → baseline_rerank → select_and_verbalize → tts
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from app.config import get_settings
from app.providers.normalizer import AlpacaNormalizerProvider, VLLMNormalizerProvider
from app.providers.retrieval import BGEEmbeddingProvider, PgvectorRetrievalProvider
from app.providers.selection_verbalizer import AlpacaSelectionVerbalizerProvider
from app.providers.stt.openai import OpenAIWhisperSTTProvider, STTProviderError
from app.providers.tts import OpenAITTSProvider, SupertonicTTSProvider
from app.providers.schemas import (
    LLMSelectionResult,
    PipelineCompleteEvent,
    PipelineResponse,
    PipelineStartEvent,
    ProviderMeta,
    RetrievalResult,
    StageCompleteEvent,
    StageError,
    StageErrorEvent,
    StageStartEvent,
    TimingResult,
    TTSResult,
)
from app.providers.timing import TimingContext, generate_request_id

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/pipeline", tags=["pipeline"])

_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


def _sse_event(event: str, data: Any) -> str:
    """Serialise data as an SSE event, ``{event}: {data}``."""
    if hasattr(data, "model_dump"):
        payload = data.model_dump()
    else:
        payload = data
    encoded = json.dumps(payload, default=str, ensure_ascii=False)
    return f"event: {event}\ndata: {encoded}\n\n"


async def _emit(queue: asyncio.Queue[bytes], event: str, data: Any) -> None:
    """Enqueue an SSE serialised event."""
    await queue.put(_sse_event(event, data).encode("utf-8"))


def _has_passed_threshold(candidates: list[Any], threshold: float) -> bool:
    """Check whether any candidate passes the similarity threshold."""
    for candidate in candidates:
        sim = getattr(candidate, "similarity", 0.0)
        if not isinstance(sim, (int, float)):
            sim = 0.0
        if sim >= threshold:
            return True
    return False


def _normalizer_for_provider(provider_name: str) -> Any:
    """Resolve the normalizer based on configured provider."""
    if provider_name == "vllm":
        provider = VLLMNormalizerProvider()
        if provider.configured:
            return provider
        logger.warning("vLLM normalizer is not configured; falling back to Alpaca.")
    return AlpacaNormalizerProvider.get_instance()


def _tts_provider_for_name(provider_name: str) -> Any:
    """Resolve TTS provider."""
    if provider_name == "openai":
        return OpenAITTSProvider()
    return SupertonicTTSProvider()


async def _run_pipeline(
    audio_bytes: bytes,
    mime_type: str,
    request_id: str,
    queue: asyncio.Queue[bytes],
    normalizer_provider_override: str | None = None,
    tts_provider_override: str | None = None,
) -> PipelineResponse:
    """Execute the full pipeline and stream SSE events.

    Args:
        audio_bytes: Raw audio data from browser upload.
        mime_type: MIME type of the source audio (e.g. ``"audio/webm"``).
        request_id: Unique ID for this pipeline run.
        queue: Async queue for SSE streaming.
    """
    settings = get_settings()
    timing = TimingContext()
    errors: list[StageError] = []

    # -----------------------------------------------------------------
    # 1. Speech-to-Text
    # -----------------------------------------------------------------
    await _emit(queue, "stage_start", StageStartEvent(
        request_id=request_id, stage="stt",
    ).model_dump())

    stt_provider = OpenAIWhisperSTTProvider(mime_type=mime_type)
    try:
        with timing.stage("stt"):
            stt_result = await stt_provider.process(audio_bytes)
    except STTProviderError as exc:
        logger.error("STT failed: %s", exc)
        await _emit(queue, "stage_error", StageErrorEvent(
            request_id=request_id, stage="stt",
            message=exc.code, detail=str(exc), recoverable=False,
        ).model_dump())
        errors.append(StageError(stage="stt", message=exc.code, detail=str(exc)))
        # STT failure is fatal — use empty transcript, stop pipeline
        return PipelineResponse(
            request_id=request_id,
            transcript="",
            answer=settings.fallback_answer,
            spoken_answer=settings.fallback_answer,
            errors=errors,
            timing=TimingResult(**timing.to_dict()),
        )
    except Exception as exc:
        logger.exception("Unexpected STT error")
        await _emit(queue, "stage_error", StageErrorEvent(
            request_id=request_id, stage="stt",
            message="STT_API_ERROR", detail=str(exc), recoverable=False,
        ).model_dump())
        errors.append(StageError(stage="stt", message="STT_UNEXPECTED", detail=str(exc)))
        return PipelineResponse(
            request_id=request_id, transcript="",
            answer=settings.fallback_answer, spoken_answer=settings.fallback_answer,
            errors=errors, timing=TimingResult(**timing.to_dict()),
        )

    transcript = str(stt_result.get("transcript", ""))
    await _emit(queue, "stage_complete", StageCompleteEvent(
        request_id=request_id, stage="stt",
        data={"transcript": transcript, "language": stt_result.get("language")},
    ).model_dump())

    # -----------------------------------------------------------------
    # 2. Normalization
    # -----------------------------------------------------------------
    await _emit(queue, "stage_start", StageStartEvent(
        request_id=request_id, stage="normalize",
    ).model_dump())

    normalized_query: str = transcript
    normalizer_meta: ProviderMeta | None = None
    normalizer_provider_name = normalizer_provider_override or settings.normalizer_provider

    try:
        normalizer = _normalizer_for_provider(normalizer_provider_name)
        with timing.stage("normalization"):
            normalizer_start = time.perf_counter()
            normalized_query = await normalizer.process(transcript)
            normalizer_latency = (time.perf_counter() - normalizer_start) * 1000

        normalizer_display = (
            "vllm" if isinstance(normalizer, VLLMNormalizerProvider) else "alpaca"
        )
        normalizer_meta = ProviderMeta(
            provider=normalizer_display, latency_ms=round(normalizer_latency, 2),
        )

        await _emit(queue, "stage_complete", StageCompleteEvent(
            request_id=request_id, stage="normalize",
            data={"normalized_query": normalized_query},
        ).model_dump())

    except Exception as exc:
        logger.warning("Normalization failed, falling back to raw transcript: %s", exc)
        errors.append(StageError(stage="normalize", message=str(exc)))
        await _emit(queue, "stage_error", StageErrorEvent(
            request_id=request_id, stage="normalize",
            message=str(exc), recoverable=True,
        ).model_dump())
        # Fallback to raw transcript — proceed with pipeline
        normalized_query = transcript

    # -----------------------------------------------------------------
    # 3. Embedding
    # -----------------------------------------------------------------
    await _emit(queue, "stage_start", StageStartEvent(
        request_id=request_id, stage="embed",
    ).model_dump())

    embedding_provider = BGEEmbeddingProvider()
    try:
        with timing.stage("embedding"):
            embedding = await embedding_provider.process(normalized_query)
    except Exception as exc:
        logger.exception("Embedding failed")
        await _emit(queue, "stage_error", StageErrorEvent(
            request_id=request_id, stage="embed",
            message=str(exc), detail=str(exc), recoverable=False,
        ).model_dump())
        errors.append(StageError(stage="embed", message="EMBED_ERROR", detail=str(exc)))
        return PipelineResponse(
            request_id=request_id, transcript=transcript,
            normalized_query=normalized_query,
            answer=settings.fallback_answer, spoken_answer=settings.fallback_answer,
            errors=errors, timing=TimingResult(**timing.to_dict()),
        )

    await _emit(queue, "stage_complete", StageCompleteEvent(
        request_id=request_id, stage="embed",
        data={"embedding_dim": len(embedding)},
    ).model_dump())

    # -----------------------------------------------------------------
    # 4. Retrieval + Baseline Rerank
    # -----------------------------------------------------------------
    await _emit(queue, "stage_start", StageStartEvent(
        request_id=request_id, stage="retrieve",
    ).model_dump())

    retrieval_result: RetrievalResult
    retriever = PgvectorRetrievalProvider()
    try:
        with timing.stage("retrieval"):
            retrieval_result = await retriever.process(
                embedding=embedding, query=normalized_query,
            )
    except Exception as exc:
        logger.exception("Retrieval failed")
        await _emit(queue, "stage_error", StageErrorEvent(
            request_id=request_id, stage="retrieve",
            message=str(exc), detail=str(exc), recoverable=False,
        ).model_dump())
        errors.append(StageError(stage="retrieve", message="RETRIEVE_ERROR", detail=str(exc)))
        return PipelineResponse(
            request_id=request_id, transcript=transcript,
            normalized_query=normalized_query,
            answer=settings.fallback_answer, spoken_answer=settings.fallback_answer,
            errors=errors, timing=TimingResult(**timing.to_dict()),
        )

    await _emit(queue, "stage_complete", StageCompleteEvent(
        request_id=request_id, stage="retrieve",
        data={"candidates": [c.model_dump() for c in retrieval_result.candidates],
              "answered": retrieval_result.answered},
    ).model_dump())

    # -----------------------------------------------------------------
    # 5. Threshold Gate + 6. LLM Selection & Verbalization
    # -----------------------------------------------------------------
    llm_selection: LLMSelectionResult | None = None
    answer_text: str = settings.fallback_answer
    spoken_answer_text: str = settings.fallback_answer

    threshold = settings.retrieval_similarity_threshold
    candidates_list = retrieval_result.candidates

    await _emit(queue, "stage_start", StageStartEvent(
        request_id=request_id, stage="baseline_rerank",
    ).model_dump())

    if not retrieval_result.answered or not _has_passed_threshold(candidates_list, threshold):
        logger.info("Threshold gate: no candidate passes similarity threshold (%.2f). Using fallback.", threshold)
        # Skipping LLM selection — all candidates below threshold
        await _emit(queue, "stage_complete", StageCompleteEvent(
            request_id=request_id, stage="baseline_rerank",
            data={"threshold_gate": "SKIPPED", "reason": "all_below_threshold"},
        ).model_dump())
        await _emit(queue, "stage_start", StageStartEvent(
            request_id=request_id, stage="select_and_verbalize",
        ).model_dump())
        await _emit(queue, "stage_complete", StageCompleteEvent(
            request_id=request_id, stage="select_and_verbalize",
            data={"skipped": True, "reason": "threshold_gate"},
        ).model_dump())
    else:
        # Baseline rerank done (already in retrieval_result.baseline_rerank_selected)
        baseline = retrieval_result.baseline_rerank_selected
        await _emit(queue, "stage_complete", StageCompleteEvent(
            request_id=request_id, stage="baseline_rerank",
            data={"selected": baseline.model_dump() if baseline else None},
        ).model_dump())

        # LLM Selection & Verbalization
        await _emit(queue, "stage_start", StageStartEvent(
            request_id=request_id, stage="select_and_verbalize",
        ).model_dump())

        selection_provider = AlpacaSelectionVerbalizerProvider()
        try:
            with timing.stage("llm_selection"):
                selection_data = await selection_provider.process(
                    query=normalized_query,
                    candidates=[
                        {
                            "question": c.question,
                            "answer": c.answer,
                            "similarity": c.similarity,
                            "rerank_score": c.rerank_score,
                        }
                        for c in candidates_list
                    ],
                    raw_transcript=transcript,
                )

            selected_rank = selection_data.get("selected_rank")
            selected_question = ""
            if isinstance(selected_rank, int) and 1 <= selected_rank <= len(candidates_list):
                selected_question = candidates_list[selected_rank - 1].question

            llm_selection = LLMSelectionResult(
                provider=selection_data.get("provider", "alpaca"),
                selected_rank=selected_rank if isinstance(selected_rank, int) else None,
                selected_question=selected_question,
                selected_answer=selection_data.get("selected_answer", ""),
                reason=selection_data.get("reason", ""),
                latency_ms=selection_data.get("latency_ms", 0.0),
                fallback_used=selection_data.get("fallback_used", False),
                refused=selection_data.get("refused", False),
                refusal_reason=selection_data.get("refusal_reason", ""),
            )

            if not llm_selection.refused and isinstance(selected_rank, int) and 1 <= selected_rank <= len(candidates_list):
                # Never trust the LLM to invent or mutate answer content: selected_answer
                # is always copied from the selected retrieved candidate.
                selected_candidate = candidates_list[selected_rank - 1]
                answer_text = selected_candidate.answer
                spoken_answer_text = selection_data.get("spoken_answer", "") or answer_text
                llm_selection.selected_question = selected_candidate.question
                llm_selection.selected_answer = selected_candidate.answer
            else:
                # LLM refused — use the configured fallback answer, never a baseline answer.
                logger.info("LLM refused selection: %s", llm_selection.refusal_reason)
                answer_text = settings.fallback_answer
                spoken_answer_text = settings.fallback_answer

            await _emit(queue, "stage_complete", StageCompleteEvent(
                request_id=request_id, stage="select_and_verbalize",
                data=llm_selection.model_dump(),
            ).model_dump())

        except Exception as exc:
            logger.warning("LLM Selection+Verbalization failed: %s", exc)
            errors.append(StageError(stage="select_and_verbalize", message=str(exc)))
            await _emit(queue, "stage_error", StageErrorEvent(
                request_id=request_id, stage="select_and_verbalize",
                message=str(exc), recoverable=True,
            ).model_dump())
            # Fallback to baseline selected answer
            if baseline:
                answer_text = baseline.answer
                spoken_answer_text = baseline.answer
            # else stays with fallback_answer

            llm_selection = LLMSelectionResult(
                provider="alpaca",
                selected_rank=None,
                fallback_used=True,
                refused=False,
                reason=f"Selection provider error: {exc}",
            )

    # -----------------------------------------------------------------
    # 7. TTS
    # -----------------------------------------------------------------
    tts_result: TTSResult | None = None

    await _emit(queue, "stage_start", StageStartEvent(
        request_id=request_id, stage="tts",
    ).model_dump())

    try:
        tts_provider_name = tts_provider_override or settings.tts_provider
        tts_provider = _tts_provider_for_name(tts_provider_name)
        with timing.stage("tts"):
            tts_data = await tts_provider.process(
                text=spoken_answer_text, request_id=request_id,
            )
        tts_result = TTSResult(
            provider=tts_data.get("provider", tts_provider_name),
            fallback_used=tts_data.get("fallback_used", False),
            audio_url=tts_data.get("audio_url", ""),
            latency_ms=tts_data.get("latency_ms", 0.0),
        )
        await _emit(queue, "stage_complete", StageCompleteEvent(
            request_id=request_id, stage="tts",
            data=tts_result.model_dump(),
        ).model_dump())
    except Exception as exc:
        logger.warning("TTS failed — returning text-only answer: %s", exc)
        errors.append(StageError(stage="tts", message=str(exc)))
        await _emit(queue, "stage_error", StageErrorEvent(
            request_id=request_id, stage="tts",
            message=str(exc), recoverable=True,
        ).model_dump())
        tts_result = TTSResult(
            provider=tts_provider_override or settings.tts_provider,
            fallback_used=True,
            audio_url="",
            latency_ms=0.0,
        )

    # -----------------------------------------------------------------
    # Build complete response
    # -----------------------------------------------------------------
    timing_dict = timing.to_dict()
    pipeline_response = PipelineResponse(
        request_id=request_id,
        transcript=transcript,
        normalized_query=normalized_query,
        normalizer=normalizer_meta,
        retrieval=retrieval_result,
        answer=answer_text,
        spoken_answer=spoken_answer_text,
        llm_selection=llm_selection,
        tts=tts_result,
        timing=TimingResult(
            stt_ms=timing_dict.get("stt_ms", 0.0),
            normalization_ms=timing_dict.get("normalization_ms", 0.0),
            embedding_ms=timing_dict.get("embedding_ms", 0.0),
            retrieval_ms=timing_dict.get("retrieval_ms", 0.0),
            llm_selection_ms=timing_dict.get("llm_selection_ms", 0.0),
            tts_ms=timing_dict.get("tts_ms", 0.0),
            total_ms=timing.total_ms,
        ),
        errors=errors,
    )

    return pipeline_response


@router.post("/audio-query/stream")
async def audio_query_stream(
    audio: UploadFile = File(...),
    mime_type: str = Form(default="audio/webm"),
    normalizer_provider: str | None = Form(default=None),
    tts_provider: str | None = Form(default=None),
) -> StreamingResponse:
    """Accept browser-recorded audio and stream the full pipeline via SSE.

    Pipeline stages (in order):

    1. STT — Whisper transcription
    2. Normalize — informal-to-formal rewrite (falls back to transcript on failure)
    3. Embed — BGE-M3 dense vector encoding
    4. Retrieve — pgvector + hybrid keyword-overlap rerank + threshold gate
    5. Select + Verbalize — LLM chooses best candidate and rephrases for speech
    6. TTS — Supertonic-3 (or OpenAI fallback) synthesis

    SSE events emitted:

    - ``pipeline_start`` — request begins
    - ``stage_start`` — a stage is about to execute
    - ``stage_complete`` — a stage finished successfully (with partial data)
    - ``stage_error`` — a stage encountered an error
    - ``pipeline_complete`` — full ``PipelineResponse`` payload

    Returns:
        An SSE stream (``text/event-stream``) of pipeline progress events.
    """
    request_id = generate_request_id()
    settings = get_settings()

    audio_bytes = await audio.read()

    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio upload")

    max_bytes = settings.max_audio_upload_mb * 1024 * 1024
    if len(audio_bytes) > max_bytes:
        raise HTTPException(status_code=400, detail=f"Audio exceeds {settings.max_audio_upload_mb} MB limit")

    queue: asyncio.Queue[bytes] = asyncio.Queue()

    async def event_generator():
        """Yield SSE events from the queue."""
        yield _sse_event("pipeline_start", PipelineStartEvent(
            request_id=request_id,
        ).model_dump()).encode("utf-8")

        pipeline_done = asyncio.Event()
        final_result: PipelineResponse | None = None

        async def runner():
            nonlocal final_result
            try:
                final_result = await _run_pipeline(
                    audio_bytes=audio_bytes,
                    mime_type=mime_type,
                    request_id=request_id,
                    queue=queue,
                    normalizer_provider_override=normalizer_provider,
                    tts_provider_override=tts_provider,
                )
                await _emit(queue, "pipeline_complete", PipelineCompleteEvent(
                    request_id=request_id, timestamp=time.time(),
                    response=final_result,
                ).model_dump())
            except Exception as exc:
                logger.exception("Pipeline runner crashed unexpectedly")
                final_result = PipelineResponse(
                    request_id=request_id,
                    answer=settings.fallback_answer,
                    spoken_answer=settings.fallback_answer,
                    errors=[StageError(stage="pipeline", message="INTERNAL_ERROR", detail=str(exc))],
                )
                await _emit(queue, "pipeline_complete", PipelineCompleteEvent(
                    request_id=request_id, timestamp=time.time(),
                    response=final_result,
                ).model_dump())
            finally:
                pipeline_done.set()

        asyncio.create_task(runner())

        # Consume from the queue until pipeline is done and queue is empty
        while True:
            if pipeline_done.is_set() and queue.empty():
                break
            try:
                item = await asyncio.wait_for(queue.get(), timeout=0.05)
                yield item
            except asyncio.TimeoutError:
                continue

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )
