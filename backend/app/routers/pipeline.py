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
import os
import subprocess
import tempfile
import time
from typing import Any

from app.config import get_settings
from app.providers.normalizer import (AlpacaNormalizerProvider,
                                      VLLMNormalizerProvider)
from app.providers.retrieval import (BGEEmbeddingProvider,
                                     PgvectorRetrievalProvider)
from app.providers.schemas import (LLMSelectionResult, PipelineCompleteEvent,
                                   PipelineResponse, PipelineStartEvent,
                                   ProviderMeta, RetrievalResult,
                                   StageCompleteEvent, StageError,
                                   StageErrorEvent, StageStartEvent,
                                   TimingResult, TTSResult)
from app.providers.selection_verbalizer import \
    OpenAISelectionVerbalizerProvider
from app.providers.stt.openai import OpenAIWhisperSTTProvider, STTProviderError
from app.providers.timing import TimingContext, generate_request_id
from app.providers.tts import OpenAITTSProvider, SupertonicTTSProvider
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/pipeline", tags=["pipeline"])

ALLOWED_NORMALIZER_PROVIDERS = {"alpaca", "vllm"}
ALLOWED_TTS_PROVIDERS = {"supertonic", "openai"}
PIPELINE_STAGES = [
    "stt",
    "normalize",
    "embed",
    "retrieve",
    "baseline_rerank",
    "select_and_verbalize",
    "tts",
]

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


def _safe_error_detail(exc: Exception) -> str:
    """Return a useful but non-sensitive error string for SSE/debug payloads."""
    text = str(exc).replace("\n", " ").strip()
    if len(text) > 300:
        text = f"{text[:300]}..."
    return text or exc.__class__.__name__


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


def _validate_provider_options(normalizer_provider: str | None, tts_provider: str | None) -> tuple[str | None, str | None]:
    """Validate optional client-selected providers."""
    if normalizer_provider is not None and normalizer_provider not in ALLOWED_NORMALIZER_PROVIDERS:
        raise HTTPException(status_code=422, detail="Unsupported normalizer_provider")
    if tts_provider is not None and tts_provider not in ALLOWED_TTS_PROVIDERS:
        raise HTTPException(status_code=422, detail="Unsupported tts_provider")
    return normalizer_provider, tts_provider


def _tts_provider_for_name(provider_name: str) -> Any:
    """Resolve TTS provider."""
    if provider_name == "openai":
        return OpenAITTSProvider.get_instance()
    return SupertonicTTSProvider.get_instance()


def _validated_spoken_answer(spoken_answer: str, selected_answer: str) -> str:
    """Use plain-text verbalizer output if it is non-empty."""
    candidate = spoken_answer.strip()
    if not candidate:
        return selected_answer
    return candidate


async def _emit_stage_complete_for_error(
    queue: asyncio.Queue[bytes], request_id: str, stage: str, message: str
) -> None:
    await _emit(queue, "stage_complete", StageCompleteEvent(
        request_id=request_id,
        stage=stage,
        data={"failed": True, "reason": message},
    ).model_dump())


async def _read_limited_upload(upload: UploadFile, max_bytes: int) -> bytes:
    """Read an UploadFile in chunks, aborting when max_bytes is exceeded."""
    chunks: list[bytes] = []
    total = 0
    chunk_size = 1024 * 1024
    while True:
        chunk = await upload.read(chunk_size)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(status_code=400, detail="Audio exceeds configured upload limit")
        chunks.append(chunk)
    return b"".join(chunks)


def _check_audio_duration(audio_bytes: bytes, max_seconds: int) -> None:
    """Validate audio duration using ffprobe, raising 400 if exceeded."""
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                tmp_path,
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            duration = float(result.stdout.strip())
            if duration > max_seconds:
                raise HTTPException(
                    status_code=400,
                    detail=f"Audio duration {duration:.1f}s exceeds maximum allowed {max_seconds}s",
                )
    except subprocess.TimeoutExpired:
        logger.warning("ffprobe timed out checking audio duration")
    except ValueError:
        logger.warning("Could not parse audio duration from ffprobe output")
    except FileNotFoundError:
        logger.warning("ffprobe not found; skipping audio duration check")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


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
        detail = _safe_error_detail(exc)
        await _emit(queue, "stage_error", StageErrorEvent(
            request_id=request_id, stage="stt",
            message=exc.code, detail=detail, recoverable=False,
        ).model_dump())
        errors.append(StageError(stage="stt", message=exc.code, detail=detail))
        await _emit_stage_complete_for_error(queue, request_id, "stt", exc.code)
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
        detail = _safe_error_detail(exc)
        await _emit(queue, "stage_error", StageErrorEvent(
            request_id=request_id, stage="stt",
            message="STT_API_ERROR", detail=detail, recoverable=False,
        ).model_dump())
        errors.append(StageError(stage="stt", message="STT_UNEXPECTED", detail=detail))
        await _emit_stage_complete_for_error(queue, request_id, "stt", "STT_API_ERROR")
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
        detail = _safe_error_detail(exc)
        errors.append(StageError(stage="normalize", message="NORMALIZE_ERROR", detail=detail))
        await _emit(queue, "stage_error", StageErrorEvent(
            request_id=request_id, stage="normalize",
            message="NORMALIZE_ERROR", detail=detail, recoverable=True,
        ).model_dump())
        # Fallback to raw transcript — proceed with pipeline
        normalized_query = transcript

    # -----------------------------------------------------------------
    # 3. Embedding
    # -----------------------------------------------------------------
    await _emit(queue, "stage_start", StageStartEvent(
        request_id=request_id, stage="embed",
    ).model_dump())

    embedding_provider = BGEEmbeddingProvider.get_instance()
    try:
        with timing.stage("embedding"):
            embedding = await embedding_provider.process(normalized_query)
    except Exception as exc:
        logger.exception("Embedding failed")
        detail = _safe_error_detail(exc)
        await _emit(queue, "stage_error", StageErrorEvent(
            request_id=request_id, stage="embed",
            message="EMBED_ERROR", detail=detail, recoverable=False,
        ).model_dump())
        errors.append(StageError(stage="embed", message="EMBED_ERROR", detail=detail))
        await _emit_stage_complete_for_error(queue, request_id, "embed", "EMBED_ERROR")
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
        detail = _safe_error_detail(exc)
        await _emit(queue, "stage_error", StageErrorEvent(
            request_id=request_id, stage="retrieve",
            message="RETRIEVE_ERROR", detail=detail, recoverable=False,
        ).model_dump())
        errors.append(StageError(stage="retrieve", message="RETRIEVE_ERROR", detail=detail))
        await _emit_stage_complete_for_error(queue, request_id, "retrieve", "RETRIEVE_ERROR")
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

    if not retrieval_result.answered:
        logger.info("Threshold gate: RRF top-1 candidate is below similarity threshold (%.2f). Using fallback.", threshold)
        # Skipping LLM selection — selected RRF top-1 is below threshold
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

        selection_provider = OpenAISelectionVerbalizerProvider()
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
                provider=selection_data.get("provider", "openai"),
                selected_rank=selected_rank if isinstance(selected_rank, int) else None,
                selected_question=selected_question,
                selected_answer=selection_data.get("selected_answer", ""),
                spoken_answer=selection_data.get("spoken_answer", ""),
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
                spoken_answer_text = _validated_spoken_answer(selection_data.get("spoken_answer", ""), answer_text)
                llm_selection.selected_question = selected_candidate.question
                llm_selection.selected_answer = selected_candidate.answer
                llm_selection.spoken_answer = spoken_answer_text
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
            detail = _safe_error_detail(exc)
            errors.append(StageError(stage="select_and_verbalize", message="SELECT_VERBALIZE_ERROR", detail=detail))
            await _emit(queue, "stage_error", StageErrorEvent(
                request_id=request_id, stage="select_and_verbalize",
                message="SELECT_VERBALIZE_ERROR", detail=detail, recoverable=True,
            ).model_dump())
            # Fallback to baseline selected answer
            if baseline:
                answer_text = baseline.answer
                spoken_answer_text = baseline.answer
            # else stays with fallback_answer

            llm_selection = LLMSelectionResult(
                provider="openai",
                selected_rank=None,
                fallback_used=True,
                refused=False,
                reason="Selection provider error",
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
            try:
                tts_data = await tts_provider.process(
                    text=spoken_answer_text, request_id=request_id,
                )
            except Exception as primary_exc:
                logger.warning("Primary TTS failed: %s. Trying fallback.", primary_exc)
                if tts_provider_name != "openai":
                    fallback_provider = OpenAITTSProvider.get_instance()
                    tts_data = await fallback_provider.process(
                        text=spoken_answer_text, request_id=request_id,
                    )
                else:
                    raise primary_exc
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
        detail = _safe_error_detail(exc)
        errors.append(StageError(stage="tts", message="TTS_ERROR", detail=detail))
        await _emit(queue, "stage_error", StageErrorEvent(
            request_id=request_id, stage="tts",
            message="TTS_ERROR", detail=detail, recoverable=True,
        ).model_dump())
        tts_result = TTSResult(
            provider=tts_provider_override or settings.tts_provider,
            fallback_used=True,
            audio_url="",
            latency_ms=0.0,
        )
        await _emit_stage_complete_for_error(queue, request_id, "tts", "TTS_ERROR")

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

    normalizer_provider, tts_provider = _validate_provider_options(normalizer_provider, tts_provider)
    max_bytes = settings.max_audio_upload_mb * 1024 * 1024
    audio_bytes = await _read_limited_upload(audio, max_bytes)

    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio upload")

    _check_audio_duration(audio_bytes, settings.max_recording_seconds)

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
                    final_response=final_result,
                ).model_dump())
            except Exception as exc:
                logger.exception("Pipeline runner crashed unexpectedly")
                final_result = PipelineResponse(
                    request_id=request_id,
                    answer=settings.fallback_answer,
                    spoken_answer=settings.fallback_answer,
                    errors=[StageError(stage="pipeline", message="INTERNAL_ERROR", detail=_safe_error_detail(exc))],
                )
                await _emit(queue, "pipeline_complete", PipelineCompleteEvent(
                    request_id=request_id, timestamp=time.time(),
                    final_response=final_result,
                ).model_dump())
            finally:
                pipeline_done.set()

        runner_task = asyncio.create_task(runner())
        try:
            # Consume from the queue until pipeline is done and queue is empty
            while True:
                if pipeline_done.is_set() and queue.empty():
                    break
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=0.05)
                    yield item
                except asyncio.TimeoutError:
                    continue
        finally:
            if not runner_task.done():
                runner_task.cancel()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )
