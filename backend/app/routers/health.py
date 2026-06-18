"""Aggregate health-check router for all pipeline providers.

``GET /api/health`` returns a structured JSON report for every subsystem.
Each provider section is wrapped in try/except so a single failure never
blocks the entire health check. Optional providers (vLLM, Supertonic) are
reported as ``unavailable`` rather than causing a fatal error.
"""

from __future__ import annotations

import logging
from typing import Any

from app.config import get_settings
from app.db import get_db
from app.providers.normalizer import (AlpacaNormalizerProvider,
                                      VLLMNormalizerProvider)
from app.providers.retrieval import BGEEmbeddingProvider
from app.providers.stt.openai import OpenAIWhisperSTTProvider
from app.providers.tts import OpenAITTSProvider, SupertonicTTSProvider
from fastapi import APIRouter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/health", tags=["health"])

__version__ = "1.0.0"


async def _db_health() -> dict[str, Any]:
    """Database connectivity and schema readiness."""
    try:
        db = get_db()
        schema = db.schema
        return {
            "status": "ok" if db.connected else "unavailable",
            "connected": db.connected,
            "table_exists": schema.table_exists,
            "row_count": schema.row_count,
            "pgvector_available": schema.pgvector_available,
        }
    except Exception as exc:
        logger.warning("DB health check failed: %s", exc)
        return {"status": "error", "error": "db_health_failed"}


async def _embedding_health() -> dict[str, Any]:
    """Embedding model readiness."""
    try:
        provider = BGEEmbeddingProvider.get_instance()
        return await provider.health()
    except Exception as exc:
        logger.warning("Embedding health check failed: %s", exc)
        return {"status": "error", "error": "embedding_health_failed"}


async def _stt_health() -> dict[str, Any]:
    """STT provider readiness — API key reported as boolean only."""
    try:
        provider = OpenAIWhisperSTTProvider()
        result = await provider.health()
        # Enforce: api_key_configured is bool, never the actual key value
        result["api_key_configured"] = bool(get_settings().openai_api_key)
        return result
    except Exception as exc:
        logger.warning("STT health check failed: %s", exc)
        return {"status": "error", "error": "stt_health_failed"}


async def _normalizer_health() -> dict[str, Any]:
    """Normalizer sub-providers: Alpaca (required) and vLLM (optional)."""
    alpaca: dict[str, Any] = {}
    try:
        normalizer = AlpacaNormalizerProvider.get_instance()
        alpaca = await normalizer.health()
    except Exception as exc:
        logger.warning("Alpaca normalizer health check failed: %s", exc)
        alpaca = {"status": "error", "error": "alpaca_health_failed"}

    vllm: dict[str, Any] = {}
    try:
        provider = VLLMNormalizerProvider()
        vllm = await provider.health()
    except Exception as exc:
        # vLLM is optional — unreachable is not fatal
        logger.warning("vLLM normalizer health check failed (optional): %s", exc)
        vllm = {"provider": "vllm", "status": "unavailable", "error": "vllm_health_failed"}

    return {"alpaca": alpaca, "vllm": vllm}


async def _tts_health() -> dict[str, Any]:
    """TTS sub-providers: Supertonic (optional, local GPU) and OpenAI (fallback)."""
    supertonic: dict[str, Any] = {}
    try:
        provider = SupertonicTTSProvider.get_instance()
        supertonic = await provider.health()
    except Exception as exc:
        # Supertonic is optional — may not be installed or GPU unavailable
        logger.warning("Supertonic TTS health check failed (optional): %s", exc)
        supertonic = {"provider": "supertonic", "status": "unavailable", "error": "supertonic_health_failed"}

    openai_tts: dict[str, Any] = {}
    try:
        provider = OpenAITTSProvider.get_instance()
        result = await provider.health()
        # Enforce: api_key_configured is bool only
        result["api_key_configured"] = bool(get_settings().openai_api_key)
        openai_tts = result
    except Exception as exc:
        logger.warning("OpenAI TTS health check failed: %s", exc)
        openai_tts = {"status": "error", "error": "openai_tts_health_failed"}

    return {"supertonic": supertonic, "openai": openai_tts}


async def _admin_health() -> dict[str, Any]:
    """Admin token presence — reported as boolean, never the value."""
    settings = get_settings()
    return {"token_configured": bool(settings.admin_token and settings.admin_token.strip())}


@router.get("")
async def aggregate_health() -> dict[str, Any]:
    """Aggregate health for every pipeline subsystem.

    Returns structured JSON with per-provider status dicts. Optional
    providers (vLLM, Supertonic) may report ``unavailable`` without
    causing the overall health check to fail. Secrets are never exposed.
    """
    return {
        "version": __version__,
        "db": await _db_health(),
        "embedding": await _embedding_health(),
        "stt": await _stt_health(),
        "normalizers": await _normalizer_health(),
        "tts": await _tts_health(),
        "admin": await _admin_health(),
    }
