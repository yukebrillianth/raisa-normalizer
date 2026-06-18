"""OpenAI TTS fallback provider.

Uses the OpenAI ``/v1/audio/speech`` endpoint (tts-1 / tts-1-hd) to
synthesise speech when the primary Supertonic-3 provider is unavailable
or fails.
"""

from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path
from typing import Any

import openai
from app.config import get_settings
from app.providers.base import TTSProvider

logger = logging.getLogger(__name__)

AUDIO_DIR = Path(__file__).resolve().parent.parent.parent.parent / "audio"


class OpenAITTSError(RuntimeError):
    """Raised when OpenAI TTS fails."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


class OpenAITTSProvider(TTSProvider):
    """Text-to-speech using the OpenAI Audio API as a fallback."""

    _instance: OpenAITTSProvider | None = None

    @classmethod
    def get_instance(cls) -> OpenAITTSProvider:
        """Return (or create) the process-wide singleton."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        self.settings = get_settings()

    async def process(self, text: str, request_id: str | None = None) -> dict[str, Any]:
        """Synthesise *text* via OpenAI and return metadata.

        Parameters
        ----------
        text:
            The text to speak.
        request_id:
            Optional identifier used as the filename stem.
        """
        started_at = time.perf_counter()

        if not self.settings.openai_api_key:
            raise OpenAITTSError("TTS_NO_KEY", "OpenAI API key is not configured")

        if not text or not text.strip():
            raise OpenAITTSError("TTS_EMPTY_TEXT", "Nothing to synthesise")

        req_id = request_id or uuid.uuid4().hex

        try:
            openai.api_key = self.settings.openai_api_key
            response = openai.audio.speech.create(
                model=self.settings.openai_tts_model,
                voice=self.settings.openai_tts_voice,
                input=text.strip(),
            )
            audio_bytes: bytes = response.content
        except OpenAITTSError:
            raise
        except Exception as exc:
            raise OpenAITTSError(
                "TTS_API_ERROR",
                f"OpenAI TTS request failed: {exc}",
            ) from exc

        audio_path = self._save_audio(audio_bytes, req_id)
        latency = round((time.perf_counter() - started_at) * 1000, 2)

        return {
            "audio_url": f"/api/audio/{audio_path.name}",
            "provider": "openai",
            "latency_ms": latency,
            "fallback_used": True,
        }

    async def health(self) -> dict[str, Any]:
        """Report provider readiness."""
        return {
            "provider": "openai",
            "model": self.settings.openai_tts_model,
            "voice": self.settings.openai_tts_voice,
            "api_key_configured": bool(self.settings.openai_api_key),
        }

    @staticmethod
    def _save_audio(audio_bytes: bytes, request_id: str) -> Path:
        """Write *audio_bytes* to ``backend/audio/<request_id>.wav``."""
        AUDIO_DIR.mkdir(parents=True, exist_ok=True)
        dest = AUDIO_DIR / f"{request_id}.wav"
        dest.write_bytes(audio_bytes)
        logger.debug("Saved OpenAI TTS audio to %s (%d bytes)", dest, len(audio_bytes))
        return dest
