"""OpenAI Whisper speech-to-text provider."""

from __future__ import annotations

import io
import time
from typing import Any

import openai

from app.config import get_settings
from app.providers.base import STTProvider


class STTProviderError(RuntimeError):
    """Provider-level STT error with a stable machine-readable code."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


class OpenAIWhisperSTTProvider(STTProvider):
    """Transcribe uploaded browser audio using OpenAI Whisper."""

    SUPPORTED_MIME_TYPES = frozenset(
        {
            "audio/webm",
            "audio/webm;codecs=opus",
            "audio/ogg",
            "audio/ogg;codecs=opus",
            "audio/mpeg",
            "audio/mp3",
            "audio/mp4",
            "audio/m4a",
            "audio/wav",
            "audio/x-wav",
            "audio/wave",
        }
    )

    def __init__(self, mime_type: str = "audio/webm") -> None:
        self.settings = get_settings()
        self.mime_type = mime_type
        self.model = self.settings.openai_whisper_model
        self.max_audio_bytes = self.settings.max_audio_upload_mb * 1024 * 1024

    async def process(self, audio: bytes) -> dict[str, str | float | None]:
        """Validate and transcribe audio bytes."""
        started_at = time.perf_counter()
        self._validate_audio(audio)

        if not self.settings.openai_api_key:
            raise STTProviderError("STT_NO_KEY", "OpenAI API key is not configured")

        audio_file = io.BytesIO(audio)
        audio_file.name = self._filename_for_mime_type(self.mime_type)

        try:
            response = await self._create_transcription(audio_file)
        except STTProviderError:
            raise
        except Exception as exc:  # noqa: BLE001 - normalise third-party failures.
            raise STTProviderError("STT_API_ERROR", "OpenAI Whisper transcription failed") from exc

        transcript = self._extract_text(response)
        if not transcript:
            raise STTProviderError("STT_EMPTY_TRANSCRIPT", "OpenAI Whisper returned an empty transcript")

        return {
            "transcript": transcript,
            "language": self._extract_language(response),
            "latency_ms": round((time.perf_counter() - started_at) * 1000, 2),
        }

    async def health(self) -> dict[str, Any]:
        """Report provider readiness without exposing secrets."""
        return {
            "provider": "openai_whisper",
            "model": self.model,
            "api_key_configured": bool(self.settings.openai_api_key),
            "max_audio_upload_mb": self.settings.max_audio_upload_mb,
            "supported_mime_types": sorted(self.SUPPORTED_MIME_TYPES),
        }

    def _validate_audio(self, audio: bytes) -> None:
        if self._normalised_mime_type(self.mime_type) not in self.SUPPORTED_MIME_TYPES:
            raise STTProviderError("STT_INVALID_AUDIO", f"Unsupported audio MIME type: {self.mime_type}")
        if not audio:
            raise STTProviderError("STT_INVALID_AUDIO", "Audio upload is empty")
        if len(audio) > self.max_audio_bytes:
            raise STTProviderError(
                "STT_INVALID_AUDIO",
                f"Audio upload exceeds {self.settings.max_audio_upload_mb} MB limit",
            )

    async def _create_transcription(self, audio_file: io.BytesIO) -> Any:
        client = openai.AsyncOpenAI(api_key=self.settings.openai_api_key)  # pyright: ignore[reportAttributeAccessIssue]
        return await client.audio.transcriptions.create(
            model=self.model,
            file=audio_file,
            response_format="verbose_json",
        )

    @staticmethod
    def _extract_text(response: Any) -> str:
        if isinstance(response, dict):
            text = response.get("text", "")
        else:
            text = getattr(response, "text", "")
        return str(text).strip()

    @staticmethod
    def _extract_language(response: Any) -> str | None:
        if isinstance(response, dict):
            language = response.get("language")
        else:
            language = getattr(response, "language", None)
        return str(language).strip() if language else None

    @classmethod
    def _filename_for_mime_type(cls, mime_type: str) -> str:
        normalised = cls._normalised_mime_type(mime_type)
        extension_by_mime_type = {
            "audio/webm": "webm",
            "audio/webm;codecs=opus": "webm",
            "audio/ogg": "ogg",
            "audio/ogg;codecs=opus": "ogg",
            "audio/mpeg": "mp3",
            "audio/mp3": "mp3",
            "audio/mp4": "mp4",
            "audio/m4a": "m4a",
            "audio/wav": "wav",
            "audio/x-wav": "wav",
            "audio/wave": "wav",
        }
        return f"upload.{extension_by_mime_type.get(normalised, 'webm')}"

    @staticmethod
    def _normalised_mime_type(mime_type: str) -> str:
        return mime_type.strip().lower()
