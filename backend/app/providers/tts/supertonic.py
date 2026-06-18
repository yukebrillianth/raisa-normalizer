"""Supertonic-3 text-to-speech provider.

Supertonic-3 is a 99M-parameter ONNX-based Indonesian TTS model.
It runs locally via the ``supertonic`` Python SDK and synthesises
speech using a diffusion pipeline with configurable step count.
"""

from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from app.config import get_settings
from app.providers.base import TTSProvider

logger = logging.getLogger(__name__)

AUDIO_DIR = Path(__file__).resolve().parent.parent.parent.parent / "audio"


class TTSSynthesisError(RuntimeError):
    """Raised when Supertonic-3 synthesis fails."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


class SupertonicTTSProvider(TTSProvider):
    """Indonesian TTS using the Supertonic-3 ONNX model."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self._tts: Any | None = None
        self._voice_style: Any | None = None

    def _ensure_loaded(self) -> None:
        """Lazy-initialise the TTS model and voice style."""
        if self._tts is not None:
            return

        try:
            from supertonic import TTS
        except ImportError as exc:
            raise TTSSynthesisError(
                "TTS_MISSING_SDK",
                "The 'supertonic' package is not installed. Run: pip install supertonic",
            ) from exc

        logger.info("Loading Supertonic-3 model (auto_download=True)...")
        self._tts = TTS(auto_download=True)
        self._voice_style = self._tts.get_voice_style(
            voice_name=self.settings.supertonic_voice,
        )
        logger.info("Supertonic-3 model loaded with voice '%s'.", self.settings.supertonic_voice)

    async def process(self, text: str, request_id: str | None = None) -> dict[str, Any]:
        """Synthesise *text* to a WAV file and return metadata.

        Parameters
        ----------
        text:
            The text to speak (should be normalised Indonesian).
        request_id:
            Optional identifier used as the filename stem.  A random UUID
            is generated when not provided.
        """
        if not text or not text.strip():
            raise TTSSynthesisError("TTS_EMPTY_TEXT", "Nothing to synthesise")

        started_at = time.perf_counter()
        self._ensure_loaded()

        req_id = request_id or uuid.uuid4().hex

        try:
            result = self._tts.synthesize(
                text=text.strip(),
                voice_style=self._voice_style,
                total_steps=self.settings.supertonic_total_steps,
            )
        except Exception as exc:
            raise TTSSynthesisError(
                "TTS_SYNTHESIS_ERROR",
                f"Supertonic-3 synthesis failed: {exc}",
            ) from exc

        wav = result if isinstance(result, np.ndarray) else result[0]
        audio_path = AUDIO_DIR / f"{req_id}.wav"
        AUDIO_DIR.mkdir(parents=True, exist_ok=True)
        self._tts.save_audio(wav, str(audio_path))

        latency = round((time.perf_counter() - started_at) * 1000, 2)

        return {
            "audio_url": f"/api/audio/{audio_path.name}",
            "provider": "supertonic",
            "latency_ms": latency,
            "fallback_used": False,
        }

    async def health(self) -> dict[str, Any]:
        """Report provider readiness."""
        return {
            "provider": "supertonic",
            "voice": self.settings.supertonic_voice,
            "total_steps": self.settings.supertonic_total_steps,
            "model_loaded": self._tts is not None,
        }
