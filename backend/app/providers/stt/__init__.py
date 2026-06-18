"""Speech-to-text provider implementations."""

from app.providers.stt.openai import OpenAIWhisperSTTProvider, STTProviderError

__all__ = ["OpenAIWhisperSTTProvider", "STTProviderError"]
