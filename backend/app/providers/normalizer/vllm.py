"""vLLM-based normalizer using OpenAI-compatible chat completions API."""

from __future__ import annotations

import logging
import re
from typing import Any

import httpx

from app.config import get_settings
from app.providers.base import NormalizerProvider

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "Anda adalah sistem layanan informasi Kampus Institut Teknologi Sepuluh Nopember (ITS). "
    "Ubah input kalimat Bahasa Indonesia tidak baku berikut menjadi Bahasa Indonesia baku dan formal. "
    "Jangan menambahkan informasi baru. "
    "Gunakan kalimat singkat, jelas, baku, dan langsung ke inti."
)

_CLEAN_PATTERNS = [
    (re.compile(r"<think>.*?</think>", flags=re.DOTALL), ""),
    (re.compile(r"['\"]"), ""),
    (re.compile(r"\s+"), " "),
]


def _clean_output(text: str) -> str:
    cleaned = text
    for pattern, replacement in _CLEAN_PATTERNS:
        cleaned = pattern.sub(replacement, cleaned)
    return cleaned.strip()


class VLLMNormalizerProvider(NormalizerProvider):
    """Query normalizer that calls a vLLM server via OpenAI-compatible API."""

    def __init__(self) -> None:
        settings = get_settings()
        self._base_url = settings.normalizer_vllm_base_url.rstrip("/")
        self._model = settings.normalizer_vllm_model
        self._max_tokens = settings.normalizer_max_new_tokens
        self._configured = bool(self._base_url)

    @property
    def configured(self) -> bool:
        return self._configured

    async def process(self, transcript: str) -> str:
        if not self._configured:
            raise RuntimeError("vLLM normalizer is not configured")

        url = f"{self._base_url}/v1/chat/completions"
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": transcript},
            ],
            "max_tokens": self._max_tokens,
            "temperature": 0.0,
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(f"vLLM HTTP error: {exc.response.status_code}") from exc
        except httpx.RequestError as exc:
            raise RuntimeError(f"vLLM request error: {exc}") from exc

        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError("vLLM returned empty choices")

        content = choices[0].get("message", {}).get("content", "")
        if not content:
            raise RuntimeError("vLLM returned empty content")

        return _clean_output(content)

    async def health(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "provider": "vllm",
            "configured": self._configured,
            "base_url": self._base_url,
            "model": self._model,
        }

        if not self._configured:
            result["reachable"] = False
            result["status"] = "unavailable"
            return result

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self._base_url}/v1/models")
                result["reachable"] = response.status_code == 200
                result["status"] = "ok" if result["reachable"] else "unreachable"
        except httpx.RequestError:
            result["reachable"] = False
            result["status"] = "unreachable"

        return result
