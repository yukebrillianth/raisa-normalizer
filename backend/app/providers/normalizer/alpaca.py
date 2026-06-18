"""Alpaca-format normalizer provider for informal-to-formal Indonesian query rewriting.

Uses unsloth.FastLanguageModel (matching the training notebook exactly) with optional
LoRA adapter via peft to rewrite informal Indonesian queries into formal (baku)
Indonesian following the exact Alpaca prompt format from the training notebook.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from pathlib import Path
from typing import Any

import torch
from app.config import get_settings
from app.providers.base import NormalizerProvider
from transformers import (AutoTokenizer, BitsAndBytesConfig,
                          PreTrainedTokenizerFast)
from unsloth import FastLanguageModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Serialise GPU inference to prevent OOM when multiple requests stack up
# ---------------------------------------------------------------------------
_INFERENCE_LOCK = threading.Lock()

# ---------------------------------------------------------------------------
# Prompt format — MUST match notebook exactly
# ---------------------------------------------------------------------------

ALPACA_PROMPT_TEMPLATE = """### Instruction:
Anda adalah sistem layanan informasi Kampus Institut Teknologi Sepuluh Nopember (ITS). Ubah input kalimat Bahasa Indonesia tidak baku berikut menjadi Bahasa Indonesia baku dan formal.
Jangan menambahkan informasi baru.
Gunakan kalimat singkat, jelas, baku, dan langsung ke inti.

### Input:
{informal_text}

### Response:
"""


def _format_prompt(informal_text: str) -> str:
    """Render the Alpaca prompt with the user's informal text."""
    return ALPACA_PROMPT_TEMPLATE.format(informal_text=informal_text)


def _clean_output(text: str) -> str:
    """Clean generated output.

    Mirrors the notebook's ``clean_output``:
    1. Remove ``<think>...</think>`` blocks
    2. Strip surrounding quotes
    3. Collapse whitespace
    """
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    cleaned = cleaned.strip().strip('"').strip("'")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


class AlpacaNormalizerProvider(NormalizerProvider):
    """Normalizer using a fine-tuned Alpaca-format LLM.

    Loads base model from ``NORMALIZER_BASE_MODEL_PATH`` and an optional
    LoRA adapter from ``NORMALIZER_LORA_PATH``.  Inference is deterministic
    (greedy decoding) and serialised behind a mutex.

    Singleton-friendly: call ``.get_instance()`` to share the loaded model
    across the application.
    """

    _instance: AlpacaNormalizerProvider | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def __init__(self) -> None:
        self._model_loaded = False
        self._adapter_loaded = False
        self._device: str = "cpu"
        self._model: Any = None
        self._tokenizer: AutoTokenizer | None = None

        settings = get_settings()
        self._base_model_path: str = settings.normalizer_base_model_path
        self._lora_path: str = settings.normalizer_lora_path
        self._max_input_tokens: int = settings.normalizer_max_input_tokens
        self._max_new_tokens: int = settings.normalizer_max_new_tokens

    @classmethod
    def get_instance(cls) -> AlpacaNormalizerProvider:
        """Return (or create) the process-wide singleton."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------
    # Model loading (lazy)
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Load the base model and LoRA adapter.

        Safe to call multiple times — immediately returns if already loaded.
        """
        if self._model_loaded:
            return

        try:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info("Loading tokenizer from %s", self._base_model_path)

            # self._tokenizer = AutoTokenizer.from_pretrained(
            #     self._base_model_path,
            #     trust_remote_code=True,
            #     fix_mistral_regex=True,
            # )
            self._tokenizer = PreTrainedTokenizerFast.from_pretrained(
                self._base_model_path,
                trust_remote_code=True,
            )
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            logger.info("Loading base model from %s (device: %s)", self._base_model_path, self._device)
            quantization_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0)
            self._model, _ = FastLanguageModel.from_pretrained(
                self._base_model_path,
                max_seq_length=128,
                dtype=None,
                load_in_4bit=False,
                load_in_8bit=True,
                quantization_config=quantization_config,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            self._model_loaded = True

            # Optional LoRA adapter
            lora_path = Path(self._lora_path)
            adapter_config = lora_path / "adapter_config.json"
            if self._lora_path and adapter_config.exists():
                from peft import PeftModel  # type: ignore[import-untyped]

                logger.info("Loading LoRA adapter from %s", self._lora_path)
                self._model = PeftModel.from_pretrained(self._model, self._lora_path)
                self._adapter_loaded = True
            elif self._lora_path:
                logger.warning(
                    "LoRA path configured but adapter_config.json missing at %s",
                    adapter_config,
                )

            self._model.eval()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info(
                "Alpaca normalizer: model_loaded=%s, adapter_loaded=%s, device=%s",
                self._model_loaded,
                self._adapter_loaded,
                self._device,
            )

        except Exception:
            logger.exception("Failed to load Alpaca normalizer model")
            self._model_loaded = False
            self._adapter_loaded = False

    # ------------------------------------------------------------------
    # NormalizerProvider interface
    # ------------------------------------------------------------------

    async def process(self, transcript: str) -> str:
        """Normalize an informal Indonesian transcript into a formal query.

        Returns the original transcript unchanged if the model failed to load.
        """
        if not transcript.strip():
            return transcript

        self._load_model()
        if not self._model_loaded or self._model is None or self._tokenizer is None:
            logger.error("Alpaca normalizer: model not available — returning raw transcript")
            return transcript

        prompt = _format_prompt(transcript)

        start = time.perf_counter()

        with _INFERENCE_LOCK:
            inputs = self._tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self._max_input_tokens,
            ).to(self._device)

            self._model.eval()  # type: ignore[union-attr]

            with torch.no_grad():
                outputs = self._model.generate(  # type: ignore[union-attr]
                    **inputs,
                    max_new_tokens=self._max_new_tokens,
                    do_sample=False,
                    temperature=0.0,
                    top_p=1.0,
                    num_beams=1,
                    pad_token_id=self._tokenizer.pad_token_id,
                    eos_token_id=self._tokenizer.eos_token_id,
                    repetition_penalty=1.05,
                    use_cache=True,
                )

            decoded = self._tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[-1] :],
                skip_special_tokens=True,
            )

            del inputs, outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        latency_ms = (time.perf_counter() - start) * 1000
        logger.info("Alpaca normalizer: %.1f ms", latency_ms)

        return _clean_output(decoded)

    async def health(self) -> dict[str, Any]:
        """Return readiness information for the normalizer."""
        if not self._model_loaded:
            try:
                self._load_model()
            except Exception:
                pass

        return {
            "model_loaded": self._model_loaded,
            "adapter_loaded": self._adapter_loaded,
            "device": self._device,
            "provider": "alpaca",
        }