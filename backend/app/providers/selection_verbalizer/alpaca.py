"""Alpaca Selection and Verbalization Provider.

Uses the fine-tuned Alpaca model (reusing normalizer's singleton model load)
to select the best candidate answer from retrieved results and rephrase it
into a natural spoken Indonesian answer.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

import torch

from app.config import get_settings
from app.providers.base import SelectionVerbalizerProvider
from app.providers.normalizer.alpaca import AlpacaNormalizerProvider

logger = logging.getLogger(__name__)

VERBALIZATION_PROMPT_TEMPLATE = """### Instruction:
Anda adalah sistem asisten suara Kampus Institut Teknologi Sepuluh Nopember (ITS).
Tugas Anda adalah memilih jawaban terbaik dari kandidat yang diberikan berdasarkan pertanyaan pengguna, lalu mengubah jawaban formal tersebut menjadi kalimat lisan (spoken answer) yang alami, ramah, mudah diucapkan, dan langsung ke inti dalam Bahasa Indonesia baku yang percakapan.

Aturan:
1. Analisis transkrip mentah (raw transcript), query yang sudah dinormalisasi (normalized query), dan daftar top-3 kandidat jawaban.
2. Pilih salah satu kandidat yang paling sesuai (selected_rank dari 1 sampai 3). Jika tidak ada kandidat yang relevan atau semuanya salah, set `refused` menjadi true.
3. Untuk kandidat terpilih:
   - Gunakan jawaban asli untuk `selected_answer`. Jangan mengarang isi baru!
   - Tulis ulang jawaban tersebut untuk `spoken_answer` agar terdengar alami saat diucapkan (lisan). Jaga agar singkat (1-3 kalimat), ramah, tanpa simbol aneh/markup, mudah didengar.
4. Output harus berupa JSON valid tanpa teks penjelasan di luar JSON. Format JSON:
{{
  "selected_rank": <int atau null>,
  "selected_answer": "<jawaban tertulis asli dari kandidat terpilih>",
  "spoken_answer": "<jawaban lisan hasil refrase Anda>",
  "reason": "<alasan memilih>",
  "refused": <true atau false>,
  "refusal_reason": "<alasan penolakan jika tidak ada yang sesuai>"
}}

### Context:
- Raw Transcript: {raw_transcript}
- Normalized Query: {normalized_query}

Kandidat Jawaban:
{candidates_text}

### Response:
"""


class AlpacaSelectionVerbalizerProvider(SelectionVerbalizerProvider):
    """LLM selection and verbalization provider powered by Alpaca."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.enabled = self.settings.verbalizer_enabled
        self.max_new_tokens = self.settings.verbalizer_max_new_tokens

    async def process(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        raw_transcript: str = "",
    ) -> dict[str, Any]:
        """Select the best QA candidate and verbalize it for TTS."""
        if not self.enabled:
            logger.info("Verbalization is disabled. Returning baseline/fallback.")
            return self._build_disabled_response(candidates)

        normalizer = AlpacaNormalizerProvider.get_instance()
        normalizer._load_model()

        if not normalizer._model_loaded or normalizer._model is None or normalizer._tokenizer is None:
            logger.error("Alpaca model not available for Selection+Verbalization. Returning baseline.")
            return self._build_disabled_response(candidates)

        candidates_text_list = []
        for i, candidate in enumerate(candidates, 1):
            candidates_text_list.append(
                f"Kandidat {i}:\n"
                f"- Pertanyaan Terkait: {candidate.get('question', '')}\n"
                f"- Jawaban: {candidate.get('answer', '')}\n"
                f"- Similarity: {candidate.get('similarity', 0.0):.4f}\n"
                f"- Rerank Score: {candidate.get('rerank_score', 0.0):.4f}\n"
            )
        candidates_text = "\n".join(candidates_text_list)

        prompt = VERBALIZATION_PROMPT_TEMPLATE.format(
            raw_transcript=raw_transcript or query,
            normalized_query=query,
            candidates_text=candidates_text,
        )

        start_time = time.perf_counter()
        from app.providers.normalizer.alpaca import _INFERENCE_LOCK

        with _INFERENCE_LOCK:
            inputs = normalizer._tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
            ).to(normalizer._device)

            normalizer._model.eval()
            with torch.no_grad():
                outputs = normalizer._model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    temperature=0.0,
                    top_p=1.0,
                    num_beams=1,
                    pad_token_id=normalizer._tokenizer.pad_token_id,
                    eos_token_id=normalizer._tokenizer.eos_token_id,
                    repetition_penalty=1.05,
                    use_cache=True,
                )

            decoded = normalizer._tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[-1] :],
                skip_special_tokens=True,
            )
            del inputs, outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        logger.info("Alpaca Selection+Verbalizer: %.1f ms", latency_ms)

        parsed = self._parse_json_response(decoded)
        parsed["latency_ms"] = latency_ms
        parsed["provider"] = "alpaca"
        parsed["fallback_used"] = False
        return parsed

    def _parse_json_response(self, text: str) -> dict[str, Any]:
        """Extract and parse valid JSON from LLM output."""
        cleaned = re.sub(r"<think>.*?</think>", "", text.strip(), flags=re.DOTALL).strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if match:
            cleaned = match.group(0)

        try:
            data = json.loads(cleaned)
            selected_rank = self._coerce_selected_rank(data.get("selected_rank"))
            return {
                "selected_rank": selected_rank,
                "selected_answer": str(data.get("selected_answer", "")).strip(),
                "spoken_answer": str(data.get("spoken_answer", "")).strip(),
                "reason": str(data.get("reason", "")).strip(),
                "refused": bool(data.get("refused", False)),
                "refusal_reason": str(data.get("refusal_reason", "")).strip(),
            }
        except Exception as exc:
            logger.error("Failed to parse Selection+Verbalization JSON. Raw: %s, Error: %s", cleaned, exc)
            return {
                "selected_rank": None,
                "selected_answer": "",
                "spoken_answer": "",
                "reason": f"Failed to parse LLM output JSON: {exc}",
                "refused": True,
                "refusal_reason": "Output format error",
            }

    @staticmethod
    def _coerce_selected_rank(value: Any) -> int | None:
        """Coerce common LLM JSON variants into a valid 1-3 rank or None."""
        if value is None:
            return None
        try:
            rank = int(value)
        except (TypeError, ValueError):
            return None
        return rank if 1 <= rank <= 3 else None

    @staticmethod
    def _build_disabled_response(candidates: list[dict[str, Any]]) -> dict[str, Any]:
        """Fallback response structure when verbalization is disabled or model fails."""
        if candidates:
            best = candidates[0]
            return {
                "selected_rank": 1,
                "selected_answer": best.get("answer", ""),
                "spoken_answer": best.get("answer", ""),
                "reason": "Verbalization disabled/failed, using top-1 candidate",
                "refused": False,
                "refusal_reason": "",
                "latency_ms": 0.0,
                "provider": "alpaca",
                "fallback_used": True,
            }
        return {
            "selected_rank": None,
            "selected_answer": "",
            "spoken_answer": "",
            "reason": "No candidates available and verbalization disabled/failed",
            "refused": True,
            "refusal_reason": "No candidate",
            "latency_ms": 0.0,
            "provider": "alpaca",
            "fallback_used": True,
        }

    async def health(self) -> dict[str, Any]:
        """Report provider health."""
        normalizer = AlpacaNormalizerProvider.get_instance()
        return {
            "provider": "alpaca_selection_verbalizer",
            "enabled": self.enabled,
            "model_loaded": normalizer._model_loaded,
        }
