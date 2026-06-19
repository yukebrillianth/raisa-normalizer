"""OpenAI selection and verbalization provider."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import openai
from app.config import get_settings
from app.providers.base import SelectionVerbalizerProvider

logger = logging.getLogger(__name__)

ITS_SYSTEM_MESSAGE = """Anda adalah sistem asisten suara Kampus Institut Teknologi Sepuluh Nopember (ITS).
Tugas Anda adalah memilih jawaban terbaik dari kandidat yang diberikan berdasarkan pertanyaan pengguna, lalu mengubah jawaban formal tersebut menjadi kalimat lisan (spoken answer) yang alami, ramah, mudah diucapkan, dan langsung ke inti dalam Bahasa Indonesia baku yang percakapan.

Aturan:
1. Analisis transkrip mentah (raw transcript), query yang sudah dinormalisasi (normalized query), dan seluruh kandidat jawaban yang diberikan.
2. Pilih kandidat yang paling sesuai dengan MAKSUD pertanyaan, bukan sekadar skor similarity tertinggi. Jika tidak ada kandidat yang relevan atau semuanya salah, set `refused` menjadi true.
3. Cocokkan tipe pertanyaan: `siapa/siapa nama` harus memilih jawaban nama orang/pihak; jangan pilih kandidat definisi/peran/tugas jika ada kandidat nama yang sesuai. `berapa` harus memilih jumlah/angka, `kapan` waktu/tanggal, `di mana/dimana` lokasi, `bagaimana/cara` prosedur, dan `apa/apa itu` definisi.
4. Untuk kandidat terpilih:
   - Gunakan jawaban asli untuk `selected_answer`. Jangan mengarang isi baru!
   - Tulis ulang jawaban tersebut untuk `spoken_answer` agar terdengar alami saat diucapkan (lisan). Jaga agar singkat (1-3 kalimat), ramah, tanpa simbol aneh/markup, mudah didengar."""

PROMPT_TEMPLATE = """Pertanyaan mentah hasil transkripsi: {raw_transcript}
Pertanyaan hasil normalisasi model: {normalized_query}

Kandidat jawaban:
{candidates_text}

Berikan output dalam JSON:
{{"selected_rank": <nomor kandidat atau null>, "selected_question": "<pertanyaan kandidat terpilih atau kosong>", "selected_answer": "<jawaban kandidat terpilih atau kosong>", "spoken_answer": "<jawaban natural>", "reason": "<alasan memilih>", "refused": <true/false>, "refusal_reason": "<alasan menolak jika refused>"}}

Aturan:
- Pilih kandidat PALING sesuai dengan pertanyaan pengguna
- Jangan otomatis memilih rank #1. Jika query bertanya "siapa/nama", kandidat berisi nama orang yang tepat lebih baik daripada kandidat yang hanya menjelaskan peran/definisi.
- spoken_answer harus kalimat lisan alami, bukan copy-paste formal
- Jangan mengarang fakta baru di luar jawaban kandidat dan jangan mengurangi informasi daru jawaban kandidat
- Jika semua kandidat tidak relevan, set refused=true dan selected_rank=null
- Output hanya JSON valid, tanpa markdown atau penjelasan tambahan"""


class OpenAISelectionVerbalizerProvider(SelectionVerbalizerProvider):
    """Select the best retrieved answer and verbalize it using GPT-4.1-mini."""

    model = "gpt-5.4-nano"

    def __init__(self) -> None:
        self.settings = get_settings()
        self.enabled = self.settings.verbalizer_enabled

    async def process(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        raw_transcript: str = "",
    ) -> dict[str, Any]:
        """Select the best QA candidate and rephrase it for spoken output."""
        if not self.enabled:
            logger.info("OpenAI verbalization is disabled. Returning top candidate fallback.")
            return self._build_fallback_response(candidates, "Verbalization disabled")

        if not self.settings.openai_api_key:
            logger.warning("OpenAI API key is not configured. Returning top candidate fallback.")
            return self._build_fallback_response(candidates, "OpenAI API key is not configured")

        top_candidates = candidates[: self.settings.selection_candidate_k]
        if not top_candidates:
            return self._build_no_candidate_response("No candidates available")

        prompt = PROMPT_TEMPLATE.format(
            raw_transcript=(raw_transcript or query).strip(),
            normalized_query=query.strip(),
            candidates_text=self._format_candidates(top_candidates),
        )

        started_at = time.perf_counter()
        try:
            client = openai.AsyncOpenAI(api_key=self.settings.openai_api_key)  # pyright: ignore[reportAttributeAccessIssue]
            response = await client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": ITS_SYSTEM_MESSAGE},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                response_format={"type": "json_object"},
            )
        except Exception as exc:  # noqa: BLE001 - normalize third-party failures.
            logger.warning("OpenAI Selection+Verbalization request failed: %s", exc)
            fallback = self._build_fallback_response(candidates, f"OpenAI API unavailable: {exc}")
            fallback["latency_ms"] = round((time.perf_counter() - started_at) * 1000, 2)
            return fallback

        latency_ms = round((time.perf_counter() - started_at) * 1000, 2)
        content = self._extract_content(response)
        parsed = self._parse_json_response(content, top_candidates)
        parsed["latency_ms"] = latency_ms
        parsed["provider"] = "openai"
        parsed["model"] = self.model
        parsed["fallback_used"] = False
        return parsed

    async def health(self) -> dict[str, Any]:
        """Report provider readiness without exposing secrets."""
        return {
            "provider": "openai_selection_verbalizer",
            "model": self.model,
            "enabled": self.enabled,
            "api_key_configured": bool(self.settings.openai_api_key),
        }

    @staticmethod
    def _format_candidates(candidates: list[dict[str, Any]]) -> str:
        lines: list[str] = []
        for index, candidate in enumerate(candidates, 1):
            similarity = OpenAISelectionVerbalizerProvider._coerce_float(candidate.get("similarity"))
            rerank_score = OpenAISelectionVerbalizerProvider._coerce_float(candidate.get("rerank_score"))
            lines.append(
                f"{index}. Pertanyaan: {candidate.get('question', '')} | "
                f"Jawaban: {candidate.get('answer', '')} | "
                f"Skor: similarity={similarity:.4f}, rerank_score={rerank_score:.4f}"
            )
        return "\n".join(lines)

    @staticmethod
    def _extract_content(response: Any) -> str:
        try:
            return str(response.choices[0].message.content or "").strip()
        except Exception:  # noqa: BLE001 - support dict-like SDK variants.
            if isinstance(response, dict):
                choices = response.get("choices") or []
                if choices:
                    message = choices[0].get("message") or {}
                    return str(message.get("content") or "").strip()
            return ""

    def _parse_json_response(self, content: str, candidates: list[dict[str, Any]]) -> dict[str, Any]:
        try:
            data = json.loads(content)
        except json.JSONDecodeError as exc:
            logger.warning("Failed to parse OpenAI Selection+Verbalization JSON: %s", exc)
            return self._build_fallback_response(candidates, f"Invalid JSON response: {exc}")

        selected_rank = self._coerce_selected_rank(data.get("selected_rank"), len(candidates))
        refused = bool(data.get("refused", False))
        if refused or selected_rank is None:
            return {
                "selected_rank": None,
                "selected_question": "",
                "selected_answer": "",
                "spoken_answer": "",
                "reason": str(data.get("reason", "")).strip(),
                "refused": True,
                "refusal_reason": str(data.get("refusal_reason", "Tidak ada kandidat yang relevan")).strip(),
            }

        selected_candidate = candidates[selected_rank - 1]
        return {
            "selected_rank": selected_rank,
            "selected_question": str(selected_candidate.get("question", "")).strip(),
            "selected_answer": str(selected_candidate.get("answer", "")).strip(),
            "spoken_answer": str(data.get("spoken_answer", "")).strip(),
            "reason": str(data.get("reason", "")).strip(),
            "refused": False,
            "refusal_reason": "",
        }

    @staticmethod
    def _coerce_selected_rank(value: Any, candidate_count: int) -> int | None:
        if value is None:
            return None
        try:
            rank = int(value)
        except (TypeError, ValueError):
            return None
        return rank if 1 <= rank <= candidate_count else None

    @staticmethod
    def _coerce_float(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _build_fallback_response(candidates: list[dict[str, Any]], reason: str) -> dict[str, Any]:
        if not candidates:
            return OpenAISelectionVerbalizerProvider._build_no_candidate_response(reason)

        best = candidates[0]
        return {
            "selected_rank": 1,
            "selected_question": str(best.get("question", "")).strip(),
            "selected_answer": str(best.get("answer", "")).strip(),
            "spoken_answer": str(best.get("answer", "")).strip(),
            "reason": f"{reason}; using top-1 candidate fallback",
            "refused": False,
            "refusal_reason": "",
            "latency_ms": 0.0,
            "provider": "openai",
            "model": OpenAISelectionVerbalizerProvider.model,
            "fallback_used": True,
        }

    @staticmethod
    def _build_no_candidate_response(reason: str) -> dict[str, Any]:
        return {
            "selected_rank": None,
            "selected_question": "",
            "selected_answer": "",
            "spoken_answer": "",
            "reason": reason,
            "refused": True,
            "refusal_reason": "No candidate",
            "latency_ms": 0.0,
            "provider": "openai",
            "model": OpenAISelectionVerbalizerProvider.model,
            "fallback_used": True,
        }
