"""pgvector + BM25 retrieval with Reciprocal Rank Fusion reranking.

Matches the retrieval logic from ``test_retrievalddd.md``:
build a BM25 index over all database questions, retrieve top-k vector and
BM25 candidates, fuse both rank lists with RRF, then apply the threshold to
the selected candidate's original vector cosine similarity.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from app.config import get_settings
from app.db import get_db
from app.providers.base import RetrievalProvider
from app.providers.schemas import (BaselineRerankSelected, RetrievalCandidate,
                                   RetrievalResult)
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


NAME_QUERY_TERMS = {"siapa", "siapakah", "nama"}
QUESTION_STOPWORDS = {
    "apa",
    "apakah",
    "bagaimana",
    "berapa",
    "dan",
    "di",
    "dimana",
    "mana",
    "nama",
    "siapa",
    "siapakah",
    "yang",
}
GENERIC_NAME_TOKENS = {"nama", "sekretaris", "kepala", "ketua", "direktur", "dekan"}
NAME_CANDIDATE_PHRASES = ("siapa nama", "siapakah nama")
DEFINITION_CANDIDATE_PHRASES = (
    "apa yang dimaksud",
    "apa itu",
    "peran",
    "tugas",
    "fungsi",
    "definisi",
)


@dataclass(frozen=True)
class BM25Candidate:
    question: str
    answer: str
    bm25_score: float
    bm25_rank: int


@dataclass(frozen=True)
class BM25Corpus:
    questions: list[str]
    answers: dict[str, str]
    index: BM25Okapi | None


class PgvectorRetrievalProvider(RetrievalProvider):
    """Retrieves QA candidates with pgvector + BM25 Reciprocal Rank Fusion.

    Pipeline:

    1. Encode the input embedding vector as a pgvector‑compatible literal.
    2. Query ``qa_ground_truth`` via cosine distance (``<=>`` operator).
    3. Query the in-memory BM25 index for top keyword candidates.
    4. Fuse vector and BM25 ranks: ``1 / (RRF_K + rank)`` per source.
    5. Sort by rerank_score descending; select top‑1 as baseline.
    6. Apply threshold on **original similarity** to determine ``answered``.

    Parameters read from config:
    - ``RETRIEVAL_TOP_K`` — max candidates to retrieve (default 3).
    - ``RETRIEVAL_SIMILARITY_THRESHOLD`` — min similarity to flag as answered.
    - ``RETRIEVAL_BM25_K`` — max BM25 candidates (defaults to top_k).
    - ``RETRIEVAL_RRF_K`` — RRF rank constant (default 60).
    - ``QA_TABLE`` — the target table name.
    """

    _bm25_corpus: BM25Corpus | None = None

    def __init__(self) -> None:
        settings = get_settings()
        self.top_k: int = settings.retrieval_top_k
        self.candidate_k: int = max(settings.retrieval_candidate_k, self.top_k)
        self.similarity_threshold: float = settings.retrieval_similarity_threshold
        self.bm25_k: int = max(getattr(settings, "retrieval_bm25_k", self.top_k), self.candidate_k)
        self.rrf_k: int = getattr(settings, "retrieval_rrf_k", 60)
        self.intent_prior_enabled: bool = settings.retrieval_intent_prior_enabled
        self.name_intent_boost: float = settings.retrieval_name_intent_boost
        self.name_intent_penalty: float = settings.retrieval_name_intent_penalty
        self.qa_table: str = settings.qa_table

    @classmethod
    def invalidate_bm25_index(cls) -> None:
        """Force the next retrieval request to rebuild the BM25 corpus."""
        cls._bm25_corpus = None

    def _load_bm25_corpus(self) -> BM25Corpus:
        """Build or return the cached BM25 corpus from all DB questions."""
        if self.__class__._bm25_corpus is not None:
            return self.__class__._bm25_corpus

        db = get_db()
        order_col = "id" if "id" in db.schema.columns else "question"
        rows = db.execute_query(
            f"SELECT question, answer FROM {self.qa_table} ORDER BY {order_col};"
        )
        questions = [str(row[0]) for row in rows]
        answers = {str(question): str(answer) for question, answer in rows}
        tokenized_corpus = [question.lower().split() for question in questions]
        corpus = BM25Corpus(
            questions=questions,
            answers=answers,
            index=BM25Okapi(tokenized_corpus) if tokenized_corpus else None,
        )
        self.__class__._bm25_corpus = corpus
        logger.info("BM25 index built: %d database questions", len(questions))
        return corpus

    def _get_bm25_candidates(self, query: str) -> list[BM25Candidate]:
        """Return top-k BM25 candidates following the reference notebook."""
        corpus = self._load_bm25_corpus()
        if not corpus.questions or corpus.index is None:
            return []

        tokenized_query = query.lower().split()
        scores = corpus.index.get_scores(tokenized_query)
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda idx: float(scores[idx]),
            reverse=True,
        )[: self.bm25_k]
        return [
            BM25Candidate(
                question=corpus.questions[index],
                answer=corpus.answers[corpus.questions[index]],
                bm25_score=float(scores[index]),
                bm25_rank=rank + 1,
            )
            for rank, index in enumerate(ranked_indices)
        ]

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return set(re.findall(r"[a-z0-9]+", text.lower()))

    def _intent_adjustment(
        self,
        query: str,
        candidate_question: str,
        similarity: float,
    ) -> float:
        """Small deterministic prior for matching Indonesian question intent.

        BM25/RRF can over-rank entity-overlap candidates such as definitions
        ("Apa yang dimaksud ...") for name queries ("Siapa nama ...").  This
        keeps the RRF structure intact while nudging close ties toward the
        candidate whose question type answers the user's intent.
        """
        if not self.intent_prior_enabled:
            return 0.0

        query_tokens = self._tokens(query)
        if not (query_tokens & NAME_QUERY_TERMS):
            return 0.0

        candidate_tokens = self._tokens(candidate_question)
        entity_tokens = {
            token
            for token in query_tokens - QUESTION_STOPWORDS - GENERIC_NAME_TOKENS
            if len(token) > 2
        }
        entity_overlap = entity_tokens & candidate_tokens
        entity_matches = not entity_tokens or entity_overlap == entity_tokens

        candidate = candidate_question.lower().strip()
        if similarity <= 0.0:
            return -self.name_intent_penalty
        if not entity_matches:
            return -self.name_intent_penalty
        if any(phrase in candidate for phrase in NAME_CANDIDATE_PHRASES):
            return self.name_intent_boost
        if candidate.startswith("siapa") or candidate.startswith("siapakah"):
            return self.name_intent_boost / 2
        if any(phrase in candidate for phrase in DEFINITION_CANDIDATE_PHRASES):
            return -self.name_intent_penalty
        return 0.0

    async def process(self, embedding: list[float], query: str = "") -> RetrievalResult:
        """Run pgvector retrieval + BM25 Reciprocal Rank Fusion.

        Args:
            embedding: Dense query vector (from ``BGEEmbeddingProvider``).
            query: The original normalised query text, required for keyword
                   BM25 tokenization.

        Returns:
            A ``RetrievalResult`` with candidates, baseline selection, and
            ``answered`` flag.
        """
        # --- 1. Build pgvector-compatible vector literal ---
        # Notebook pattern:
        #   vector_literal = "[" + ",".join(map(str, row["query_vector"])) + "]"
        vector_literal = "[" + ",".join(map(str, embedding)) + "]"

        # --- 2. pgvector retrieval (notebook:418-426) ---
        db = get_db()
        rows = db.execute_query(
            f"""SELECT question, answer,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM {self.qa_table}
                ORDER BY embedding <=> %s::vector
                LIMIT %s;""",
            (vector_literal, vector_literal, self.candidate_k),
        )

        # --- 3. Build vector rank map (notebook:433-436) ---
        vec_map: dict[str, dict[str, Any]] = {}
        for rank, (question, answer, similarity) in enumerate(rows, start=1):
            vec_map[str(question)] = {
                "answer": str(answer),
                "similarity": float(similarity),
                "vector_rank": rank,
            }

        # --- 4. BM25 search (notebook:438-440) ---
        bm25_candidates = self._get_bm25_candidates(query)
        bm25_map = {candidate.question: candidate for candidate in bm25_candidates}

        # --- 5. Reciprocal Rank Fusion (notebook:442-464) ---
        fused: list[dict[str, Any]] = []
        all_questions = list(dict.fromkeys([*vec_map.keys(), *bm25_map.keys()]))
        for question in all_questions:
            vector_rank = (
                vec_map[question]["vector_rank"]
                if question in vec_map
                else self.candidate_k + 1
            )
            bm25_rank = (
                bm25_map[question].bm25_rank
                if question in bm25_map
                else self.bm25_k + 1
            )
            rrf_score = 1 / (self.rrf_k + vector_rank) + 1 / (self.rrf_k + bm25_rank)
            similarity = vec_map[question]["similarity"] if question in vec_map else 0.0
            intent_adjustment = self._intent_adjustment(query, question, similarity)
            answer = (
                vec_map[question]["answer"]
                if question in vec_map
                else bm25_map[question].answer
            )

            fused.append(
                {
                    "question": question,
                    "answer": answer,
                    "similarity": similarity,
                    "keyword_score": 0.0,
                    "rerank_score": rrf_score + intent_adjustment,
                    "vector_rank": vector_rank,
                    "bm25_rank": bm25_rank,
                    "intent_adjustment": intent_adjustment,
                }
            )

        reranked = sorted(
            fused,
            key=lambda c: (c["rerank_score"], -c["vector_rank"], -c["bm25_rank"]),
            reverse=True,
        )

        # --- 6. Baseline rerank selection ---
        baseline_rerank_selected: BaselineRerankSelected | None = None
        answered: bool = False

        if reranked:
            # Threshold applied to ORIGINAL similarity, NOT rerank_score
            # (notebook:352 — reranked[0]["similarity"] >= SIMILARITY_THRESHOLD)
            if reranked[0]["similarity"] >= self.similarity_threshold:
                answered = True
            baseline_rerank_selected = BaselineRerankSelected(
                question=reranked[0]["question"],
                answer=reranked[0]["answer"],
                similarity=reranked[0]["similarity"],
                rerank_score=reranked[0]["rerank_score"],
            )

        # --- 7. Wrap in RetrievalResult ---
        result = RetrievalResult(
            top_k=self.top_k,
            similarity_threshold=self.similarity_threshold,
            rerank_keyword_weight=0.0,
            candidates=[
                RetrievalCandidate(
                    question=c["question"],
                    answer=c["answer"],
                    similarity=c["similarity"],
                    keyword_score=c.get("keyword_score", 0.0),
                    rerank_score=c.get("rerank_score", 0.0),
                )
                for c in reranked[: self.candidate_k]
            ],
            baseline_rerank_selected=baseline_rerank_selected,
            answered=answered,
        )

        logger.debug(
            "Retrieval: %d fused candidates, answered=%s, baseline_sim=%.4f",
            len(reranked),
            answered,
            baseline_rerank_selected.similarity if baseline_rerank_selected else 0.0,
        )
        return result

    async def health(self) -> dict[str, Any]:
        """Report retrieval provider readiness."""
        return {
            "provider": "pgvector",
            "table": self.qa_table,
            "top_k": self.top_k,
            "candidate_k": self.candidate_k,
            "bm25_k": self.bm25_k,
            "rrf_k": self.rrf_k,
            "similarity_threshold": self.similarity_threshold,
            "intent_prior_enabled": self.intent_prior_enabled,
            "rerank_strategy": "bm25_vector_rrf",
        }
