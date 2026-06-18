"""pgvector-based retrieval with hybrid keyword‑overlap reranking.

Matches the retrieval and reranking logic from the evaluation notebook:
``_MConverter.eu_test_retrieval.md:313-317`` (keyword overlap),
``:326-357`` (pgvector query + hybrid rerank + threshold).
"""

from __future__ import annotations

import logging
from typing import Any

from app.config import get_settings
from app.db import get_db
from app.providers.base import RetrievalProvider
from app.providers.schemas import BaselineRerankSelected, RetrievalCandidate, RetrievalResult

logger = logging.getLogger(__name__)


def keyword_overlap_score(query: str, candidate: str) -> float:
    """Jaccard‑like keyword overlap between query and candidate question.

    Exact reproduction of the notebook's ``keyword_overlap`` function
    (``_MConverter.eu_test_retrieval.md:313-316``).

    Args:
        query: The normalised query string.
        candidate: The candidate question string.

    Returns:
        A float in [0, 1] — higher values mean more word overlap.
    """
    q_words = set(str(query).lower().split())
    c_words = set(str(candidate).lower().split())
    return len(q_words & c_words) / (len(q_words) + 1e-6)


class PgvectorRetrievalProvider(RetrievalProvider):
    """Retrieves QA candidates from PostgreSQL with pgvector and hybrid reranking.

    Pipeline:

    1. Encode the input embedding vector as a pgvector‑compatible literal.
    2. Query ``qa_ground_truth`` via cosine distance (``<=>`` operator).
    3. Compute keyword overlap score for every candidate.
    4. Hybrid reranking: ``rerank_score = similarity + K * keyword_score``.
    5. Sort by rerank_score descending; select top‑1 as baseline.
    6. Apply threshold on **original similarity** to determine ``answered``.

    Parameters read from config:
    - ``RETRIEVAL_TOP_K`` — max candidates to retrieve (default 3).
    - ``RETRIEVAL_SIMILARITY_THRESHOLD`` — min similarity to flag as answered.
    - ``RERANK_KEYWORD_WEIGHT`` — weight of keyword score in hybrid formula.
    - ``QA_TABLE`` — the target table name.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self.top_k: int = settings.retrieval_top_k
        self.similarity_threshold: float = settings.retrieval_similarity_threshold
        self.rerank_keyword_weight: float = settings.rerank_keyword_weight
        self.qa_table: str = settings.qa_table

    async def process(self, embedding: list[float], query: str = "") -> RetrievalResult:
        """Run pgvector retrieval + hybrid keyword‑overlap reranking.

        Args:
            embedding: Dense query vector (from ``BGEEmbeddingProvider``).
            query: The original normalised query text, required for keyword
                   overlap scoring (default ``""`` skips overlap scoring).

        Returns:
            A ``RetrievalResult`` with candidates, baseline selection, and
            ``answered`` flag.
        """
        # --- 1. Build pgvector-compatible vector literal ---
        # Notebook pattern:
        #   vector_literal = "[" + ",".join(map(str, row["query_vector"])) + "]"
        vector_literal = "[" + ",".join(map(str, embedding)) + "]"

        # --- 2. pgvector retrieval (notebook:326-332) ---
        db = get_db()
        rows = db.execute_query(
            f"""SELECT question, answer,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM {self.qa_table}
                ORDER BY embedding <=> %s::vector
                LIMIT %s;""",
            (vector_literal, vector_literal, self.top_k),
        )

        # --- 3. Build candidate list ---
        candidates: list[dict[str, Any]] = [
            {"question": q, "answer": a, "similarity": float(s)}
            for (q, a, s) in rows
        ]

        # --- 4. Hybrid reranking (notebook:344-347) ---
        for c in candidates:
            c["keyword_score"] = keyword_overlap_score(query, c["question"])
            c["rerank_score"] = (
                c["similarity"] + self.rerank_keyword_weight * c["keyword_score"]
            )

        # Sort by rerank_score descending (notebook:349)
        reranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)

        # --- 5. Baseline rerank selection ---
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

        # --- 6. Wrap in RetrievalResult ---
        result = RetrievalResult(
            top_k=self.top_k,
            similarity_threshold=self.similarity_threshold,
            rerank_keyword_weight=self.rerank_keyword_weight,
            candidates=[
                RetrievalCandidate(
                    question=c["question"],
                    answer=c["answer"],
                    similarity=c["similarity"],
                    keyword_score=c.get("keyword_score", 0.0),
                    rerank_score=c.get("rerank_score", 0.0),
                )
                for c in candidates
            ],
            baseline_rerank_selected=baseline_rerank_selected,
            answered=answered,
        )

        logger.debug(
            "Retrieval: %d candidates, answered=%s, baseline_sim=%.4f",
            len(candidates),
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
            "similarity_threshold": self.similarity_threshold,
            "rerank_keyword_weight": self.rerank_keyword_weight,
        }
