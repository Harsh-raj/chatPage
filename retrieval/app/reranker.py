"""
Reranker abstraction.

NoOpReranker passes candidates through unchanged (sorted by whatever score
they already had) -- used for dev/test, and as the default so reranking is
opt-in via USE_RERANKER.

CrossEncoderReranker uses a real cross-encoder model that reads the query
and each candidate TOGETHER (unlike dense/BM25 search, which score them
independently), giving a much more accurate relevance judgment.

Both implementations support score-threshold filtering: candidates scoring
below the threshold are dropped entirely, rather than always padding the
result out to top_k regardless of quality. This is what prevents weak,
"fake-looking" matches from being force-fed into the LLM's context.
"""

from __future__ import annotations


class Reranker:
    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int = 5,
        threshold: float | None = None,
    ) -> list[dict]:
        raise NotImplementedError


class NoOpReranker(Reranker):
    """Pass-through: keeps whatever ordering/score the candidates already had."""

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int = 5,
        threshold: float | None = None,
    ) -> list[dict]:
        results = candidates[:top_k]
        if threshold is not None:
            results = [c for c in results if c.get("score", c.get("rrf_score", 0)) >= threshold]
        return results


class CrossEncoderReranker(Reranker):
    """
    Real reranking via a cross-encoder model. Default model
    (cross-encoder/ms-marco-MiniLM-L-6-v2) is small and CPU-feasible.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int = 5,
        threshold: float | None = None,
    ) -> list[dict]:
        if not candidates:
            return []

        import math

        pairs = [(query, c["text"]) for c in candidates]
        raw_scores = self.model.predict(pairs)
        # Normalize raw logit scores to 0-1 via sigmoid, so a threshold like
        # 0.5 has a consistent, interpretable meaning regardless of model.
        scored = [{**c, "rerank_score": 1 / (1 + math.exp(-s))} for c, s in zip(candidates, raw_scores, strict=True)]
        scored.sort(key=lambda x: -x["rerank_score"])

        if threshold is not None:
            scored = [c for c in scored if c["rerank_score"] >= threshold]

        return scored[:top_k]
