"""Cross-encoder reranker using sentence-transformers CrossEncoder."""

from __future__ import annotations

from sentence_transformers import CrossEncoder


class Reranker:
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_k: int = 5,
    ):
        self._model = CrossEncoder(model_name)
        self._top_k = top_k

    def rerank(self, query: str, hits: list[dict]) -> list[dict]:
        """Score each hit against the query and return top_k sorted by relevance."""
        if not hits:
            return []

        pairs = [(query, hit["text"]) for hit in hits]
        scores = self._model.predict(pairs)

        ranked = sorted(
            zip(scores, hits),
            key=lambda x: x[0],
            reverse=True,
        )
        return [
            {**hit, "rerank_score": float(score)}
            for score, hit in ranked[: self._top_k]
        ]
