"""Hybrid retriever: fuses BM25 + vector results via Reciprocal Rank Fusion."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.ingestion.embedder import Embedder
from src.ingestion.indexer import DualIndexer


def _rrf_score(rank: int, k: int) -> float:
    return 1.0 / (k + rank + 1)


def reciprocal_rank_fusion(
    bm25_results: list[dict[str, Any]],
    vector_results: list[dict[str, Any]],
    rrf_k: int = 60,
) -> list[dict[str, Any]]:
    """Merge two ranked lists using RRF. Returns dicts sorted by fused score descending."""
    scores: dict[str, float] = {}
    by_id: dict[str, dict[str, Any]] = {}

    for rank, hit in enumerate(bm25_results):
        cid = hit["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + _rrf_score(rank, rrf_k)
        by_id[cid] = hit

    for rank, hit in enumerate(vector_results):
        cid = hit["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + _rrf_score(rank, rrf_k)
        by_id[cid] = hit

    fused = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)
    return [{**by_id[cid], "rrf_score": scores[cid]} for cid in fused]


class HybridRetriever:
    def __init__(
        self,
        indexer: DualIndexer,
        embedder: Embedder,
        bm25_top_k: int = 20,
        vector_top_k: int = 20,
        rrf_k: int = 60,
    ):
        self._indexer = indexer
        self._embedder = embedder
        self._bm25_top_k = bm25_top_k
        self._vector_top_k = vector_top_k
        self._rrf_k = rrf_k

    def retrieve(self, query: str) -> list[dict[str, Any]]:
        query_emb: np.ndarray = self._embedder.embed_query(query)
        bm25_hits = self._indexer.bm25_search(query, self._bm25_top_k)
        vector_hits = self._indexer.vector_search(query_emb, self._vector_top_k)
        return reciprocal_rank_fusion(bm25_hits, vector_hits, self._rrf_k)
