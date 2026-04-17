"""Unit tests for HybridRetriever class."""

from unittest.mock import MagicMock

import numpy as np

from src.retrieval.hybrid import HybridRetriever


def _hit(cid: str) -> dict:
    return {"chunk_id": cid, "text": "text", "source": "f.pdf", "page": 1, "score": 0.5}


def test_hybrid_retriever_fuses_results():
    indexer = MagicMock()
    indexer.bm25_search.return_value = [_hit("a"), _hit("b")]
    indexer.vector_search.return_value = [_hit("b"), _hit("c")]

    embedder = MagicMock()
    embedder.embed_query.return_value = np.zeros(384)

    retriever = HybridRetriever(indexer, embedder, bm25_top_k=5, vector_top_k=5)
    results = retriever.retrieve("test query")

    ids = [r["chunk_id"] for r in results]
    assert set(ids) == {"a", "b", "c"}
    # "b" appears in both lists — should rank first
    assert ids[0] == "b"


def test_hybrid_retriever_calls_embed_query():
    indexer = MagicMock()
    indexer.bm25_search.return_value = []
    indexer.vector_search.return_value = []

    embedder = MagicMock()
    embedder.embed_query.return_value = np.zeros(384)

    retriever = HybridRetriever(indexer, embedder)
    retriever.retrieve("hello")

    embedder.embed_query.assert_called_once_with("hello")
