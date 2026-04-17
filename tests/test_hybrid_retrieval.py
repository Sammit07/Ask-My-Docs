"""Unit tests for RRF fusion and hybrid retriever."""

import pytest

from src.retrieval.hybrid import reciprocal_rank_fusion


def _hit(chunk_id: str, score: float = 1.0) -> dict:
    return {"chunk_id": chunk_id, "text": f"text for {chunk_id}", "score": score, "source": "test.txt", "page": 1}


def test_rrf_combines_disjoint_lists():
    bm25 = [_hit("a"), _hit("b"), _hit("c")]
    vec = [_hit("d"), _hit("e"), _hit("a")]
    result = reciprocal_rank_fusion(bm25, vec)
    ids = [r["chunk_id"] for r in result]
    # "a" appears in both lists — should rank highest
    assert ids[0] == "a"
    assert set(ids) == {"a", "b", "c", "d", "e"}


def test_rrf_empty_inputs():
    assert reciprocal_rank_fusion([], []) == []


def test_rrf_scores_attached():
    result = reciprocal_rank_fusion([_hit("x")], [_hit("x")])
    assert "rrf_score" in result[0]
    assert result[0]["rrf_score"] > 0


def test_rrf_single_list_passthrough():
    hits = [_hit("a"), _hit("b")]
    result = reciprocal_rank_fusion(hits, [])
    ids = [r["chunk_id"] for r in result]
    assert ids == ["a", "b"]
