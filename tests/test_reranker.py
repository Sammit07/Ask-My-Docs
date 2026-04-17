"""Unit tests for the cross-encoder reranker (model mocked)."""

from unittest.mock import MagicMock, patch

import pytest

from src.reranking.cross_encoder import Reranker


def _hit(chunk_id: str, text: str = "some context") -> dict:
    return {"chunk_id": chunk_id, "text": text, "source": "doc.pdf", "page": 1, "rrf_score": 0.1}


@patch("src.reranking.cross_encoder.CrossEncoder")
def test_reranker_returns_top_k(mock_ce_cls):
    mock_model = MagicMock()
    mock_model.predict.return_value = [0.9, 0.3, 0.7, 0.5, 0.1]
    mock_ce_cls.return_value = mock_model

    reranker = Reranker(top_k=3)
    hits = [_hit(f"c{i}") for i in range(5)]
    result = reranker.rerank("query", hits)

    assert len(result) == 3
    # highest score first
    assert result[0]["rerank_score"] >= result[1]["rerank_score"]


@patch("src.reranking.cross_encoder.CrossEncoder")
def test_reranker_attaches_score(mock_ce_cls):
    mock_model = MagicMock()
    mock_model.predict.return_value = [0.8]
    mock_ce_cls.return_value = mock_model

    reranker = Reranker(top_k=1)
    result = reranker.rerank("q", [_hit("x")])
    assert "rerank_score" in result[0]
    assert result[0]["rerank_score"] == pytest.approx(0.8)


@patch("src.reranking.cross_encoder.CrossEncoder")
def test_reranker_empty_hits(mock_ce_cls):
    mock_ce_cls.return_value = MagicMock()
    reranker = Reranker(top_k=5)
    assert reranker.rerank("q", []) == []
