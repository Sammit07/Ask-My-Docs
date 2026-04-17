"""Integration tests for FastAPI endpoints (pipeline mocked)."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api import app
from src.generation.generator import AnswerResult


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def mock_pipeline():
    result = AnswerResult(
        answer="Mocked answer [SOURCE: doc.pdf, chunk abc].",
        cited_chunk_ids={"abc"},
        citation_coverage=1.0,
        retrieved_hits=[{"chunk_id": "abc", "text": "ctx", "source": "doc.pdf", "page": 1}],
    )
    mock = MagicMock()
    mock.ask.return_value = result
    mock.indexer.chunk_count = 5

    with patch("src.api.get_pipeline", return_value=mock):
        yield mock


def test_health_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["indexed_chunks"] == 5


def test_ask_returns_answer(client):
    resp = client.post("/ask", json={"question": "What is X?"})
    assert resp.status_code == 200
    data = resp.json()
    assert "answer" in data
    assert "citation_coverage" in data
    assert isinstance(data["sources"], list)


def test_ask_empty_question_rejected(client):
    resp = client.post("/ask", json={"question": "  "})
    assert resp.status_code == 422
