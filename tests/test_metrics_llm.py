"""Unit tests for LLM-backed metrics (client mocked)."""

from unittest.mock import MagicMock

import pytest

from src.evaluation.metrics import _llm_binary, answer_relevancy, faithfulness


def _mock_client(response_text: str) -> MagicMock:
    mock_resp = MagicMock()
    mock_resp.output_text = response_text
    client = MagicMock()
    client.responses.create.return_value = mock_resp
    return client


def test_llm_binary_parses_float():
    client = _mock_client("0.85")
    score = _llm_binary("some prompt", client, "gpt-4.1-mini")
    assert score == pytest.approx(0.85)


def test_llm_binary_returns_zero_on_unparseable():
    client = _mock_client("I cannot determine this.")
    score = _llm_binary("prompt", client, "gpt-4.1-mini")
    assert score == pytest.approx(0.0)


def test_faithfulness_calls_client():
    client = _mock_client("0.9")
    score = faithfulness("answer text", ["ctx1", "ctx2"], client, "gpt-4.1-mini")
    assert score == pytest.approx(0.9)
    client.responses.create.assert_called_once()


def test_answer_relevancy_calls_client():
    client = _mock_client("0.75")
    score = answer_relevancy("question?", "answer", client, "gpt-4.1-mini")
    assert score == pytest.approx(0.75)
