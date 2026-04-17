"""Unit tests for the answer generator (OpenAI mocked)."""

from unittest.mock import MagicMock, patch

from src.generation.generator import AnswerGenerator, AnswerResult, _extract_cited_chunk_ids, _has_citations
from src.generation.prompts import build_context_block, build_prompt


def test_has_citations_true():
    text = "The answer is X [SOURCE: doc.pdf, chunk abc123]."
    assert _has_citations(text) is True


def test_has_citations_false():
    text = "No citation here."
    assert _has_citations(text) is False


def test_extract_cited_chunk_ids():
    text = "Fact A [SOURCE: a.pdf, chunk id1]. Fact B [SOURCE: b.pdf, chunk id2]."
    ids = _extract_cited_chunk_ids(text)
    assert ids == {"id1", "id2"}


def test_build_context_block():
    hits = [{"chunk_id": "abc", "text": "Hello world", "source": "test.pdf", "page": 1}]
    block = build_context_block(hits)
    assert "SOURCE: test.pdf, page 1, chunk abc" in block
    assert "Hello world" in block


def test_build_context_block_no_page():
    hits = [{"chunk_id": "abc", "text": "Hello world", "source": "test.pdf", "page": None}]
    block = build_context_block(hits)
    assert "page" not in block


def test_build_prompt_contains_question():
    hits = [{"chunk_id": "x", "text": "context", "source": "f.pdf", "page": None}]
    prompt = build_prompt("What is X?", hits)
    assert "What is X?" in prompt


def test_answer_result_to_dict():
    result = AnswerResult(
        answer="Test [SOURCE: a.pdf, chunk c1].",
        cited_chunk_ids={"c1"},
        citation_coverage=1.0,
        retrieved_hits=[{"chunk_id": "c1", "text": "ctx", "source": "a.pdf", "page": 2}],
    )
    d = result.to_dict()
    assert d["answer"] == "Test [SOURCE: a.pdf, chunk c1]."
    assert "c1" in d["cited_chunk_ids"]
    assert d["citation_coverage"] == 1.0
    assert len(d["sources"]) == 1
    assert d["sources"][0]["chunk_id"] == "c1"


@patch("src.generation.generator.OpenAI")
def test_generator_returns_answer_result(mock_openai):
    mock_response = MagicMock()
    mock_response.output_text = "Answer text [SOURCE: doc.pdf, chunk chunk1]."
    mock_client = MagicMock()
    mock_client.responses.create.return_value = mock_response
    mock_openai.return_value = mock_client

    gen = AnswerGenerator(api_key="fake-key")
    hits = [{"chunk_id": "chunk1", "text": "some context", "source": "doc.pdf", "page": 1}]
    result = gen.generate("What is it?", hits)

    assert "chunk1" in result.cited_chunk_ids
    assert result.citation_coverage > 0


@patch("src.generation.generator.OpenAI")
def test_generator_logs_warning_when_no_citations(mock_openai):
    mock_response = MagicMock()
    mock_response.output_text = "An answer with no citations at all."
    mock_client = MagicMock()
    mock_client.responses.create.return_value = mock_response
    mock_openai.return_value = mock_client

    gen = AnswerGenerator(api_key="fake-key")
    hits = [{"chunk_id": "c1", "text": "ctx", "source": "f.pdf", "page": None}]
    result = gen.generate("Q?", hits)
    assert result.answer == "I cannot answer this question from the provided documents."
    assert result.citation_coverage == 0.0
