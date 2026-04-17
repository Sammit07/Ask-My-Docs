"""Unit tests for evaluation metrics (non-LLM parts)."""

import pytest

from src.evaluation.metrics import citation_coverage, rouge_scores


def _hit(chunk_id: str) -> dict:
    return {"chunk_id": chunk_id, "text": "sample text"}


def test_citation_coverage_full():
    answer = "Answer [SOURCE: doc.pdf, chunk abc123] more text [SOURCE: doc.pdf, chunk def456]"
    hits = [_hit("abc123"), _hit("def456")]
    assert citation_coverage(answer, hits) == pytest.approx(1.0)


def test_citation_coverage_partial():
    answer = "Only this [SOURCE: doc.pdf, chunk abc123] was cited."
    hits = [_hit("abc123"), _hit("def456")]
    assert citation_coverage(answer, hits) == pytest.approx(0.5)


def test_citation_coverage_none():
    answer = "No citations here at all."
    hits = [_hit("abc123")]
    assert citation_coverage(answer, hits) == pytest.approx(0.0)


def test_citation_coverage_empty_hits():
    assert citation_coverage("anything", []) == pytest.approx(1.0)


def test_rouge_perfect_match():
    scores = rouge_scores("hello world", "hello world")
    assert scores["rouge1_f"] == pytest.approx(1.0)
    assert scores["rougeL_f"] == pytest.approx(1.0)


def test_rouge_no_overlap():
    scores = rouge_scores("foo bar", "baz qux")
    assert scores["rouge1_f"] == pytest.approx(0.0)
