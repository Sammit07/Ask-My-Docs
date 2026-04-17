"""Unit tests for the RAG pipeline orchestrator."""

from unittest.mock import MagicMock, patch

import pytest

from src.generation.generator import AnswerResult


@patch("src.pipeline.Embedder")
@patch("src.pipeline.DualIndexer")
@patch("src.pipeline.HybridRetriever")
@patch("src.pipeline.Reranker")
@patch("src.pipeline.AnswerGenerator")
def test_pipeline_ask_full_flow(mock_gen_cls, mock_reranker_cls, mock_retriever_cls, mock_indexer_cls, mock_embedder_cls, tmp_path, monkeypatch):
    from src.pipeline import RAGPipeline
    from src.config import Settings

    monkeypatch.setenv("OPENAI_API_KEY", "fake")

    cfg = Settings(
        openai_api_key="fake",
        chroma_persist_dir=tmp_path / "chroma",
    )

    # Wire mocks
    mock_indexer = MagicMock()
    mock_indexer.chunk_count = 3
    mock_indexer_cls.return_value = mock_indexer

    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [
        {"chunk_id": "c1", "text": "ctx", "source": "f.pdf", "page": 1, "rrf_score": 0.5}
    ]
    mock_retriever_cls.return_value = mock_retriever

    mock_reranker = MagicMock()
    mock_reranker.rerank.return_value = [
        {"chunk_id": "c1", "text": "ctx", "source": "f.pdf", "page": 1, "rerank_score": 0.9}
    ]
    mock_reranker_cls.return_value = mock_reranker

    expected_result = AnswerResult(
        answer="Answer [SOURCE: f.pdf, chunk c1].",
        cited_chunk_ids={"c1"},
        citation_coverage=1.0,
        retrieved_hits=[{"chunk_id": "c1", "text": "ctx", "source": "f.pdf", "page": 1}],
    )
    mock_gen = MagicMock()
    mock_gen.generate.return_value = expected_result
    mock_gen_cls.return_value = mock_gen

    pipeline = RAGPipeline(cfg)
    result = pipeline.ask("What is X?")

    assert result.answer == "Answer [SOURCE: f.pdf, chunk c1]."
    mock_retriever.retrieve.assert_called_once_with("What is X?")
    mock_reranker.rerank.assert_called_once()
    mock_gen.generate.assert_called_once()


@patch("src.pipeline.Embedder")
@patch("src.pipeline.DualIndexer")
@patch("src.pipeline.HybridRetriever")
@patch("src.pipeline.Reranker")
@patch("src.pipeline.AnswerGenerator")
def test_pipeline_returns_no_docs_message_when_empty(mock_gen_cls, mock_reranker_cls, mock_retriever_cls, mock_indexer_cls, mock_embedder_cls, tmp_path, monkeypatch):
    from src.pipeline import RAGPipeline
    from src.config import Settings

    monkeypatch.setenv("OPENAI_API_KEY", "fake")
    cfg = Settings(openai_api_key="fake", chroma_persist_dir=tmp_path / "chroma")

    mock_indexer = MagicMock()
    mock_indexer.chunk_count = 0
    mock_indexer_cls.return_value = mock_indexer

    pipeline = RAGPipeline(cfg)
    result = pipeline.ask("Any question?")

    assert "No documents" in result.answer
    assert result.citation_coverage == 0.0
