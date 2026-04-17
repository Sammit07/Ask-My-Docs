"""Unit tests for DualIndexer (ChromaDB mocked)."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.ingestion.models import Chunk


def _make_chunk(chunk_id: str, text: str = "hello world") -> Chunk:
    return Chunk(chunk_id=chunk_id, doc_id="doc1", source="f.txt", page=1, text=text, token_count=2)


@patch("src.ingestion.indexer.chromadb")
def test_add_and_bm25_search(mock_chroma, tmp_path):
    from src.ingestion.indexer import DualIndexer

    mock_collection = MagicMock()
    mock_collection.count.return_value = 2
    mock_client = MagicMock()
    mock_client.get_or_create_collection.return_value = mock_collection
    mock_chroma.PersistentClient.return_value = mock_client

    embedder = MagicMock()
    embedder.embed_chunks.return_value = np.zeros((2, 384))

    indexer = DualIndexer(tmp_path, "test_col", embedder)
    # 3 docs required: BM25 IDF is 0 when a term appears in half of a 2-doc corpus
    chunks = [
        _make_chunk("c1", "python programming tutorial"),
        _make_chunk("c2", "java enterprise development"),
        _make_chunk("c3", "sql database queries"),
    ]
    embedder.embed_chunks.return_value = np.zeros((3, 384))
    indexer.add_chunks(chunks)

    assert indexer.chunk_count == 3
    results = indexer.bm25_search("python", top_k=5)
    assert any(r["chunk_id"] == "c1" for r in results)


@patch("src.ingestion.indexer.chromadb")
def test_vector_search(mock_chroma, tmp_path):
    from src.ingestion.indexer import DualIndexer

    mock_collection = MagicMock()
    mock_collection.count.return_value = 1
    mock_collection.query.return_value = {
        "ids": [["c1"]],
        "documents": [["some text"]],
        "metadatas": [[{"doc_id": "doc1", "source": "f.txt", "page": 1, "token_count": 2}]],
        "distances": [[0.1]],
    }
    mock_client = MagicMock()
    mock_client.get_or_create_collection.return_value = mock_collection
    mock_chroma.PersistentClient.return_value = mock_client

    embedder = MagicMock()
    embedder.embed_chunks.return_value = np.zeros((1, 384))

    indexer = DualIndexer(tmp_path, "test_col", embedder)
    indexer.add_chunks([_make_chunk("c1", "some text")])

    query_emb = np.zeros(384)
    results = indexer.vector_search(query_emb, top_k=5)
    assert len(results) == 1
    assert results[0]["chunk_id"] == "c1"
    assert results[0]["score"] == pytest.approx(0.9)


@patch("src.ingestion.indexer.chromadb")
def test_bm25_search_empty_index(mock_chroma, tmp_path):
    from src.ingestion.indexer import DualIndexer

    mock_collection = MagicMock()
    mock_client = MagicMock()
    mock_client.get_or_create_collection.return_value = mock_collection
    mock_chroma.PersistentClient.return_value = mock_client

    embedder = MagicMock()
    indexer = DualIndexer(tmp_path, "test_col", embedder)
    assert indexer.bm25_search("anything", top_k=5) == []


@patch("src.ingestion.indexer.chromadb")
def test_add_empty_chunks_noop(mock_chroma, tmp_path):
    from src.ingestion.indexer import DualIndexer

    mock_collection = MagicMock()
    mock_client = MagicMock()
    mock_client.get_or_create_collection.return_value = mock_collection
    mock_chroma.PersistentClient.return_value = mock_client

    embedder = MagicMock()
    indexer = DualIndexer(tmp_path, "test_col", embedder)
    indexer.add_chunks([])
    assert indexer.chunk_count == 0
