from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.ingestion.embedder import Embedder
from src.ingestion.models import Chunk


@patch("src.ingestion.embedder.OpenAI")
def test_openai_embedder_normalizes_embeddings(mock_openai):
    mock_client = MagicMock()
    mock_client.embeddings.create.return_value = SimpleNamespace(
        data=[
            SimpleNamespace(embedding=[3.0, 4.0]),
            SimpleNamespace(embedding=[0.0, 5.0]),
        ]
    )
    mock_openai.return_value = mock_client

    embedder = Embedder(
        model_name="text-embedding-3-small", api_key="fake", batch_size=2, dimensions=2
    )
    vectors = embedder.embed_texts(["alpha", "beta"])

    assert vectors.shape == (2, 2)
    assert np.allclose(np.linalg.norm(vectors, axis=1), np.array([1.0, 1.0]))
    assert embedder.dimension == 2


@patch("src.ingestion.embedder.SentenceTransformer")
def test_sentence_transformer_backend_is_supported(mock_sentence_transformer):
    mock_model = MagicMock()
    mock_model.get_sentence_embedding_dimension.return_value = 384
    mock_model.encode.return_value = np.array([[0.5, 0.5]], dtype=np.float32)
    mock_sentence_transformer.return_value = mock_model

    embedder = Embedder(model_name="sentence-transformers/all-MiniLM-L6-v2")

    assert embedder.dimension == 384
    assert np.allclose(embedder.embed_query("hello"), np.array([0.5, 0.5], dtype=np.float32))

    chunks = [
        Chunk(
            chunk_id="c1",
            doc_id="d1",
            source="doc.txt",
            text="chunk body",
            page=None,
            token_count=2,
            metadata={},
        )
    ]
    chunk_vectors = embedder.embed_chunks(chunks)
    assert chunk_vectors.shape == (1, 2)


@patch("src.ingestion.embedder.OpenAI")
def test_openai_dimension_requires_dimensions_setting(mock_openai):
    mock_openai.return_value = MagicMock()
    embedder = Embedder(model_name="text-embedding-3-small", api_key="fake", dimensions=None)

    with pytest.raises(ValueError):
        _ = embedder.dimension
