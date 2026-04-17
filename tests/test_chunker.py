"""Unit tests for the document chunker."""

import pytest
import tiktoken

from src.ingestion.chunker import DocumentChunker


@pytest.fixture
def chunker() -> DocumentChunker:
    return DocumentChunker(chunk_size=50, overlap=10)


@pytest.fixture
def enc():
    return tiktoken.get_encoding("cl100k_base")


def test_short_text_returns_single_chunk(chunker):
    chunks = chunker.chunk_text("Hello world.", doc_id="doc1", source="test.txt")
    assert len(chunks) == 1
    assert chunks[0].text == "Hello world."


def test_chunk_ids_are_unique(chunker):
    text = " ".join(["word"] * 300)
    chunks = chunker.chunk_text(text, doc_id="doc1", source="test.txt")
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_chunks_have_doc_id(chunker):
    chunks = chunker.chunk_text("Some text.", doc_id="my_doc", source="file.txt")
    for c in chunks:
        assert c.doc_id == "my_doc"


def test_token_count_within_budget(chunker, enc):
    text = " ".join(["longword"] * 200)
    chunks = chunker.chunk_text(text, doc_id="d", source="f.txt")
    for c in chunks:
        assert c.token_count <= 60  # chunk_size=50 + small tolerance


def test_empty_text_returns_no_chunks(chunker):
    chunks = chunker.chunk_text("   ", doc_id="d", source="f.txt")
    assert chunks == []
