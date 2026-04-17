"""Dual-index writer: ChromaDB (vector) + BM25 (lexical), persisted to disk."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import chromadb
import numpy as np
from rank_bm25 import BM25Okapi

from src.ingestion.chunker import DocumentChunker
from src.ingestion.embedder import Embedder
from src.ingestion.models import Chunk


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


class DualIndexer:
    def __init__(
        self,
        persist_dir: Path,
        collection_name: str,
        embedder: Embedder,
    ):
        persist_dir.mkdir(parents=True, exist_ok=True)
        self._persist_dir = persist_dir
        self._bm25_path = persist_dir / "bm25_index.pkl"
        self._chunks_path = persist_dir / "chunks.jsonl"

        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._embedder = embedder

        # BM25 state (rebuilt from chunks on load)
        self._chunks: list[Chunk] = []
        self._bm25: BM25Okapi | None = None
        self._load_bm25()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_chunks(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        # deduplicate within batch before upserting
        seen: dict[str, Chunk] = {}
        for c in chunks:
            seen.setdefault(c.chunk_id, c)
        chunks = list(seen.values())
        embeddings = self._embedder.embed_chunks(chunks)
        self._collection.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings.tolist(),
            documents=[c.text for c in chunks],
            metadatas=[
                {
                    "doc_id": c.doc_id,
                    "source": c.source,
                    "page": c.page or -1,
                    "token_count": c.token_count,
                    **c.metadata,
                }
                for c in chunks
            ],
        )

        self._chunks.extend(chunks)
        self._rebuild_bm25()
        self._save_bm25()
        self._append_chunks_to_disk(chunks)

    def vector_search(self, query_embedding: np.ndarray, top_k: int) -> list[dict]:
        results = self._collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(top_k, self._collection.count() or 1),
            include=["documents", "metadatas", "distances"],
        )
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0]
        ids = results["ids"][0]
        return [
            {
                "chunk_id": ids[i],
                "text": docs[i],
                "score": 1.0 - distances[i],  # cosine similarity
                **metas[i],
            }
            for i in range(len(docs))
        ]

    def bm25_search(self, query: str, top_k: int) -> list[dict]:
        if not self._bm25 or not self._chunks:
            return []
        tokens = _tokenize(query)
        scores = self._bm25.get_scores(tokens)
        top_indices = scores.argsort()[::-1][:top_k]
        return [
            {
                "chunk_id": self._chunks[i].chunk_id,
                "text": self._chunks[i].text,
                "score": float(scores[i]),
                "doc_id": self._chunks[i].doc_id,
                "source": self._chunks[i].source,
                "page": self._chunks[i].page,
                **self._chunks[i].metadata,
            }
            for i in top_indices
            if scores[i] > 0
        ]

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _rebuild_bm25(self) -> None:
        corpus = [_tokenize(c.text) for c in self._chunks]
        self._bm25 = BM25Okapi(corpus)

    def _save_bm25(self) -> None:
        with open(self._bm25_path, "wb") as f:
            pickle.dump((self._bm25, self._chunks), f)

    def _load_bm25(self) -> None:
        if self._bm25_path.exists():
            with open(self._bm25_path, "rb") as f:
                self._bm25, self._chunks = pickle.load(f)

    def _append_chunks_to_disk(self, chunks: list[Chunk]) -> None:
        with open(self._chunks_path, "a", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(c.__dict__) + "\n")


def build_index_from_files(
    paths: list[Path],
    persist_dir: Path,
    collection_name: str = "ask_my_docs",
    chunk_size: int = 400,
    overlap: int = 80,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> DualIndexer:
    embedder = Embedder(embedding_model)
    indexer = DualIndexer(persist_dir, collection_name, embedder)
    chunker = DocumentChunker(chunk_size, overlap)

    for path in paths:
        chunks = chunker.chunk_file(path)
        indexer.add_chunks(chunks)

    return indexer
