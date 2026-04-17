"""Sentence-transformers embedding wrapper with batched encoding."""

from __future__ import annotations

import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from src.ingestion.models import Chunk


class Embedder:
    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        batch_size: int = 64,
        api_key: str | None = None,
        dimensions: int | None = 1536,
    ):
        self._model_name = model_name
        self._batch_size = batch_size
        self._dimensions = dimensions
        self._client: OpenAI | None = None
        self._model: SentenceTransformer | None = None

        if model_name.startswith("text-embedding-"):
            self._client = OpenAI(api_key=api_key)
        else:
            self._model = SentenceTransformer(model_name)

    @property
    def dimension(self) -> int:
        if self._client is not None:
            if self._dimensions is None:
                raise ValueError("embedding_dimensions must be set for OpenAI embedding models.")
            return self._dimensions

        if self._model is None:
            raise RuntimeError("No embedding backend has been initialized.")
        return self._model.get_sentence_embedding_dimension()  # type: ignore[return-value]

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if self._client is not None:
            vectors: list[list[float]] = []
            for start in range(0, len(texts), self._batch_size):
                batch = texts[start : start + self._batch_size]
                kwargs: dict[str, object] = {
                    "model": self._model_name,
                    "input": batch,
                    "encoding_format": "float",
                }
                if self._dimensions is not None:
                    kwargs["dimensions"] = self._dimensions
                response = self._client.embeddings.create(**kwargs)
                vectors.extend(item.embedding for item in response.data)

            embeddings = np.asarray(vectors, dtype=np.float32)
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            return embeddings / np.clip(norms, a_min=1e-12, a_max=None)

        if self._model is None:
            raise RuntimeError("No embedding backend has been initialized.")
        return self._model.encode(  # type: ignore[return-value]
            texts,
            batch_size=self._batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )

    def embed_chunks(self, chunks: list[Chunk]) -> np.ndarray:
        return self.embed_texts([c.text for c in chunks])

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_texts([query])[0]
