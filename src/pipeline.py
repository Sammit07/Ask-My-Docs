"""Top-level RAG pipeline: retrieval -> reranking -> generation."""

from __future__ import annotations

import structlog

from src.config import Settings, get_settings
from src.generation.generator import AnswerGenerator, AnswerResult
from src.ingestion.embedder import Embedder
from src.ingestion.indexer import DualIndexer
from src.reranking.cross_encoder import Reranker
from src.retrieval.hybrid import HybridRetriever

logger = structlog.get_logger(__name__)


class RAGPipeline:
    def __init__(self, settings: Settings | None = None):
        cfg = settings or get_settings()

        cfg.chroma_persist_dir.mkdir(parents=True, exist_ok=True)

        self._embedder = Embedder(
            model_name=cfg.embedding_model,
            api_key=cfg.openai_api_key,
            dimensions=cfg.embedding_dimensions,
        )
        self._indexer = DualIndexer(
            persist_dir=cfg.chroma_persist_dir,
            collection_name=cfg.chroma_collection,
            embedder=self._embedder,
        )
        self._retriever = HybridRetriever(
            indexer=self._indexer,
            embedder=self._embedder,
            bm25_top_k=cfg.bm25_top_k,
            vector_top_k=cfg.vector_top_k,
            rrf_k=cfg.rrf_k,
        )
        self._reranker = Reranker(
            model_name=cfg.reranker_model,
            top_k=cfg.rerank_top_k,
        )
        self._generator = AnswerGenerator(
            api_key=cfg.openai_api_key,
            model_name=cfg.llm_model,
            temperature=cfg.temperature,
        )

    def ask(self, question: str) -> AnswerResult:
        log = logger.bind(question=question[:100])

        if self._indexer.chunk_count == 0:
            log.warning("index_empty")
            return AnswerResult(
                answer="No documents have been indexed yet. Please ingest documents first.",
                cited_chunk_ids=set(),
                citation_coverage=0.0,
                retrieved_hits=[],
            )

        log.info("retrieving")
        candidates = self._retriever.retrieve(question)

        log.info("reranking", candidates=len(candidates))
        hits = self._reranker.rerank(question, candidates)

        log.info("generating", hits=len(hits))
        result = self._generator.generate(question, hits)

        log.info(
            "pipeline_complete",
            citation_coverage=result.citation_coverage,
            cited_chunks=len(result.cited_chunk_ids),
        )
        return result

    @property
    def indexer(self) -> DualIndexer:
        return self._indexer
