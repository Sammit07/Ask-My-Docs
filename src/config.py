from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # OpenAI
    openai_api_key: str = Field(..., description="OpenAI API key")

    # Models
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    llm_model: str = "gpt-4.1-mini"
    eval_model: str = "gpt-4.1-mini"

    # ChromaDB
    chroma_persist_dir: Path = Path("./data/processed/chroma")
    chroma_collection: str = "ask_my_docs"

    # Retrieval
    bm25_top_k: int = 20
    vector_top_k: int = 20
    rerank_top_k: int = 5
    rrf_k: int = 60  # RRF constant — 60 is empirically robust across benchmarks

    # Generation
    max_context_tokens: int = 8000
    temperature: float = 0.0

    # Evaluation thresholds
    eval_dataset_path: Path = Path("./data/eval_dataset.jsonl")
    faithfulness_threshold: float = 0.80
    answer_relevancy_threshold: float = 0.75
    citation_coverage_threshold: float = 0.90


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
