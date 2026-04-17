"""FastAPI application: /ingest, /ask, /health endpoints."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Annotated

import structlog
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.ingestion.chunker import DocumentChunker
from src.pipeline import RAGPipeline

_STATIC_DIR = Path(__file__).parent / "static"

logger = structlog.get_logger(__name__)

app = FastAPI(
    title="Ask My Docs",
    description="Production RAG API with hybrid retrieval, reranking, and citation enforcement",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

_pipeline: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class AskRequest(BaseModel):
    question: str


class SourceRef(BaseModel):
    chunk_id: str
    source: str | None
    page: int | None
    text_preview: str


class AskResponse(BaseModel):
    answer: str
    cited_chunk_ids: list[str]
    citation_coverage: float
    sources: list[SourceRef]


class IngestResponse(BaseModel):
    message: str
    chunks_added: int
    total_chunks: int


class HealthResponse(BaseModel):
    status: str
    indexed_chunks: int


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/", include_in_schema=False)
def root() -> FileResponse:
    return FileResponse(str(_STATIC_DIR / "index.html"))


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    pipeline = get_pipeline()
    return HealthResponse(status="ok", indexed_chunks=pipeline.indexer.chunk_count)


@app.post("/ingest", response_model=IngestResponse)
async def ingest(files: Annotated[list[UploadFile], File(...)]) -> IngestResponse:
    """Upload one or more documents (PDF, DOCX, TXT, MD) to be indexed."""
    pipeline = get_pipeline()
    chunker = DocumentChunker()
    before = pipeline.indexer.chunk_count
    supported = {".pdf", ".docx", ".doc", ".txt", ".md"}

    with tempfile.TemporaryDirectory() as tmp:
        for upload in files:
            suffix = Path(upload.filename or "file").suffix.lower()
            if suffix not in supported:
                raise HTTPException(
                    status_code=422,
                    detail=f"Unsupported file type '{suffix}'. Allowed: {sorted(supported)}",
                )
            dest = Path(tmp) / (upload.filename or "upload" + suffix)
            with open(dest, "wb") as f:
                shutil.copyfileobj(upload.file, f)

            chunks = chunker.chunk_file(dest)
            pipeline.indexer.add_chunks(chunks)
            logger.info("ingested_file", file=upload.filename, chunks=len(chunks))

    after = pipeline.indexer.chunk_count
    added = after - before
    return IngestResponse(
        message=f"Successfully ingested {len(files)} file(s).",
        chunks_added=added,
        total_chunks=after,
    )


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    if not request.question.strip():
        raise HTTPException(status_code=422, detail="Question cannot be empty.")

    pipeline = get_pipeline()
    result = pipeline.ask(request.question)
    data = result.to_dict()

    return AskResponse(
        answer=data["answer"],
        cited_chunk_ids=data["cited_chunk_ids"],
        citation_coverage=data["citation_coverage"],
        sources=[SourceRef(**s) for s in data["sources"]],
    )
