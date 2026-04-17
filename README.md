# Ask My Docs

A production-grade RAG (Retrieval-Augmented Generation) system that lets you upload documents and ask questions against them with auditable citations.

<img width="1882" height="861" alt="image" src="https://github.com/user-attachments/assets/393e8c6d-13da-4727-afc9-74e1f767d025" />

## Features

- **Hybrid retrieval** — BM25 lexical search + OpenAI vector search fused with Reciprocal Rank Fusion (RRF)
- **Reranking** — cross-encoder reranker (`ms-marco-MiniLM-L-6-v2`) for precision
- **Citation enforcement** — every answer includes `[SOURCE: <file>, chunk <id>]` references
- **Multi-format ingestion** — PDF, DOCX, TXT, MD
- **REST API** — FastAPI with `/ingest`, `/ask`, `/health` endpoints
- **Persistent index** — ChromaDB (vectors) + BM25 (lexical), both saved to disk

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | OpenAI `gpt-4.1-mini` (Responses API) |
| Embeddings | OpenAI `text-embedding-3-small` (1536-dim) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` (local) |
| Vector store | ChromaDB (persistent, cosine similarity) |
| Lexical index | BM25Okapi (`rank-bm25`) |
| API framework | FastAPI + Uvicorn |

## Project Structure

```
src/
├── api.py              # FastAPI app — /ingest, /ask, /health
├── pipeline.py         # Top-level RAG pipeline
├── config.py           # Settings (pydantic-settings, loaded from .env)
├── ingestion/
│   ├── chunker.py      # Recursive token-aware text splitter
│   ├── embedder.py     # OpenAI / sentence-transformers embedding wrapper
│   ├── indexer.py      # DualIndexer: ChromaDB + BM25
│   └── models.py       # Chunk dataclass
├── retrieval/
│   └── hybrid.py       # RRF fusion of BM25 + vector results
├── reranking/
│   └── cross_encoder.py
├── generation/
│   ├── generator.py    # AnswerGenerator (OpenAI Responses API)
│   └── prompts.py      # System prompt + citation template
└── evaluation/
    ├── harness.py      # Evaluation CI gate
    └── metrics.py      # Faithfulness, relevancy, citation coverage
data/
└── processed/chroma/   # Persistent ChromaDB + BM25 index
tests/                  # Pytest test suite
```

## Setup

### 1. Install dependencies

```bash
pip install -e .
```

### 2. Configure environment

Copy `.env` and fill in your OpenAI API key:

```env
OPENAI_API_KEY="sk-..."
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSIONS=1536
LLM_MODEL=gpt-4.1-mini
```

### 3. Run the server

```bash
python src/main.py
```

Server starts at `http://localhost:8000`. Interactive API docs at `http://localhost:8000/docs`.

## API Usage

### Upload documents

```bash
curl -X POST http://localhost:8000/ingest \
  -F "files=@document.pdf"
```

### Ask a question

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main topic of the document?"}'
```

### Health check

```bash
curl http://localhost:8000/health
```

## How It Works

1. **Ingest** — documents are chunked (400 tokens, 80-token overlap), embedded with `text-embedding-3-small`, and stored in ChromaDB. A BM25 index is also built and persisted to disk.
2. **Retrieve** — at query time, both BM25 and vector search run in parallel (top-20 each). Results are fused using RRF (k=60).
3. **Rerank** — the cross-encoder reranker scores the top candidates and returns the top-5.
4. **Generate** — the top chunks are passed as context to `gpt-4.1-mini` with a strict citation-enforcing system prompt. Answers without valid citations are replaced with a fallback message.

## Running Tests

```bash
python -m pytest tests/
```
