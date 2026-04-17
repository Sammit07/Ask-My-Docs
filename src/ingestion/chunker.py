"""Recursive character text splitter with token-aware sizing."""

import hashlib
import re
from pathlib import Path

import tiktoken

from src.ingestion.models import Chunk

_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def _count_tokens(text: str, enc: tiktoken.Encoding) -> int:
    return len(enc.encode(text))


def _split_text(text: str, chunk_size: int, overlap: int, enc: tiktoken.Encoding) -> list[str]:
    """Recursive split that respects token budget."""
    if _count_tokens(text, enc) <= chunk_size:
        return [text.strip()] if text.strip() else []

    for sep in _SEPARATORS:
        parts = text.split(sep) if sep else list(text)
        if len(parts) < 2:
            continue

        chunks: list[str] = []
        current = ""
        for part in parts:
            candidate = (current + sep + part).lstrip() if current else part
            if _count_tokens(candidate, enc) <= chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current.strip())
                # handle overlap: carry last `overlap` tokens into next chunk
                if overlap > 0 and current:
                    words = current.split()
                    carry = ""
                    for w in reversed(words):
                        test = (w + " " + carry).strip()
                        if _count_tokens(test, enc) <= overlap:
                            carry = test
                        else:
                            break
                    current = (carry + sep + part).strip() if carry else part
                else:
                    current = part
        if current:
            chunks.append(current.strip())
        return [c for c in chunks if c]

    return [text.strip()]


def _make_id(doc_id: str, index: int) -> str:
    raw = f"{doc_id}::{index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


class DocumentChunker:
    def __init__(self, chunk_size: int = 400, overlap: int = 80, model: str = "cl100k_base"):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self._enc = tiktoken.get_encoding(model)

    def chunk_text(self, text: str, doc_id: str, source: str, page: int | None = None) -> list[Chunk]:
        raw_chunks = _split_text(text, self.chunk_size, self.overlap, self._enc)
        return [
            Chunk(
                chunk_id=_make_id(doc_id, i),
                doc_id=doc_id,
                source=source,
                page=page,
                text=c,
                token_count=_count_tokens(c, self._enc),
            )
            for i, c in enumerate(raw_chunks)
        ]

    def chunk_file(self, path: Path) -> list[Chunk]:
        suffix = path.suffix.lower()
        doc_id = path.stem

        if suffix == ".txt" or suffix == ".md":
            return self._chunk_plaintext(path, doc_id)
        if suffix == ".pdf":
            return self._chunk_pdf(path, doc_id)
        if suffix in {".docx", ".doc"}:
            return self._chunk_docx(path, doc_id)
        raise ValueError(f"Unsupported file type: {suffix}")

    def _chunk_plaintext(self, path: Path, doc_id: str) -> list[Chunk]:
        text = path.read_text(encoding="utf-8", errors="replace")
        return self.chunk_text(text, doc_id, path.name)

    def _chunk_pdf(self, path: Path, doc_id: str) -> list[Chunk]:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        chunks: list[Chunk] = []
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            text = re.sub(r"\s+", " ", text).strip()
            if text:
                chunks.extend(self.chunk_text(text, f"{doc_id}_p{page_num}", path.name, page=page_num))
        return chunks

    def _chunk_docx(self, path: Path, doc_id: str) -> list[Chunk]:
        from docx import Document

        doc = Document(str(path))
        text = "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
        return self.chunk_text(text, doc_id, path.name)
