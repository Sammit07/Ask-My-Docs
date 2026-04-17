from __future__ import annotations

from typing import Any

SYSTEM_PROMPT = """You are a precise document assistant. Answer the user's question using ONLY the provided context passages.

Rules:
1. Every factual claim MUST be followed by a citation in the format [SOURCE: <source>, chunk <chunk_id>].
2. If the context does not contain enough information to answer, respond: "I cannot answer this question from the provided documents."
3. Do NOT use prior knowledge - only cite from the context.
4. Be concise and structured. Use bullet points for multi-part answers.
5. If multiple passages support a claim, cite all of them.
6. Never cite a chunk id that is not present in the provided context."""

CONTEXT_TEMPLATE = """--- Context Passages ---
{passages}
--- End Context ---

Question: {question}

Answer (with citations):"""


def build_context_block(hits: list[dict[str, Any]]) -> str:
    blocks = []
    for hit in hits:
        source = hit.get("source", "unknown")
        chunk_id = hit.get("chunk_id", "?")
        page = hit.get("page")
        page_str = f", page {page}" if page and page > 0 else ""
        blocks.append(f"[SOURCE: {source}{page_str}, chunk {chunk_id}]\n{hit['text']}")
    return "\n\n".join(blocks)


def build_prompt(question: str, hits: list[dict[str, Any]]) -> str:
    passages = build_context_block(hits)
    return CONTEXT_TEMPLATE.format(passages=passages, question=question)
