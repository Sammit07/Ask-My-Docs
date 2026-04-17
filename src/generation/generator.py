"""OpenAI-backed answer generator with strict citation enforcement."""

from __future__ import annotations

import re

import structlog
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from src.generation.prompts import SYSTEM_PROMPT, build_prompt

logger = structlog.get_logger(__name__)

_CITATION_RE = re.compile(r"\[SOURCE:\s*[^\]]+,\s*chunk\s+\w+\]")
_FALLBACK_ANSWER = "I cannot answer this question from the provided documents."


def _has_citations(text: str) -> bool:
    return bool(_CITATION_RE.search(text))


def _extract_cited_chunk_ids(text: str) -> set[str]:
    chunk_ids = set()
    for match in re.finditer(r"chunk\s+(\w+)", text):
        chunk_ids.add(match.group(1))
    return chunk_ids


def _is_citation_free_fallback(text: str) -> bool:
    return text.strip() == _FALLBACK_ANSWER


def _answer_has_valid_citations(answer: str, available_chunk_ids: set[str]) -> bool:
    if _is_citation_free_fallback(answer):
        return True

    cited_ids = _extract_cited_chunk_ids(answer)
    return bool(cited_ids) and cited_ids.issubset(available_chunk_ids)


class AnswerGenerator:
    def __init__(
        self,
        api_key: str,
        model_name: str = "gpt-4.1-mini",
        temperature: float = 0.0,
        max_output_tokens: int = 2048,
    ):
        self._client = OpenAI(api_key=api_key)
        self._model_name = model_name
        self._temperature = temperature
        self._max_output_tokens = max_output_tokens

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def generate(self, question: str, hits: list[dict]) -> AnswerResult:
        prompt = build_prompt(question, hits)
        response = self._client.responses.create(
            model=self._model_name,
            input=[
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": SYSTEM_PROMPT}],
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": prompt}],
                },
            ],
            temperature=self._temperature,
            max_output_tokens=self._max_output_tokens,
        )
        answer_text = (response.output_text or "").strip()

        available_ids = {h["chunk_id"] for h in hits}
        if not _answer_has_valid_citations(answer_text, available_ids):
            logger.warning("answer_failed_citation_validation", question=question[:80])
            answer_text = _FALLBACK_ANSWER

        cited_ids = _extract_cited_chunk_ids(answer_text)
        citation_coverage = len(cited_ids & available_ids) / max(len(hits), 1)

        if not _has_citations(answer_text) and not _is_citation_free_fallback(answer_text):
            logger.warning("answer_missing_citations", question=question[:80])

        return AnswerResult(
            answer=answer_text,
            cited_chunk_ids=cited_ids,
            citation_coverage=citation_coverage,
            retrieved_hits=hits,
        )


class AnswerResult:
    def __init__(
        self,
        answer: str,
        cited_chunk_ids: set[str],
        citation_coverage: float,
        retrieved_hits: list[dict],
    ):
        self.answer = answer
        self.cited_chunk_ids = cited_chunk_ids
        self.citation_coverage = citation_coverage
        self.retrieved_hits = retrieved_hits

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "cited_chunk_ids": list(self.cited_chunk_ids),
            "citation_coverage": self.citation_coverage,
            "sources": [
                {
                    "chunk_id": h["chunk_id"],
                    "source": h.get("source"),
                    "page": h.get("page"),
                    "text_preview": h["text"][:200],
                }
                for h in self.retrieved_hits
            ],
        }
