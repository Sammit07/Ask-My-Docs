"""Evaluation metrics: faithfulness, answer relevancy, citation coverage, ROUGE."""

from __future__ import annotations

import re

import structlog
from openai import OpenAI
from rouge_score import rouge_scorer

logger = structlog.get_logger(__name__)

_ROUGE = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
_CITATION_RE = re.compile(r"\[SOURCE:\s*[^\]]+,\s*chunk\s+\w+\]")


def citation_coverage(answer: str, retrieved_hits: list[dict]) -> float:
    """Fraction of retrieved chunks that are cited in the answer."""
    if not retrieved_hits:
        return 1.0
    cited_ids: set[str] = set()
    for match in re.finditer(r"chunk\s+(\w+)", answer):
        cited_ids.add(match.group(1))
    available_ids = {h["chunk_id"] for h in retrieved_hits}
    return len(cited_ids & available_ids) / len(available_ids)


def rouge_scores(answer: str, reference: str) -> dict[str, float]:
    scores = _ROUGE.score(reference, answer)
    return {
        "rouge1_f": scores["rouge1"].fmeasure,
        "rougeL_f": scores["rougeL"].fmeasure,
    }


def _llm_binary(prompt: str, client: OpenAI, model_name: str) -> float:
    """Ask an OpenAI judge model to score on [0,1]. Returns 0.0 on parse failure."""
    resp = client.responses.create(
        model=model_name,
        input=prompt,
        temperature=0.0,
    )
    text = (resp.output_text or "").strip()
    match = re.search(r"\b([01](?:\.\d+)?)\b", text)
    return float(match.group(1)) if match else 0.0


def faithfulness(
    answer: str,
    context_passages: list[str],
    client: OpenAI,
    model_name: str,
) -> float:
    """LLM-as-judge: is every claim in the answer supported by context? Returns [0,1]."""
    ctx = "\n\n".join(context_passages[:5])
    prompt = (
        "Given the following context passages and an answer, "
        "rate how faithful the answer is to the context on a scale from 0 to 1 "
        "(1 = fully grounded, 0 = not grounded at all). "
        "Reply with ONLY a number.\n\n"
        f"Context:\n{ctx}\n\nAnswer:\n{answer}"
    )
    return _llm_binary(prompt, client, model_name)


def answer_relevancy(
    question: str,
    answer: str,
    client: OpenAI,
    model_name: str,
) -> float:
    """LLM-as-judge: does the answer address the question? Returns [0,1]."""
    prompt = (
        "Rate how well the following answer addresses the given question "
        "on a scale from 0 to 1 (1 = perfectly relevant, 0 = completely irrelevant). "
        "Reply with ONLY a number.\n\n"
        f"Question: {question}\n\nAnswer: {answer}"
    )
    return _llm_binary(prompt, client, model_name)
