"""CI evaluation harness: reads JSONL dataset, runs pipeline, checks thresholds."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any

import structlog
from openai import OpenAI

from src.config import Settings, get_settings
from src.evaluation.metrics import (
    answer_relevancy,
    citation_coverage,
    faithfulness,
    rouge_scores,
)
from src.pipeline import RAGPipeline

logger = structlog.get_logger(__name__)


_REPORT_PATH = Path("eval_report.json")


@dataclass
class SampleResult:
    question: str
    expected: str
    answer: str
    faithfulness: float
    answer_relevancy: float
    citation_coverage: float
    rouge1_f: float
    rougeL_f: float
    passed: bool


@dataclass
class EvalReport:
    results: list[SampleResult] = field(default_factory=list)

    @property
    def avg_faithfulness(self) -> float:
        return mean(r.faithfulness for r in self.results) if self.results else 0.0

    @property
    def avg_relevancy(self) -> float:
        return mean(r.answer_relevancy for r in self.results) if self.results else 0.0

    @property
    def avg_citation_coverage(self) -> float:
        return mean(r.citation_coverage for r in self.results) if self.results else 0.0

    @property
    def avg_rouge1(self) -> float:
        return mean(r.rouge1_f for r in self.results) if self.results else 0.0

    @property
    def pass_rate(self) -> float:
        return sum(r.passed for r in self.results) / max(len(self.results), 1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_samples": len(self.results),
            "pass_rate": self.pass_rate,
            "avg_faithfulness": self.avg_faithfulness,
            "avg_answer_relevancy": self.avg_relevancy,
            "avg_citation_coverage": self.avg_citation_coverage,
            "avg_rouge1_f": self.avg_rouge1,
        }

    def print_summary(self) -> None:
        d = self.to_dict()
        print("\n===== Evaluation Report =====")
        for k, v in d.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        print("=" * 30)

    def write_json(self, path: Path = _REPORT_PATH) -> None:
        payload = {
            **self.to_dict(),
            "results": [
                {
                    "question": result.question,
                    "expected": result.expected,
                    "answer": result.answer,
                    "faithfulness": result.faithfulness,
                    "answer_relevancy": result.answer_relevancy,
                    "citation_coverage": result.citation_coverage,
                    "rouge1_f": result.rouge1_f,
                    "rougeL_f": result.rougeL_f,
                    "passed": result.passed,
                }
                for result in self.results
            ],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_evaluation(settings: Settings | None = None) -> EvalReport:
    cfg = settings or get_settings()
    dataset_path = Path(cfg.eval_dataset_path)

    if not dataset_path.exists():
        logger.error("eval_dataset_not_found", path=str(dataset_path))
        sys.exit(1)

    judge_client = OpenAI(api_key=cfg.openai_api_key)
    pipeline = RAGPipeline(cfg)
    report = EvalReport()

    samples = [json.loads(line) for line in dataset_path.read_text().splitlines() if line.strip()]
    logger.info("eval_start", n_samples=len(samples))

    for i, sample in enumerate(samples):
        question = sample["question"]
        expected = sample.get("answer", "")

        result = pipeline.ask(question)
        context_texts = [h["text"] for h in result.retrieved_hits]

        faith = faithfulness(result.answer, context_texts, judge_client, cfg.eval_model)
        relevancy = answer_relevancy(question, result.answer, judge_client, cfg.eval_model)
        cov = citation_coverage(result.answer, result.retrieved_hits)
        rouge = (
            rouge_scores(result.answer, expected)
            if expected
            else {"rouge1_f": 0.0, "rougeL_f": 0.0}
        )

        passed = (
            faith >= cfg.faithfulness_threshold
            and relevancy >= cfg.answer_relevancy_threshold
            and cov >= cfg.citation_coverage_threshold
        )

        sample_result = SampleResult(
            question=question,
            expected=expected,
            answer=result.answer,
            faithfulness=faith,
            answer_relevancy=relevancy,
            citation_coverage=cov,
            rouge1_f=rouge["rouge1_f"],
            rougeL_f=rouge["rougeL_f"],
            passed=passed,
        )
        report.results.append(sample_result)
        logger.info(
            "eval_sample",
            i=i + 1,
            total=len(samples),
            faithfulness=faith,
            relevancy=relevancy,
            citation_coverage=cov,
            passed=passed,
        )

    return report


def main() -> None:
    report = run_evaluation()
    report.write_json()
    report.print_summary()

    cfg = get_settings()
    failed = (
        report.avg_faithfulness < cfg.faithfulness_threshold
        or report.avg_relevancy < cfg.answer_relevancy_threshold
        or report.avg_citation_coverage < cfg.citation_coverage_threshold
    )
    if failed:
        logger.error("eval_thresholds_not_met", report=report.to_dict())
        sys.exit(1)

    logger.info("eval_passed", report=report.to_dict())


if __name__ == "__main__":
    main()
