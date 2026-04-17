from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.config import Settings
from src.evaluation.harness import EvalReport, SampleResult, main, run_evaluation
from src.generation.generator import AnswerResult


def _sample_result(*, passed: bool = True) -> SampleResult:
    return SampleResult(
        question="What is the policy?",
        expected="The policy is documented.",
        answer="The policy is documented [SOURCE: handbook.pdf, chunk c1].",
        faithfulness=0.9,
        answer_relevancy=0.85,
        citation_coverage=1.0,
        rouge1_f=0.75,
        rougeL_f=0.7,
        passed=passed,
    )


def test_write_json_persists_detailed_results(tmp_path: Path):
    report = EvalReport(results=[_sample_result()])
    target = tmp_path / "eval_report.json"

    report.write_json(target)

    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["total_samples"] == 1
    assert payload["results"][0]["question"] == "What is the policy?"


@patch("src.evaluation.harness.rouge_scores", return_value={"rouge1_f": 0.8, "rougeL_f": 0.7})
@patch("src.evaluation.harness.citation_coverage", return_value=1.0)
@patch("src.evaluation.harness.answer_relevancy", return_value=0.88)
@patch("src.evaluation.harness.faithfulness", return_value=0.91)
@patch("src.evaluation.harness.RAGPipeline")
@patch("src.evaluation.harness.OpenAI")
def test_run_evaluation_aggregates_results(
    mock_openai,
    mock_pipeline_cls,
    mock_faithfulness,
    mock_answer_relevancy,
    mock_citation_coverage,
    mock_rouge_scores,
    tmp_path: Path,
):
    dataset_path = tmp_path / "eval.jsonl"
    dataset_path.write_text(
        json.dumps({"question": "What is the policy?", "answer": "The policy is documented."})
        + "\n",
        encoding="utf-8",
    )

    mock_pipeline = MagicMock()
    mock_pipeline.ask.return_value = AnswerResult(
        answer="The policy is documented [SOURCE: handbook.pdf, chunk c1].",
        cited_chunk_ids={"c1"},
        citation_coverage=1.0,
        retrieved_hits=[
            {"chunk_id": "c1", "text": "Policy text", "source": "handbook.pdf", "page": 1}
        ],
    )
    mock_pipeline_cls.return_value = mock_pipeline
    mock_openai.return_value = MagicMock()

    settings = Settings(
        openai_api_key="fake",
        eval_dataset_path=dataset_path,
        chroma_persist_dir=tmp_path / "chroma",
    )

    report = run_evaluation(settings)

    assert report.pass_rate == pytest.approx(1.0)
    assert report.avg_faithfulness == pytest.approx(0.91)
    mock_pipeline.ask.assert_called_once_with("What is the policy?")
    mock_faithfulness.assert_called_once()
    mock_answer_relevancy.assert_called_once()
    mock_citation_coverage.assert_called_once()
    mock_rouge_scores.assert_called_once()


@patch("src.evaluation.harness.get_settings")
@patch("src.evaluation.harness.run_evaluation")
def test_main_exits_when_thresholds_not_met(mock_run_evaluation, mock_get_settings):
    report = EvalReport(results=[_sample_result(passed=False)])
    report.results[0].faithfulness = 0.5
    report.results[0].answer_relevancy = 0.5
    report.results[0].citation_coverage = 0.5
    report.write_json = MagicMock()
    report.print_summary = MagicMock()
    mock_run_evaluation.return_value = report
    mock_get_settings.return_value = Settings(openai_api_key="fake")

    with pytest.raises(SystemExit):
        main()

    report.write_json.assert_called_once()
    report.print_summary.assert_called_once()


@patch("src.main.uvicorn.run")
def test_main_module_runs_uvicorn(mock_run):
    import runpy

    runpy.run_module("src.main", run_name="__main__")

    mock_run.assert_called_once_with("src.api:app", host="0.0.0.0", port=8000, reload=True)
