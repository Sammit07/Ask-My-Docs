"""Unit tests for EvalReport aggregation (no LLM calls)."""

import pytest

from src.evaluation.harness import EvalReport, SampleResult


def _sample(faith: float, relevancy: float, cov: float, passed: bool) -> SampleResult:
    return SampleResult(
        question="q",
        expected="e",
        answer="a",
        faithfulness=faith,
        answer_relevancy=relevancy,
        citation_coverage=cov,
        rouge1_f=0.5,
        rougeL_f=0.5,
        passed=passed,
    )


def test_empty_report_defaults():
    r = EvalReport()
    assert r.avg_faithfulness == 0.0
    assert r.pass_rate == 0.0


def test_avg_metrics():
    r = EvalReport(
        results=[
            _sample(0.8, 0.9, 1.0, True),
            _sample(0.6, 0.7, 0.8, False),
        ]
    )
    assert r.avg_faithfulness == pytest.approx(0.7)
    assert r.avg_relevancy == pytest.approx(0.8)
    assert r.avg_citation_coverage == pytest.approx(0.9)
    assert r.pass_rate == pytest.approx(0.5)


def test_to_dict_keys():
    r = EvalReport(results=[_sample(1.0, 1.0, 1.0, True)])
    d = r.to_dict()
    assert "total_samples" in d
    assert "pass_rate" in d
    assert "avg_faithfulness" in d


def test_print_summary_runs(capsys):
    r = EvalReport(results=[_sample(0.9, 0.85, 0.95, True)])
    r.print_summary()
    out = capsys.readouterr().out
    assert "Evaluation Report" in out
