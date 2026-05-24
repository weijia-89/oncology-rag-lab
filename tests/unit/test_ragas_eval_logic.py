"""Unit tests for RAG eval pass/baseline logic."""

from __future__ import annotations

from onclab.config import load_settings
from onclab.ragas_eval import _case_passes, _load_baseline_pass_rate, build_report_from_results
from onclab.ragas_eval_report import RAG_METRIC_THRESHOLDS


def test_case_passes_mock_only_gates_available_metrics():
    scores = {
        "faithfulness": None,
        "answer_relevancy": 0.95,
        "context_precision": 0.75,
        "context_recall": None,
    }
    passed, failure_type, _ = _case_passes(scores, mock_mode=True)
    assert passed is True
    assert failure_type is None

    scores["answer_relevancy"] = 0.5
    passed, failure_type, note = _case_passes(scores, mock_mode=True)
    assert passed is False
    assert failure_type == "lexical_threshold"
    assert "answer_relevancy" in note


def test_case_passes_live_requires_all_scored_metrics():
    scores = {name: threshold for name, threshold in RAG_METRIC_THRESHOLDS.items()}
    passed, _, _ = _case_passes(scores, mock_mode=False)
    assert passed is True

    scores["faithfulness"] = 0.1
    passed, failure_type, note = _case_passes(scores, mock_mode=False)
    assert passed is False
    assert failure_type == "rag_metric"
    assert "faithfulness" in note


def test_load_baseline_pass_rate_placeholder(tmp_path):
    path = tmp_path / "baseline.json"
    path.write_text('{"_comment": "placeholder", "pass_rate": 0.0, "cases_total": 0}\n')
    assert _load_baseline_pass_rate(path) is None


def test_load_baseline_pass_rate_real(tmp_path):
    path = tmp_path / "baseline.json"
    path.write_text('{"pass_rate": 0.85, "cases_total": 40, "cases_passed": 34}\n')
    assert _load_baseline_pass_rate(path) == 0.85


def test_build_report_regression_pass_when_no_baseline():
    settings = load_settings()
    report = build_report_from_results(
        [],
        settings=settings,
        baseline_pass_rate=None,
        embedding_model=None,
    )
    assert report["regression_status"] == "pass"
    assert report["baseline_pass_rate"] == report["pass_rate"]
