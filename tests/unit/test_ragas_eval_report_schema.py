"""Unit tests for eval_report.json schema validation."""

from __future__ import annotations

from onclab.ragas_eval_report import (
    RAG_METRIC_THRESHOLDS,
    build_rag_metric,
    metric_status,
    regression_status,
    validate_report,
)


def _minimal_report(**overrides):
    base = {
        "run_id": "run-1",
        "timestamp_utc": "2026-05-24T00:00:00Z",
        "model": "qwen3:14b",
        "embedding_model": "nomic-embed-text",
        "dataset": "synthetic-oncology-v1",
        "cases_total": 2,
        "cases_passed": 2,
        "pass_rate": 1.0,
        "baseline_pass_rate": 1.0,
        "regression_threshold": -0.05,
        "regression_status": "pass",
        "failed_cases": [],
        "rag_metrics": {name: build_rag_metric(0.9, name) for name in RAG_METRIC_THRESHOLDS},
        "latency": {"avg_ms": 10.0, "p95_ms": None},
        "token_usage": {
            "prompt_tokens_avg": None,
            "completion_tokens_avg": None,
            "total_tokens_avg": None,
            "note": "Ollama: no token counts",
        },
        "estimated_cost_usd": 0.0,
        "cost_note": "Ollama: no token counts",
        "scope_note": "Synthetic notes only; evaluation harness, not production clinical system.",
    }
    base.update(overrides)
    return base


def test_validate_report_accepts_minimal():
    assert validate_report(_minimal_report()) == []


def test_validate_report_rejects_missing_keys():
    report = _minimal_report()
    del report["run_id"]
    errors = validate_report(report)
    assert any("run_id" in e for e in errors)


def test_metric_status_thresholds():
    assert metric_status(0.8, 0.8) == "pass"
    assert metric_status(0.79, 0.8) == "fail"
    assert metric_status(None, 0.8) == "unavailable"


def test_regression_status():
    assert regression_status(0.9, 0.95) == "pass"
    assert regression_status(0.85, 0.95) == "fail"
