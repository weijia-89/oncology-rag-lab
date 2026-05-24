"""RAG eval harness: DeepEval Ragas-style metrics + eval_report.json."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from onclab.ragas_eval import run_rag_eval
from onclab.ragas_eval_report import validate_report

pytestmark = pytest.mark.eval


def test_rag_eval_writes_valid_report(eval_index, repo_root: Path, settings, mock_client, tmp_path):
    """MOCK_LLM=1 (conftest default): lexical proxies + schema-valid report."""
    out = tmp_path / "eval_report.json"
    report = run_rag_eval(
        repo_root,
        eval_index=eval_index,
        output_path=out,
        settings=settings,
        client=mock_client,
    )

    assert out.exists()
    assert validate_report(report) == []
    assert report["dataset"] == "synthetic-oncology-v1"
    assert report["cases_total"] > 0
    assert report["token_usage"]["prompt_tokens_avg"] is None
    assert report["estimated_cost_usd"] == 0.0
    assert "Ollama" in report["token_usage"]["note"]
    assert report["scope_note"].startswith("Synthetic notes only")

    for name in ("faithfulness", "answer_relevancy", "context_precision", "context_recall"):
        entry = report["rag_metrics"][name]
        assert entry["status"] in {"pass", "fail", "unavailable"}
        if entry["avg"] is not None:
            assert 0.0 <= entry["avg"] <= 1.0


def test_rag_eval_report_schema_fixture():
    """Static fixture documents the required JSON shape for reviewers."""
    fixture = {
        "run_id": "00000000-0000-4000-8000-000000000001",
        "timestamp_utc": "2026-05-24T12:00:00Z",
        "model": "mock",
        "embedding_model": None,
        "dataset": "synthetic-oncology-v1",
        "cases_total": 1,
        "cases_passed": 1,
        "pass_rate": 1.0,
        "baseline_pass_rate": 1.0,
        "regression_threshold": -0.05,
        "regression_status": "pass",
        "failed_cases": [],
        "rag_metrics": {
            "faithfulness": {"avg": 0.9, "threshold": 0.8, "status": "pass"},
            "answer_relevancy": {"avg": 0.9, "threshold": 0.8, "status": "pass"},
            "context_precision": {"avg": 0.75, "threshold": 0.7, "status": "pass"},
            "context_recall": {"avg": 0.75, "threshold": 0.7, "status": "pass"},
        },
        "latency": {"avg_ms": 12.5, "p95_ms": None},
        "token_usage": {
            "prompt_tokens_avg": None,
            "completion_tokens_avg": None,
            "total_tokens_avg": None,
            "note": "Ollama local inference does not expose prompt/completion token counts.",
        },
        "estimated_cost_usd": 0.0,
        "cost_note": "Ollama local inference does not expose prompt/completion token counts.",
        "scope_note": "Synthetic notes only; evaluation harness, not production clinical system.",
    }
    assert validate_report(fixture) == []
    # Round-trip JSON stability
    assert validate_report(json.loads(json.dumps(fixture))) == []
