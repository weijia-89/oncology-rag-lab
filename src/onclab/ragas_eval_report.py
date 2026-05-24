"""JSON schema for RAG eval reports (`reports/eval_report.json`).

The report aggregates Ragas-style RAG metrics (faithfulness, answer relevancy,
context precision/recall) plus pass-rate regression metadata. Writers and tests
share this module so the on-disk shape stays stable.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

MetricStatus = Literal["pass", "fail", "unavailable"]

RAG_METRIC_THRESHOLDS: dict[str, float] = {
    "faithfulness": 0.8,
    "answer_relevancy": 0.8,
    "context_precision": 0.7,
    "context_recall": 0.7,
}

REGRESSION_THRESHOLD = -0.05
DATASET_NAME = "synthetic-oncology-v1"
SCOPE_NOTE = (
    "Synthetic notes only; evaluation harness, not production clinical system."
)
OLLAMA_TOKEN_NOTE = (
    "Ollama local inference does not expose prompt/completion token counts; "
    "fields are null and estimated_cost_usd is 0.00."
)

REQUIRED_TOP_LEVEL_KEYS = frozenset(
    {
        "run_id",
        "timestamp_utc",
        "model",
        "embedding_model",
        "dataset",
        "cases_total",
        "cases_passed",
        "pass_rate",
        "baseline_pass_rate",
        "regression_threshold",
        "regression_status",
        "failed_cases",
        "rag_metrics",
        "latency",
        "token_usage",
        "estimated_cost_usd",
        "cost_note",
        "scope_note",
    }
)

REQUIRED_RAG_METRIC_KEYS = frozenset({"avg", "threshold", "status"})
REQUIRED_FAILED_CASE_KEYS = frozenset({"case_id", "failure_type", "expected", "actual", "notes"})


def metric_status(avg: float | None, threshold: float) -> MetricStatus:
    if avg is None:
        return "unavailable"
    return "pass" if avg >= threshold else "fail"


def build_rag_metric(avg: float | None, metric_name: str) -> dict[str, Any]:
    threshold = RAG_METRIC_THRESHOLDS[metric_name]
    return {
        "avg": avg,
        "threshold": threshold,
        "status": metric_status(avg, threshold),
    }


def regression_status(current: float, baseline: float) -> Literal["pass", "fail"]:
    if current - baseline < REGRESSION_THRESHOLD:
        return "fail"
    return "pass"


def validate_report(report: dict[str, Any]) -> list[str]:
    """Return a list of schema violations (empty == valid)."""
    errors: list[str] = []

    missing = REQUIRED_TOP_LEVEL_KEYS - report.keys()
    if missing:
        errors.append(f"missing top-level keys: {sorted(missing)}")

    rag = report.get("rag_metrics")
    if not isinstance(rag, dict):
        errors.append("rag_metrics must be an object")
    else:
        for name in RAG_METRIC_THRESHOLDS:
            entry = rag.get(name)
            if not isinstance(entry, dict):
                errors.append(f"rag_metrics.{name} must be an object")
                continue
            missing_metric = REQUIRED_RAG_METRIC_KEYS - entry.keys()
            if missing_metric:
                errors.append(f"rag_metrics.{name} missing keys: {sorted(missing_metric)}")
            status = entry.get("status")
            if status not in {"pass", "fail", "unavailable"}:
                errors.append(f"rag_metrics.{name}.status invalid: {status!r}")

    failed = report.get("failed_cases")
    if not isinstance(failed, list):
        errors.append("failed_cases must be a list")
    else:
        for idx, case in enumerate(failed):
            if not isinstance(case, dict):
                errors.append(f"failed_cases[{idx}] must be an object")
                continue
            missing_case = REQUIRED_FAILED_CASE_KEYS - case.keys()
            if missing_case:
                errors.append(f"failed_cases[{idx}] missing keys: {sorted(missing_case)}")

    latency = report.get("latency")
    if not isinstance(latency, dict):
        errors.append("latency must be an object")
    elif "avg_ms" not in latency:
        errors.append("latency.avg_ms is required")

    token_usage = report.get("token_usage")
    if not isinstance(token_usage, dict):
        errors.append("token_usage must be an object")
    elif "note" not in token_usage:
        errors.append("token_usage.note is required")

    reg_status = report.get("regression_status")
    if reg_status not in {"pass", "fail"}:
        errors.append(f"regression_status invalid: {reg_status!r}")

    return errors


def write_report(path: Path, report: dict[str, Any]) -> None:
    errors = validate_report(report)
    if errors:
        joined = "; ".join(errors)
        raise ValueError(f"invalid eval report schema: {joined}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def utc_now_iso() -> str:
    return datetime.now(tz=UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
