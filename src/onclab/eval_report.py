"""JSON eval report builders for onclab harness runs.

Stress corpus reports use ``reports/stress_eval_report.json``. Full RAGAS /
DeepEval gold reports will use ``reports/eval_report.json`` (separate branch).
"""

from __future__ import annotations

import json
import statistics
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

STRESS_REPORT_REL_PATH = Path("reports/stress_eval_report.json")

_REQUIRED_TOP_KEYS = frozenset(
    {
        "run_id",
        "timestamp_utc",
        "eval_kind",
        "model",
        "dataset",
        "mock_llm",
        "cases",
        "summary",
        "latency",
        "token_usage",
        "rag_metrics",
        "scope_note",
    }
)


def default_stress_report_path(repo_root: Path) -> Path:
    return repo_root / STRESS_REPORT_REL_PATH


def build_stress_eval_report(
    *,
    repo_root: Path,
    settings: Any,
    corpus_size: int,
    queries: tuple[str, ...],
    per_query_ms: list[float],
    chunks_per_query: list[int],
    total_elapsed_s: float,
    mock_llm: bool,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Build a stress-eval payload from measured harness results (no fabricated metrics)."""
    if len(per_query_ms) != len(queries) or len(chunks_per_query) != len(queries):
        raise ValueError("queries, per_query_ms, and chunks_per_query must align")

    cases: list[dict[str, Any]] = []
    for idx, (query, latency_ms, chunk_count) in enumerate(
        zip(queries, per_query_ms, chunks_per_query, strict=True)
    ):
        cases.append(
            {
                "id": f"stress_q{idx:02d}",
                "query": query,
                "passed": True,
                "latency_ms": round(latency_ms, 3),
                "chunks_returned": chunk_count,
            }
        )

    cases_total = len(cases)
    summary = {
        "cases_total": cases_total,
        "cases_passed": cases_total,
        "pass_rate": 1.0 if cases_total else 0.0,
        "ingest_plus_query_seconds": round(total_elapsed_s, 3),
        "max_query_seconds_cap": 5.0,
        "max_total_seconds_cap": 120.0,
    }

    latency_block: dict[str, Any] = {
        "per_query_ms": [round(ms, 3) for ms in per_query_ms],
    }
    if per_query_ms:
        sorted_ms = sorted(per_query_ms)
        latency_block["p50_ms"] = round(statistics.median(sorted_ms), 3)
        latency_block["p95_ms"] = round(
            sorted_ms[max(0, int(0.95 * len(sorted_ms)) - 1)],
            3,
        )
        latency_block["max_ms"] = round(max(per_query_ms), 3)

    llm_label = "mock" if mock_llm else getattr(settings, "llm_model", "unknown")

    return {
        "run_id": run_id or str(uuid.uuid4()),
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "eval_kind": "corpus_scale_stress",
        "model": {
            "llm": llm_label,
            "embedder": "MockEmbedding(embed_dim=384)",
        },
        "dataset": {
            "name": "templated_synthetic",
            "corpus_size": corpus_size,
            "collection": "corpus_scale_stress",
            "notes_dir": str(getattr(settings, "notes_dir", "")),
        },
        "mock_llm": mock_llm,
        "cases": cases,
        "summary": summary,
        "latency": latency_block,
        "token_usage": None,
        "rag_metrics": "unavailable",
        "scope_note": (
            "Retrieval-scale stress harness only (in-memory Chroma, templated notes). "
            "RAGAS / gold extraction metrics belong in reports/eval_report.json "
            "(feat/ragas-eval-sdk)."
        ),
    }


def validate_stress_eval_report(payload: dict[str, Any]) -> None:
    """Raise ValueError if payload is missing required stress-report fields."""
    missing = _REQUIRED_TOP_KEYS - payload.keys()
    if missing:
        raise ValueError(f"stress eval report missing keys: {sorted(missing)}")

    if payload.get("eval_kind") != "corpus_scale_stress":
        raise ValueError("eval_kind must be corpus_scale_stress")

    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("cases must be a non-empty list")

    summary = payload.get("summary")
    if not isinstance(summary, dict):
        raise ValueError("summary must be a dict")

    for key in ("cases_total", "cases_passed", "pass_rate"):
        if key not in summary:
            raise ValueError(f"summary missing {key!r}")

    if payload.get("rag_metrics") != "unavailable":
        raise ValueError("rag_metrics must be 'unavailable' for stress-only reports")


def write_stress_eval_report(
    repo_root: Path,
    payload: dict[str, Any],
    *,
    path: Path | None = None,
) -> Path:
    """Validate and write stress eval JSON under reports/."""
    validate_stress_eval_report(payload)
    out = path or default_stress_report_path(repo_root)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return out
