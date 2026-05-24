"""Ragas-style RAG evaluation harness using DeepEval native metrics.

DeepEval 4.x removed the RAGAS* wrapper classes; the equivalent metrics are
FaithfulnessMetric, AnswerRelevancyMetric, ContextualPrecisionMetric, and
ContextualRecallMetric. Raw `ragas` was not added — it fails to import against
current langchain-community on Python 3.13 (missing vertexai shim).

MOCK_LLM=1 uses deterministic lexical overlap proxies (real computed scores,
not LLM-judge). Live runs call Ollama via DeepEval's OllamaModel.
"""

from __future__ import annotations

import csv
import os
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from onclab.config import Settings, load_settings
from onclab.extract import ENTITY_TYPES, ExtractionRequest, extract_entity
from onclab.llm_client import OllamaClient
from onclab.rag import retrieve_only
from onclab.ragas_eval_report import (
    DATASET_NAME,
    OLLAMA_TOKEN_NOTE,
    RAG_METRIC_THRESHOLDS,
    REGRESSION_THRESHOLD,
    SCOPE_NOTE,
    build_rag_metric,
    regression_status,
    utc_now_iso,
    write_report,
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def _overlap_ratio(left: str, right: str) -> float:
    a, b = _tokens(left), _tokens(right)
    if not a:
        return 0.0
    return len(a & b) / len(a)


@dataclass(frozen=True)
class RagEvalCase:
    case_id: str
    patient_id: str
    entity_type: str
    question: str
    expected: str
    note_text: str


@dataclass
class CaseResult:
    case: RagEvalCase
    actual: str
    contexts: list[str]
    passed: bool
    failure_type: str | None
    notes: str
    latency_ms: float
    scores: dict[str, float | None]


def load_rag_eval_cases(repo_root: Path) -> list[RagEvalCase]:
    """Build eval cases from gold_standard.csv (base synthetic notes only)."""
    gold_path = repo_root / "data" / "gold_standard.csv"
    notes_dir = repo_root / "data" / "synthetic_notes"
    cases: list[RagEvalCase] = []

    with open(gold_path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            patient_id = row["patient_id"].strip()
            entity_type = row["entity_type"].strip()
            if entity_type not in ENTITY_TYPES:
                continue
            suffix = patient_id.split("-")[1]
            note_path = notes_dir / f"patient_{suffix}.txt"
            if not note_path.exists():
                continue
            cases.append(
                RagEvalCase(
                    case_id=f"{patient_id}:{entity_type}",
                    patient_id=patient_id,
                    entity_type=entity_type,
                    question=f"What is the {entity_type} for {patient_id}?",
                    expected=row["gold_value"].strip(),
                    note_text=note_path.read_text(encoding="utf-8"),
                )
            )
    return cases


def _lexical_scores(case: RagEvalCase, actual: str, contexts: list[str]) -> dict[str, float]:
    joined_context = " ".join(contexts)
    return {
        "faithfulness": _overlap_ratio(actual, joined_context),
        "answer_relevancy": _overlap_ratio(actual, case.expected),
        "context_precision": (
            sum(1 for ctx in contexts if _overlap_ratio(case.question, ctx) > 0) / len(contexts)
            if contexts
            else 0.0
        ),
        "context_recall": _overlap_ratio(case.expected, joined_context),
    }


def _live_deepeval_scores(
    case: RagEvalCase,
    actual: str,
    contexts: list[str],
    settings: Settings,
) -> dict[str, float | None]:
    pytest_import = __import__("pytest")
    pytest_import.importorskip("deepeval")
    from deepeval.metrics import (  # noqa: PLC0415
        AnswerRelevancyMetric,
        ContextualPrecisionMetric,
        ContextualRecallMetric,
        FaithfulnessMetric,
    )
    from deepeval.models.llms.ollama_model import OllamaModel  # noqa: PLC0415
    from deepeval.test_case import LLMTestCase  # noqa: PLC0415

    judge = OllamaModel(model=settings.llm_model, base_url=settings.ollama_host)
    test_case = LLMTestCase(
        input=case.question,
        actual_output=actual,
        expected_output=case.expected,
        retrieval_context=contexts,
    )

    metrics = {
        "faithfulness": FaithfulnessMetric(threshold=RAG_METRIC_THRESHOLDS["faithfulness"], model=judge),
        "answer_relevancy": AnswerRelevancyMetric(
            threshold=RAG_METRIC_THRESHOLDS["answer_relevancy"], model=judge
        ),
        "context_precision": ContextualPrecisionMetric(
            threshold=RAG_METRIC_THRESHOLDS["context_precision"], model=judge
        ),
        "context_recall": ContextualRecallMetric(
            threshold=RAG_METRIC_THRESHOLDS["context_recall"], model=judge
        ),
    }

    out: dict[str, float | None] = {}
    for name, metric in metrics.items():
        metric.measure(test_case)
        out[name] = float(metric.score) if metric.score is not None else None
    return out


def _case_passes(scores: dict[str, float | None], mock_mode: bool) -> tuple[bool, str | None, str]:
    if mock_mode:
        # Lexical proxy: answer relevancy only (context recall is unreliable for
        # short controlled-vocab gold strings that may not appear verbatim in chunks).
        relevancy = scores.get("answer_relevancy") or 0.0
        if relevancy >= RAG_METRIC_THRESHOLDS["answer_relevancy"]:
            return True, None, "mock lexical answer-relevancy pass"
        return False, "lexical_threshold", "mock lexical answer-relevancy below threshold"

    failures = [
        name
        for name, threshold in RAG_METRIC_THRESHOLDS.items()
        if scores.get(name) is not None and scores[name] < threshold
    ]
    if failures:
        return False, "rag_metric", f"below threshold: {', '.join(failures)}"
    return True, None, "all rag metrics at or above threshold"


def evaluate_case(
    case: RagEvalCase,
    *,
    eval_index,
    settings: Settings,
    client: OllamaClient,
    mock_mode: bool,
) -> CaseResult:
    start = time.perf_counter()
    contexts = retrieve_only(eval_index, case.question, settings)
    request = ExtractionRequest(patient_id=case.patient_id, note_text=case.note_text)
    result = extract_entity(request, case.entity_type, client=client, settings=settings)
    actual = result.value
    latency_ms = (time.perf_counter() - start) * 1000.0

    if mock_mode:
        scores = _lexical_scores(case, actual, contexts)
    else:
        scores = _live_deepeval_scores(case, actual, contexts, settings)

    passed, failure_type, note = _case_passes(scores, mock_mode)
    return CaseResult(
        case=case,
        actual=actual,
        contexts=contexts,
        passed=passed,
        failure_type=failure_type,
        notes=note,
        latency_ms=latency_ms,
        scores=scores,
    )


def _load_baseline_pass_rate(baseline_path: Path) -> float:
    if not baseline_path.exists():
        return 0.0
    import json

    data = json.loads(baseline_path.read_text(encoding="utf-8"))
    if "_comment" in data and data.get("summary", {}).get("total", 0) <= 1:
        return 0.0
    return float(data.get("pass_rate", data.get("summary", {}).get("passed", 0)))


def build_report_from_results(
    results: list[CaseResult],
    *,
    settings: Settings,
    baseline_pass_rate: float,
    embedding_model: str | None,
) -> dict[str, Any]:
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    pass_rate = passed / total if total else 0.0

    failed_cases = [
        {
            "case_id": r.case.case_id,
            "failure_type": r.failure_type or "unknown",
            "expected": r.case.expected,
            "actual": r.actual,
            "notes": r.notes,
        }
        for r in results
        if not r.passed
    ]

    metric_avgs: dict[str, float | None] = {}
    for name in RAG_METRIC_THRESHOLDS:
        values = [r.scores.get(name) for r in results if r.scores.get(name) is not None]
        metric_avgs[name] = (sum(values) / len(values)) if values else None

    latencies = [r.latency_ms for r in results]
    avg_latency = sum(latencies) / len(latencies) if latencies else None

    return {
        "run_id": str(uuid.uuid4()),
        "timestamp_utc": utc_now_iso(),
        "model": settings.llm_model,
        "embedding_model": embedding_model,
        "dataset": DATASET_NAME,
        "cases_total": total,
        "cases_passed": passed,
        "pass_rate": pass_rate,
        "baseline_pass_rate": baseline_pass_rate,
        "regression_threshold": REGRESSION_THRESHOLD,
        "regression_status": regression_status(pass_rate, baseline_pass_rate),
        "failed_cases": failed_cases,
        "rag_metrics": {name: build_rag_metric(metric_avgs[name], name) for name in RAG_METRIC_THRESHOLDS},
        "latency": {"avg_ms": avg_latency, "p95_ms": None},
        "token_usage": {
            "prompt_tokens_avg": None,
            "completion_tokens_avg": None,
            "total_tokens_avg": None,
            "note": OLLAMA_TOKEN_NOTE,
        },
        "estimated_cost_usd": 0.0,
        "cost_note": OLLAMA_TOKEN_NOTE,
        "scope_note": SCOPE_NOTE,
    }


def run_rag_eval(
    repo_root: Path,
    *,
    eval_index,
    output_path: Path,
    baseline_path: Path | None = None,
    settings: Settings | None = None,
    client: OllamaClient | None = None,
) -> dict[str, Any]:
    settings = settings or load_settings(
        notes_dir=repo_root / "data" / "synthetic_notes",
        persist_dir=repo_root / "data" / "chroma_db",
    )
    client = client or OllamaClient(host=settings.ollama_host, model=settings.llm_model)
    mock_mode = os.environ.get("MOCK_LLM", "0") == "1" or settings.mock_llm

    cases = load_rag_eval_cases(repo_root)
    results = [
        evaluate_case(case, eval_index=eval_index, settings=settings, client=client, mock_mode=mock_mode)
        for case in cases
    ]

    baseline_rate = _load_baseline_pass_rate(baseline_path) if baseline_path else 0.0
    embedding_model = None if mock_mode else settings.embed_model
    report = build_report_from_results(
        results,
        settings=settings,
        baseline_pass_rate=baseline_rate,
        embedding_model=embedding_model,
    )
    write_report(output_path, report)
    return report
