"""Corpus scale stress — in-memory Chroma over 100 templated synthetic notes.

CI-safe harness: no Ollama, no persisted ``data/chroma_db/``, no ``make ingest``.
Uses ``scripts/seed_data.py`` templating via ``make_templated_note_nodes`` and
``MockEmbedding`` so ingest + retrieval stay deterministic and fast.

Thresholds (generous — fail only on obvious blow-ups):
- Index holds exactly 100 nodes after ingest.
- Each of 12 representative queries returns at least one chunk within 5s.
- Total ingest + query wall time under 120s (laptop-class hardware).

A full 500-note file-backed Chroma run remains a manual ``scripts/seed_data.py``
+ ``make ingest`` exercise (see ROADMAP longer-term Scale stress test).
"""

from __future__ import annotations

import os
import time

import pytest

from onclab.eval_report import build_stress_eval_report, write_stress_eval_report

pytestmark = pytest.mark.eval

pytest.importorskip("llama_index", reason="llama_index not installed; skipping corpus scale stress")
pytest.importorskip("chromadb", reason="chromadb not installed; skipping corpus scale stress")

CORPUS_SIZE = 100
STRESS_QUERIES = (
    "What stage is the lung cancer patient?",
    "Which patient has glioblastoma?",
    "ovarian cancer BRCA1 mutation carboplatin",
    "renal cell carcinoma pembrolizumab axitinib",
    "high-grade serous carcinoma stage IIIC",
    "clear cell renal mass pulmonary nodules",
    "ECOG performance status treatment plan",
    "cytoreductive surgery bevacizumab olaparib",
    "AJCC 8th edition staging code",
    "What is the capital of France?",
    "ascites omental caking abdominal distension",
    "intermediate IMDC risk first-line therapy",
)
MAX_QUERY_SECONDS = 5.0
MAX_TOTAL_SECONDS = 120.0


def test_corpus_scale_stress_ingest_and_retrieval(
    repo_root, settings, build_scale_stress_index
):
    """Ingest 100 templated notes and run representative queries under latency caps."""
    from onclab.rag import retrieve_only

    t0 = time.perf_counter()
    index, node_count = build_scale_stress_index(repo_root, CORPUS_SIZE)
    # sdk-review F2: node count asserted in fixture before ingest — no _collection.count()
    assert node_count == CORPUS_SIZE, (
        f"Expected {CORPUS_SIZE} templated nodes, got {node_count}"
    )

    per_query_ms: list[float] = []
    chunks_per_query: list[int] = []

    for query in STRESS_QUERIES:
        q_start = time.perf_counter()
        chunks = retrieve_only(index, query, settings)
        q_elapsed = time.perf_counter() - q_start
        per_query_ms.append(q_elapsed * 1000)
        chunks_per_query.append(len(chunks))

        assert chunks, f"No retrieval results for query {query!r}"
        assert len(chunks) <= settings.top_k
        assert q_elapsed < MAX_QUERY_SECONDS, (
            f"Query {query!r} took {q_elapsed:.2f}s (limit {MAX_QUERY_SECONDS}s)"
        )

    total_elapsed = time.perf_counter() - t0
    assert total_elapsed < MAX_TOTAL_SECONDS, (
        f"All queries took {total_elapsed:.2f}s (limit {MAX_TOTAL_SECONDS}s); "
        f"per-query ms: {per_query_ms!r}"
    )

    report = build_stress_eval_report(
        repo_root=repo_root,
        settings=settings,
        corpus_size=CORPUS_SIZE,
        queries=STRESS_QUERIES,
        per_query_ms=per_query_ms,
        chunks_per_query=chunks_per_query,
        total_elapsed_s=total_elapsed,
        mock_llm=os.environ.get("MOCK_LLM", "0") == "1",
    )
    report_path = write_stress_eval_report(repo_root, report)
    assert report_path.is_file(), f"stress eval report not written: {report_path}"
