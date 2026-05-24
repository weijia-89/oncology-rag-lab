"""Retrieval drift under two deterministic embedding configs.

Generation drift (`scripts/drift_compare.py`) diffs entity extractions when
the LLM changes. This module diffs **top-1 chunk ids** when the embedder
changes — same synthetic corpus, same queries, no Ollama, no persisted
`data/chroma_db/`.

Both indexes use llama-index `MockEmbedding` variants so CI stays fast and
deterministic. A cyclic dimension shift on the candidate model simulates a
real embedding swap (BGE-M3 vs nomic-embed-text) without pulling HF weights.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.eval, pytest.mark.drift]

pytest.importorskip("llama_index", reason="llama_index not installed; skipping embedding drift eval")
pytest.importorskip("chromadb", reason="chromadb not installed; skipping embedding drift eval")

from llama_index.core import VectorStoreIndex  # noqa: E402

DRIFT_QUERIES = (
    "What stage is the lung cancer patient?",
    "Which patient has glioblastoma?",
    "What is the capital of France?",
)


def top1_node_id(index: VectorStoreIndex, query: str, settings) -> str:
    """Return the node id of the highest-scoring retrieved chunk."""
    from onclab.rag import build_retriever

    nodes = build_retriever(index, settings).retrieve(query)
    assert nodes, f"No retrieval results for query {query!r}"
    return nodes[0].node.id_


def compare_retrieval_top1(
    baseline_index: VectorStoreIndex,
    candidate_index: VectorStoreIndex,
    queries: tuple[str, ...],
    settings,
) -> list[tuple[str, str, str, bool]]:
    """Side-by-side top-1 ids: (query, baseline_id, candidate_id, agree)."""
    rows: list[tuple[str, str, str, bool]] = []
    for query in queries:
        baseline_id = top1_node_id(baseline_index, query, settings)
        candidate_id = top1_node_id(candidate_index, query, settings)
        rows.append((query, baseline_id, candidate_id, baseline_id == candidate_id))
    return rows


def agreement_rate(rows: list[tuple[str, str, str, bool]]) -> float:
    """Fraction of queries where baseline and candidate agree on top-1 chunk id."""
    if not rows:
        return 1.0
    agreements = sum(1 for _, _, _, agree in rows if agree)
    return agreements / len(rows)


def test_embedding_swap_produces_measurable_retrieval_drift(
    baseline_embedding_index,
    candidate_embedding_index,
    settings,
):
    """Same queries, two embedders — top-1 ids must not match on every query."""
    rows = compare_retrieval_top1(
        baseline_embedding_index,
        candidate_embedding_index,
        DRIFT_QUERIES,
        settings,
    )

    rate = agreement_rate(rows)
    assert rate < 1.0, (
        f"Expected retrieval drift between embedding configs; got 100% top-1 agreement: {rows!r}"
    )
    assert len(rows) == len(DRIFT_QUERIES)

    for query, baseline_id, candidate_id, agree in rows:
        assert baseline_id and candidate_id
        assert isinstance(agree, bool)
        assert query in DRIFT_QUERIES
