"""Retrieval drift under two deterministic embedding configs.

Generation drift (``scripts/drift_compare.py``) diffs entity extractions when
the LLM changes. This module diffs **top-1 chunk ids** when the embedder
changes — same synthetic corpus, same queries, no Ollama, no persisted
``data/chroma_db/``.

Both indexes use llama-index ``MockEmbedding`` variants so CI stays fast and
deterministic. A cyclic dimension shift on the candidate model simulates a
real embedding swap (BGE-M3 vs nomic-embed-text) without pulling HF weights.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.eval, pytest.mark.drift]

pytest.importorskip("llama_index", reason="llama_index not installed; skipping embedding drift eval")
pytest.importorskip("chromadb", reason="chromadb not installed; skipping embedding drift eval")

from onclab.embedding_compare import (  # noqa: E402
    DRIFT_QUERIES,
    agreement_rate,
    compare_retrieval_top1,
)


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
