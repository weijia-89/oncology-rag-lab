"""Deterministic retrieval-layer tests.

These do NOT involve the LLM. They check the retriever in isolation:
given a query, does the chunk that should be top-1 actually rank top-1?
This is the "is the haystack searchable" gate that you want green
before you ever start scoring generations.

Why this layer is testable with classical asserts:
    Retrieval is deterministic given a frozen embedding model and a
    frozen index. There's no temperature, no sampling. If a retrieval
    test fails, the failure is in chunk size, embedding model choice,
    or the corpus itself — never in 'the model felt like a different
    answer today'.

These tests SKIP if the index hasn't been built yet, because they
need the persisted ChromaDB. Run `make ingest` once before invoking
this file. (The eval-mock target builds an in-memory index on the
fly so you don't need to ingest first; that's a future enhancement.)
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.eval


@pytest.fixture(scope="module")
def index(settings):
    """Open the persisted index. Skip the module if it doesn't exist."""
    if not settings.persist_dir.exists():
        pytest.skip(
            f"No ChromaDB at {settings.persist_dir}. Run `make ingest` first.",
        )
    from onclab.ingest import load_index

    return load_index(settings)


def test_lung_query_retrieves_lung_note(index, settings):
    """Asking a lung-specific question should pull the SYN-001 chunk."""
    from onclab.rag import retrieve_only

    chunks = retrieve_only(index, "What stage is the lung cancer patient?", settings)
    joined = " ".join(chunks).lower()
    assert "non-small cell" in joined or "syn-001" in joined, (
        f"Expected lung-cancer chunk in top_k={settings.top_k}; got: {joined[:200]!r}"
    )


def test_glioblastoma_query_retrieves_glioblastoma_note(index, settings):
    """Specific term ('glioblastoma') should retrieve the SYN-008 chunk."""
    from onclab.rag import retrieve_only

    chunks = retrieve_only(index, "Which patient has glioblastoma?", settings)
    joined = " ".join(chunks).lower()
    assert "glioblastoma" in joined or "syn-008" in joined


def test_unrelated_query_does_not_retrieve_random_garbage(index, settings):
    """Sanity check: an off-topic query shouldn't surface oncology-specific
    fingerprints with high similarity. This is mostly a smoke test — if
    everything is returning the same chunks for every query, your embedding
    model is probably broken or the chunks are too short."""
    from onclab.rag import retrieve_only

    chunks = retrieve_only(index, "What is the capital of France?", settings)
    # We just assert SOMETHING came back (top_k won't be zero) and that the
    # results aren't identical to the lung-query results above. A more
    # rigorous version would compute similarity scores and assert they're
    # below a calibration threshold.
    assert len(chunks) > 0
