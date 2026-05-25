"""Live Ollama embedding A/B over in-memory indexes (operator-only).

Skipped unless ``ONCLAB_RUN_LIVE_EMBEDDING_AB=1`` and Ollama responds at
``/api/tags``. Does not rebuild persisted ``data/chroma_db/`` or run
``make ingest``.
"""

from __future__ import annotations

import os

import pytest

from onclab.embedding_compare import (
    DRIFT_QUERIES,
    ollama_reachable,
    run_live_embedding_ab,
    write_embedding_live_ab_report,
)

pytestmark = [pytest.mark.eval, pytest.mark.drift, pytest.mark.live_embedding]

pytest.importorskip("llama_index", reason="llama_index not installed; skipping live embedding A/B eval")
pytest.importorskip("chromadb", reason="chromadb not installed; skipping live embedding A/B eval")


@pytest.fixture(scope="module")
def live_embedding_ab_gate(settings):
    """Skip unless operator opted in and Ollama is reachable."""
    if os.environ.get("ONCLAB_RUN_LIVE_EMBEDDING_AB") != "1":
        pytest.skip("ONCLAB_RUN_LIVE_EMBEDDING_AB=1 not set")
    if not ollama_reachable(settings.ollama_host):
        pytest.skip(f"Ollama not reachable at {settings.ollama_host}")


def test_live_embedding_ab_compare(settings, repo_root, live_embedding_ab_gate):
    """Live embedders: record top-1 agreement rate and write JSON report."""
    payload = run_live_embedding_ab(settings, queries=DRIFT_QUERIES)
    report_path = write_embedding_live_ab_report(repo_root, payload)

    assert payload["ollama_reachable"] is True
    assert payload["baseline_model"] == settings.embed_model
    assert payload["candidate_model"]
    assert payload["queries_compared"] >= 1
    assert len(payload["queries"]) == len(DRIFT_QUERIES)
    assert isinstance(payload["agreement_rate"], float)
    assert 0.0 <= payload["agreement_rate"] <= 1.0
    assert report_path.is_file()
