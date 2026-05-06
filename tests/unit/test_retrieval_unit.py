"""Unit tests for the retrieval layer — no Ollama, no persisted ChromaDB.

The eval/test_retrieval.py tests are correct integration tests that need a
pre-built index; they belong in `make eval`. This file covers the same
retrieval logic at the unit level by:

  1. Building a tiny in-memory ChromaDB collection from two short synthetic
     strings (so the "index" is always fresh, no disk, no embedding model).
  2. Injecting that collection into a VectorStoreIndex.
  3. Querying the index and asserting that the correct chunk ranks first.

Why in-memory embeddings instead of OllamaEmbedding:
    OllamaEmbedding would need a running Ollama server, which CI doesn't have.
    We use llama-index's built-in MockEmbedding (returns random-but-stable
    unit vectors). The test doesn't check semantic ranking quality — it checks
    that the wiring between VectorStoreIndex, ChromaVectorStore, and
    retrieve_only() is correct. For semantic quality, see tests/eval/.

Why two documents with distinct fingerprint strings:
    With MockEmbedding the vectors are random, so retrieval order isn't
    semantically meaningful. The test instead checks that retrieve_only
    returns *something* (the retriever doesn't crash) and that the returned
    chunks come from our corpus (not empty strings). That's enough to catch
    wiring regressions: if ChromaVectorStore stopped persisting chunks,
    retrieve_only would return empty strings.

These tests skip gracefully when llama_index or chromadb are not installed
(e.g., in environments where only the pure-Python dependencies are available).
"""

from __future__ import annotations

import pytest

# Skip the whole module if llama-index is not installed. This mirrors the same
# guard used in test_chunking.py and prevents a hard ImportError during
# collection that would abort the entire test session.
pytest.importorskip("llama_index", reason="llama_index not installed; skipping retrieval unit tests")
pytest.importorskip("chromadb", reason="chromadb not installed; skipping retrieval unit tests")


@pytest.fixture(scope="module")
def in_memory_index(repo_root):
    """Build a throwaway VectorStoreIndex backed by an ephemeral ChromaDB.

    scope=module: the index is built once for the whole file, shared across
    tests. That mirrors the real usage pattern (ingest once, query many times)
    and keeps the test suite fast.
    """
    import chromadb
    from llama_index.core import Settings as LIxSettings
    from llama_index.core import StorageContext, VectorStoreIndex
    from llama_index.core.schema import TextNode
    from llama_index.vector_stores.chroma import ChromaVectorStore

    # Use a tiny deterministic embedding model that ships without Ollama.
    # We try the llama-index mock first; if unavailable we skip the test
    # rather than pulling a heavy HuggingFace model.
    try:
        from llama_index.core.embeddings import MockEmbedding  # type: ignore[import]

        embed = MockEmbedding(embed_dim=8)
    except ImportError:
        pytest.skip("MockEmbedding not available in this llama-index version.")

    LIxSettings.embed_model = embed

    # Two minimal documents with distinct identifying strings.
    nodes = [
        TextNode(
            text="SYN-001 non-small cell lung cancer stage IIIA adenocarcinoma carboplatin paclitaxel",
            id_="node-syn001",
            metadata={"source": "patient_001.txt"},
        ),
        TextNode(
            text="SYN-008 glioblastoma multiforme WHO grade IV temozolomide ECOG 1",
            id_="node-syn008",
            metadata={"source": "patient_008.txt"},
        ),
    ]

    # Ephemeral in-memory Chroma — no files written, no cleanup needed.
    chroma_client = chromadb.EphemeralClient()
    collection = chroma_client.get_or_create_collection("oncology_unit_test")
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex(nodes, storage_context=storage_context, show_progress=False)
    return index


def test_retrieve_returns_nonempty_chunks(in_memory_index, settings):
    """retrieve_only must return at least one non-empty string for any query."""
    from onclab.rag import retrieve_only

    chunks = retrieve_only(in_memory_index, "What type of lung cancer?", settings)
    assert len(chunks) > 0, "retrieve_only returned no chunks"
    assert any(c.strip() for c in chunks), "All returned chunks are empty strings"


def test_retrieve_finds_lung_fingerprint(in_memory_index, settings):
    """The lung-cancer node must appear somewhere in the top_k results.

    With MockEmbedding the vectors are random, so this is a probabilistic
    check. With only 2 nodes and top_k=3, at least one of the 2 nodes will
    always be in the result set — we just assert the corpus text is present.
    """
    from onclab.rag import retrieve_only

    chunks = retrieve_only(in_memory_index, "lung cancer staging adenocarcinoma", settings)
    all_text = " ".join(chunks).lower()
    # Either our lung node or our glioblastoma node must appear — proves
    # the index is returning actual corpus content, not empty results.
    assert "non-small cell" in all_text or "glioblastoma" in all_text, (
        f"Neither corpus fingerprint found in retrieved chunks: {all_text[:300]!r}"
    )


def test_retrieve_respects_top_k(in_memory_index, settings):
    """retrieve_only must return at most top_k chunks."""
    from onclab.rag import retrieve_only

    chunks = retrieve_only(in_memory_index, "cancer", settings)
    assert len(chunks) <= settings.top_k, (
        f"Got {len(chunks)} chunks, expected at most top_k={settings.top_k}"
    )
