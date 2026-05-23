"""Shared pytest fixtures.

`conftest.py` is auto-discovered by pytest. Anything you define here is
visible to every test in this directory tree without an explicit import.
The convention is to put fixtures here that more than one test file uses;
single-file fixtures stay inside that file.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# Force MOCK_LLM=1 for ALL tests by default. Individual tests that want to
# hit a real Ollama instance can override (rare; usually you'd run those
# manually outside pytest). This is a guardrail so a CI run doesn't try
# to dial out to localhost:11434 and hang.
os.environ.setdefault("MOCK_LLM", "1")


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Project root path. Useful when tests need to read fixture files
    (gold_standard.csv, synthetic notes) without hard-coding strings."""
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def settings(repo_root: Path):
    """Settings instance pointing at the in-tree data + a tmp persist dir.

    Why session-scoped: building Settings is cheap, but having a single
    instance per test session means every test sees the same model name
    and chunk size. Tests that need to override (e.g., a smaller chunk
    size) construct their own Settings inline.
    """
    from onclab.config import load_settings

    return load_settings(
        notes_dir=repo_root / "data" / "synthetic_notes",
        persist_dir=repo_root / "data" / "chroma_db",
    )


def build_in_memory_eval_index(repo_root: Path, settings):
    """Ephemeral Chroma index from synthetic notes (no Ollama, no disk ingest).

    Used by tests/eval/test_retrieval.py when data/chroma_db/ is absent so
    eval-mock and local runs get signal without `make ingest`.
    """
    pytest.importorskip("llama_index")
    pytest.importorskip("chromadb")

    import chromadb
    from llama_index.core import Settings as LIxSettings
    from llama_index.core import StorageContext, VectorStoreIndex
    from llama_index.core.schema import Document
    from llama_index.vector_stores.chroma import ChromaVectorStore

    try:
        from llama_index.core.embeddings import MockEmbedding  # type: ignore[import]
    except ImportError:
        pytest.skip("MockEmbedding not available in this llama-index version.")

    LIxSettings.embed_model = MockEmbedding(embed_dim=8)

    note_paths = sorted((repo_root / "data" / "synthetic_notes").glob("*.txt"))
    if not note_paths:
        pytest.skip("No synthetic notes under data/synthetic_notes/")

    documents = [
        Document(text=path.read_text(encoding="utf-8"), metadata={"source": path.name})
        for path in note_paths
    ]

    chroma_client = chromadb.EphemeralClient()
    collection = chroma_client.get_or_create_collection("oncology_eval_test")
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    return VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=False,
    )


@pytest.fixture(scope="module")
def eval_retrieval_index(settings, repo_root):
    """Persisted Chroma if present; otherwise in-memory synthetic notes."""
    if settings.persist_dir.exists():
        from onclab.ingest import load_index

        return load_index(settings)

    return build_in_memory_eval_index(repo_root, settings)


@pytest.fixture(scope="module")
def retrieval_settings(settings, repo_root):
    """Widen top_k when using the in-memory fallback (MockEmbedding)."""
    if settings.persist_dir.exists():
        return settings

    from dataclasses import replace

    note_count = len(list((repo_root / "data" / "synthetic_notes").glob("*.txt")))
    if note_count == 0:
        return settings

    return replace(settings, top_k=note_count)


@pytest.fixture
def mock_client(settings):
    """OllamaClient instance honoring MOCK_LLM=1.

    Function-scoped (default): each test gets a fresh client. Important
    if a test ever stubs the client's internals — no spillover into the
    next test.
    """
    from onclab.llm_client import OllamaClient

    return OllamaClient(host=settings.ollama_host, model=settings.llm_model)
