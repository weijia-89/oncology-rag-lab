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


@pytest.fixture
def mock_client(settings):
    """OllamaClient instance honoring MOCK_LLM=1.

    Function-scoped (default): each test gets a fresh client. Important
    if a test ever stubs the client's internals — no spillover into the
    next test.
    """
    from onclab.llm_client import OllamaClient

    return OllamaClient(host=settings.ollama_host, model=settings.llm_model)


def load_eval_note_nodes(settings):
    """Load one TextNode per synthetic note — shared by eval index fixtures."""
    pytest.importorskip("llama_index", reason="llama_index not installed")
    pytest.importorskip("chromadb", reason="chromadb not installed")

    from llama_index.core import SimpleDirectoryReader
    from llama_index.core.schema import TextNode

    note_paths = sorted(settings.notes_dir.glob("*.txt"))
    if not note_paths:
        pytest.skip(f"No .txt files found in {settings.notes_dir}")

    documents = SimpleDirectoryReader(
        input_files=[str(path) for path in note_paths],
    ).load_data()

    return [
        TextNode(text=doc.text, metadata=doc.metadata, id_=doc.doc_id)
        for doc in documents
    ]


def build_in_memory_eval_index(settings, embed_model, *, collection_name: str):
    """Build an ephemeral Chroma-backed index with a caller-supplied embedder."""
    pytest.importorskip("llama_index", reason="llama_index not installed")
    pytest.importorskip("chromadb", reason="chromadb not installed")

    import chromadb
    from llama_index.core import Settings as LIxSettings
    from llama_index.core import StorageContext, VectorStoreIndex
    from llama_index.vector_stores.chroma import ChromaVectorStore

    LIxSettings.embed_model = embed_model
    nodes = load_eval_note_nodes(settings)

    chroma_client = chromadb.EphemeralClient()
    collection = chroma_client.get_or_create_collection(name=collection_name)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    return VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        show_progress=False,
    )


@pytest.fixture(scope="session")
def eval_index(settings):
    """In-memory Chroma index built from synthetic notes — no Ollama, no disk.

    Used by tests/eval/test_retrieval.py so CI can assert retrieval quality
    without a prior `make ingest`. Uses a lightweight bag-of-words embedder
    so keyword overlap drives ranking (good enough for the 8-note corpus).
    """
    from llama_index.core.embeddings import BaseEmbedding

    class _BagOfWordsEmbedding(BaseEmbedding):
        """Deterministic keyword overlap embedder for CI-friendly retrieval tests."""

        embed_dim: int = 384

        def _bucket(self, token: str) -> int:
            # Stable across processes — unlike built-in hash().
            return int.from_bytes(token.encode(), "big") % self.embed_dim

        def _vectorize(self, text: str) -> list[float]:
            vec = [0.0] * self.embed_dim
            for token in text.lower().split():
                vec[self._bucket(token)] += 1.0
            norm = sum(x * x for x in vec) ** 0.5
            if norm:
                vec = [x / norm for x in vec]
            return vec

        def _get_query_embedding(self, query: str) -> list[float]:
            return self._vectorize(query)

        def _get_text_embedding(self, text: str) -> list[float]:
            return self._vectorize(text)

        async def _aget_query_embedding(self, query: str) -> list[float]:
            return self._vectorize(query)

    return build_in_memory_eval_index(
        settings,
        _BagOfWordsEmbedding(),
        collection_name="oncology_notes",
    )


@pytest.fixture(scope="module")
def baseline_embedding_index(settings):
    """MockEmbedding baseline index for retrieval drift A/B eval."""
    pytest.importorskip("llama_index", reason="llama_index not installed")
    from llama_index.core.embeddings import MockEmbedding

    return build_in_memory_eval_index(
        settings,
        MockEmbedding(embed_dim=384),
        collection_name="embedding_drift_baseline",
    )


@pytest.fixture(scope="module")
def candidate_embedding_index(settings):
    """Shifted MockEmbedding candidate index for retrieval drift A/B eval."""
    pytest.importorskip("llama_index", reason="llama_index not installed")
    from llama_index.core.embeddings import MockEmbedding

    class ShiftedMockEmbedding(MockEmbedding):
        shift: int = 17

        def _get_text_embedding(self, text: str) -> list[float]:
            vec = super()._get_text_embedding(text)
            if not self.shift:
                return vec
            n = len(vec)
            offset = self.shift % n
            return vec[offset:] + vec[:offset]

        def _get_query_embedding(self, query: str) -> list[float]:
            return self._get_text_embedding(query)

        async def _aget_query_embedding(self, query: str) -> list[float]:
            return self._get_query_embedding(query)

    return build_in_memory_eval_index(
        settings,
        ShiftedMockEmbedding(embed_dim=384, shift=17),
        collection_name="embedding_drift_candidate",
    )
