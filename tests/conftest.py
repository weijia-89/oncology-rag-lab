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

    from onclab.eval_embedders import load_eval_note_nodes as _load_eval_note_nodes

    try:
        return _load_eval_note_nodes(settings)
    except FileNotFoundError as exc:
        pytest.skip(str(exc))


def build_in_memory_eval_index(
    settings,
    embed_model,
    *,
    collection_name: str,
    nodes=None,
):
    """Build an ephemeral Chroma-backed index with a caller-supplied embedder."""
    pytest.importorskip("llama_index", reason="llama_index not installed")
    pytest.importorskip("chromadb", reason="chromadb not installed")

    from onclab.eval_embedders import build_in_memory_eval_index as _build_in_memory_eval_index

    return _build_in_memory_eval_index(
        settings,
        embed_model,
        collection_name=collection_name,
        nodes=nodes,
    )


def make_templated_note_nodes(repo_root: Path, count: int, *, seed: int = 42):
    """Build in-memory TextNodes from scripts/seed_data.py templates — no disk writes."""
    pytest.importorskip("llama_index", reason="llama_index not installed")

    import importlib.util

    from llama_index.core.schema import TextNode

    spec = importlib.util.spec_from_file_location(
        "seed_data",
        repo_root / "scripts" / "seed_data.py",
    )
    seed_data = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(seed_data)

    # sdk-review F3: reuse seed_data.generate_notes so paths/IDs stay aligned with disk writes
    nodes: list[TextNode] = []
    for path, text in seed_data.generate_notes(count, seed=seed):
        nodes.append(
            TextNode(
                text=text,
                id_=path,
                metadata={"file_path": path, "file_name": path},
            )
        )
    return nodes


@pytest.fixture(scope="session")
def eval_index(settings):
    """In-memory Chroma index built from synthetic notes — no Ollama, no disk.

    Used by tests/eval/test_retrieval.py so CI can assert retrieval quality
    without a prior `make ingest`. Uses a lightweight bag-of-words embedder
    so keyword overlap drives ranking (good enough for the 8-note corpus).
    """
    from onclab.eval_embedders import BagOfWordsEmbedding

    # sdk-review F1: shared embedder — must match scripts/ragas_eval.py retrieval
    return build_in_memory_eval_index(
        settings,
        BagOfWordsEmbedding(),
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


@pytest.fixture
def build_scale_stress_index(settings):
    """Factory: build an in-memory index from templated notes (for timing ingest in tests)."""

    def _build(repo_root: Path, count: int = 100, *, seed: int = 42):
        nodes = make_templated_note_nodes(repo_root, count=count, seed=seed)
        from llama_index.core.embeddings import MockEmbedding

        # sdk-review F2: assert node count before index build — avoid Chroma private accessors
        assert len(nodes) == count
        index = build_in_memory_eval_index(
            settings,
            MockEmbedding(embed_dim=384),
            collection_name="corpus_scale_stress",
            nodes=nodes,
        )
        return index, len(nodes)

    return _build
