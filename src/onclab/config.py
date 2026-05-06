"""Centralized config.

Why a config module instead of magic strings inside each file:
    Every model name, path, or threshold gets read from environment variables
    with a sensible default. Change a default here, and the entire pipeline
    picks it up. This is the same pattern Ontada's pipeline almost certainly
    uses for model versioning — no hard-coded "qwen3:14b" sprinkled across
    20 files when you want to A/B test against a new model.

Why pydantic-settings (or in this case, a plain dataclass with os.environ):
    Pydantic validates types at load time. We're using a plain dataclass +
    os.getenv to keep the dep surface small; if config grows complicated,
    swap to pydantic-settings without changing the public API.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    """Read once at import time; immutable for the rest of the process.

    `frozen=True` means you can't accidentally mutate `settings.llm_model`
    halfway through a pipeline run. If you need to swap models for drift
    testing, you build a *new* Settings instance and pass it explicitly —
    there's no global mutation, which makes the drift comparison test honest.
    """

    # ---- Connection ----
    ollama_host: str
    """URL of the local Ollama server. Default: http://localhost:11434."""

    # ---- Models ----
    llm_model: str
    """Name of the LLM model in Ollama (e.g., 'qwen3:14b')."""

    embed_model: str
    """Name of the embedding model in Ollama (e.g., 'nomic-embed-text')."""

    # ---- Storage ----
    persist_dir: Path
    """Where ChromaDB writes its files. Resets on `make clean`."""

    notes_dir: Path
    """Where the synthetic clinical notes live."""

    # ---- Retrieval knobs ----
    chunk_size: int = 500
    """Tokens (well, characters via the recursive splitter) per chunk.
    500 is a reasonable default for clinical-style notes — long enough
    to keep a paragraph intact, short enough that retrieval doesn't return
    huge slabs of irrelevant text. Smaller chunks = more precise retrieval
    but more chunks to manage; larger = the opposite."""

    chunk_overlap: int = 50
    """Characters of overlap between adjacent chunks. Overlap exists so
    that a fact straddling a chunk boundary doesn't get cut in half. 10%
    of chunk_size is a common rule of thumb."""

    top_k: int = 3
    """Number of chunks to retrieve per query. With only 8 small notes
    you don't need a large top_k. In production-scale corpora you'd tune
    this empirically against retrieval-precision metrics."""

    # ---- Eval thresholds ----
    # These are the "what counts as passing" bars for the DeepEval suite.
    # Setting them in config (not in test code) is so that you can tighten
    # them as the pipeline matures without editing test files.
    hallucination_threshold: float = 0.5
    faithfulness_threshold: float = 0.7
    answer_relevancy_threshold: float = 0.7

    # ---- Behavior toggles ----
    mock_llm: bool = False
    """When True, llm_client returns canned responses instead of calling Ollama.
    Lets unit tests run in CI without a GPU box."""

    phoenix_project: str = "oncology-rag-lab"


def load_settings(*, persist_dir: Path | None = None, notes_dir: Path | None = None) -> Settings:
    """Construct Settings from environment variables.

    Pattern: explicit overrides > env vars > defaults.
    The kwargs let CLI commands force a specific path without touching env.
    """
    repo_root = Path(__file__).resolve().parents[2]  # src/onclab/config.py -> ../.. = project root

    return Settings(
        ollama_host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        llm_model=os.getenv("ONCLAB_LLM_MODEL", "qwen3:14b"),
        embed_model=os.getenv("ONCLAB_EMBED_MODEL", "nomic-embed-text"),
        persist_dir=persist_dir or (repo_root / "data" / "chroma_db"),
        notes_dir=notes_dir or (repo_root / "data" / "synthetic_notes"),
        mock_llm=os.getenv("MOCK_LLM", "0") == "1",
        phoenix_project=os.getenv("PHOENIX_PROJECT_NAME", "oncology-rag-lab"),
    )
