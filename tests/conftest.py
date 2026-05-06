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
