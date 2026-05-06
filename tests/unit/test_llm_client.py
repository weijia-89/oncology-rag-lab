"""Unit tests for the Ollama wrapper.

In MOCK_LLM=1 mode (which conftest.py sets globally), generate() and chat()
should never reach out to Ollama. These tests verify the mock plumbing
so we can trust unit-test runs without an Ollama server.
"""

from __future__ import annotations

from onclab.llm_client import MOCK_RESPONSES, OllamaClient


def test_mock_returns_known_key(mock_client: OllamaClient):
    out = mock_client.generate("ignored prompt", mock_key="extract:syn-001:cancer_type")
    assert out == MOCK_RESPONSES["extract:syn-001:cancer_type"]


def test_mock_falls_back_to_default_for_unknown_key(mock_client: OllamaClient):
    out = mock_client.generate("ignored prompt", mock_key="nope:nothing:here")
    assert out == MOCK_RESPONSES["default"]


def test_mock_works_without_key(mock_client: OllamaClient):
    """A callsite that forgets to pass mock_key should still return *something*
    deterministic (the 'default' canned answer) instead of crashing."""
    out = mock_client.generate("ignored prompt")
    assert out == MOCK_RESPONSES["default"]
