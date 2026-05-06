"""Unit tests for retry logic in OllamaClient.generate().

We test the retry loop in OllamaClient.generate() without touching the real
Ollama server. The strategy:
  - Keep MOCK_LLM=1 (set by conftest) so the module stays importable.
  - Directly patch MOCK_LLM to False on the module object for tests that
    need the real code path, rather than reloading the module (reloading
    would poison the module cache and break other test files).
  - Patch time.sleep so the 2-second backoff doesn't slow the test suite.
"""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest

import onclab.llm_client as _llm_mod
from onclab.llm_client import OllamaClient

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_generate_succeeds_on_first_try():
    """No retry needed — a single call returns immediately."""
    client = OllamaClient(host="http://localhost:11434", model="qwen3:14b")
    fake_response = {"response": "  non-small cell lung cancer  "}
    client._client = MagicMock()
    client._client.generate.return_value = fake_response

    with patch.object(_llm_mod, "MOCK_LLM", False), patch("time.sleep") as mock_sleep:
        result = client.generate("extract cancer_type")

    assert result == "non-small cell lung cancer"
    assert client._client.generate.call_count == 1
    mock_sleep.assert_not_called()


def test_generate_retries_on_transient_failure_then_succeeds():
    """First two attempts raise; third attempt succeeds."""
    client = OllamaClient(host="http://localhost:11434", model="qwen3:14b")
    fake_response = {"response": "glioblastoma"}
    client._client = MagicMock()
    client._client.generate.side_effect = [
        ConnectionError("Ollama is restarting"),
        ConnectionError("Ollama is still restarting"),
        fake_response,
    ]

    with patch.object(_llm_mod, "MOCK_LLM", False), patch("time.sleep") as mock_sleep:
        result = client.generate("extract cancer_type")

    assert result == "glioblastoma"
    assert client._client.generate.call_count == 3
    # Should have slept twice (between attempt 1→2 and 2→3).
    assert mock_sleep.call_count == 2
    mock_sleep.assert_has_calls([call(2), call(2)])


def test_generate_raises_after_all_retries_exhausted():
    """All 3 attempts fail — RuntimeError is raised, not swallowed."""
    client = OllamaClient(host="http://localhost:11434", model="qwen3:14b")
    client._client = MagicMock()
    client._client.generate.side_effect = TimeoutError("Ollama timeout")

    with patch.object(_llm_mod, "MOCK_LLM", False), patch("time.sleep"):
        with pytest.raises(RuntimeError, match="failed after 3 attempts"):
            client.generate("extract cancer_type")

    assert client._client.generate.call_count == 3


def test_mock_mode_never_calls_real_client():
    """In MOCK_LLM=1 mode the real client is never touched — no retry needed."""
    client = OllamaClient(host="http://localhost:11434", model="qwen3:14b")
    client._client = MagicMock()

    # MOCK_LLM is already True (set by conftest before module import).
    with patch("time.sleep") as mock_sleep:
        result = client.generate("any prompt")

    assert client._client.generate.call_count == 0
    mock_sleep.assert_not_called()
    assert isinstance(result, str)
