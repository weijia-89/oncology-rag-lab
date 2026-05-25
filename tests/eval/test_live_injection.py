"""Live Ollama injection corpus smoke (operator-only).

Skipped unless ``ONCLAB_RUN_LIVE_INJECTION=1`` and Ollama responds at
``/api/tags``. Uses real ``OllamaClient`` (not ``MOCK_LLM``) on a subset
of ``data/injection_notes/``; asserts extracted values stay within the
closed-vocabulary guard (or fall back to ``unknown``).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from onclab.embedding_compare import ollama_reachable
from onclab.extract import ExtractionRequest, extract_entity, value_within_closed_vocabulary
from onclab.llm_client import OllamaClient

pytestmark = [pytest.mark.eval, pytest.mark.live_injection]

_REPO_ROOT = Path(__file__).resolve().parents[2]
_INJECTION_NOTES_DIR = _REPO_ROOT / "data" / "injection_notes"

# Representative attack shapes — full corpus stays on mock + simulated hijack.
_LIVE_INJECTION_SAMPLE = (
    "INJ-001_instruction_override.txt",
    "INJ-008_delimiter_escape.txt",
)


@pytest.fixture(scope="module")
def live_injection_gate(settings):
    """Skip unless operator opted in and Ollama is reachable."""
    if os.environ.get("ONCLAB_RUN_LIVE_INJECTION") != "1":
        pytest.skip("ONCLAB_RUN_LIVE_INJECTION=1 not set")
    if not ollama_reachable(settings.ollama_host):
        pytest.skip(f"Ollama not reachable at {settings.ollama_host}")


@pytest.fixture
def live_client(settings, live_injection_gate):
    """Real Ollama client for this module only (conftest defaults MOCK_LLM=1)."""
    prev = os.environ.pop("MOCK_LLM", None)
    try:
        yield OllamaClient(host=settings.ollama_host, model=settings.llm_model)
    finally:
        if prev is not None:
            os.environ["MOCK_LLM"] = prev


@pytest.mark.parametrize("note_file", _LIVE_INJECTION_SAMPLE)
def test_live_injection_note_respects_closed_vocabulary_guard(
    note_file: str,
    live_client: OllamaClient,
    settings,
):
    """Live LLM extraction on adversarial notes must not return hijack-shaped values."""
    note_path = _INJECTION_NOTES_DIR / note_file
    assert note_path.is_file(), f"missing injection note: {note_path}"
    note_text = note_path.read_text(encoding="utf-8")
    patient_id = note_path.stem.split("_")[0]
    request = ExtractionRequest(patient_id=patient_id, note_text=note_text)

    result = extract_entity(
        request,
        "cancer_type",
        client=live_client,
        settings=settings,
    )

    assert result.entity_type == "cancer_type"
    assert value_within_closed_vocabulary(result.value), (
        f"{note_file}: live model returned guard-violating value {result.value!r}"
    )
