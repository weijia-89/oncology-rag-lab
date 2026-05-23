"""Adversarial injection eval: extraction output must stay within closed vocabulary.

Corpus lives in data/injection_notes/. Run locally with MOCK_LLM=1 (no Ollama):
    MOCK_LLM=1 uv run pytest tests/eval/test_injection.py -q
"""

from __future__ import annotations

from pathlib import Path

import pytest

from onclab.extract import ENTITY_TYPES, ExtractionRequest, extract_entity, value_within_closed_vocabulary
from onclab.llm_client import OllamaClient

pytestmark = pytest.mark.eval


def _injection_note_paths(repo_root: Path) -> list[Path]:
    notes_dir = repo_root / "data" / "injection_notes"
    return sorted(notes_dir.glob("INJ-*.txt"))


@pytest.mark.parametrize("note_path", _injection_note_paths(Path(__file__).resolve().parents[2]))
def test_injection_note_output_stays_in_closed_vocabulary(
    note_path: Path,
    mock_client: OllamaClient,
    settings,
):
    """Each injection-pattern note must not produce a hijacked entity value."""
    note_text = note_path.read_text(encoding="utf-8")
    patient_id = note_path.stem.split("_")[0]
    request = ExtractionRequest(patient_id=patient_id, note_text=note_text)

    result = extract_entity(request, "cancer_type", client=mock_client, settings=settings)

    assert result.entity_type in ENTITY_TYPES
    assert value_within_closed_vocabulary(result.value), (
        f"{patient_id}: value {result.value!r} failed closed-vocabulary check"
    )
