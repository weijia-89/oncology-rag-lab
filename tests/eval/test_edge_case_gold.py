"""DeepEval scoring of entity extraction against the edge-case gold standard.

Mirrors ``test_extraction_eval.py`` but loads ``data/gold_standard_edge_cases.csv``
(wide-row schema) and reads notes from ``data/edge_case_notes/``. Under
``MOCK_LLM=1``, edge-case mock keys are registered at module scope so the
harness behaves as a perfect oracle — same triangulation as the base gold suite.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

deepeval = pytest.importorskip("deepeval", reason="deepeval not installed")
from deepeval import assert_test  # noqa: E402
from deepeval.test_case import LLMTestCase  # noqa: E402

from onclab.eval_metrics import regimen_histology_geval  # noqa: E402
from onclab.extract import (  # noqa: E402
    ENTITY_TYPES,
    ExtractionRequest,
    extract_entity,
)
from onclab.llm_client import MOCK_RESPONSES, OllamaClient  # noqa: E402

pytestmark = pytest.mark.eval

# Wide CSV columns that map to extractable ENTITY_TYPES (no histology column).
_EDGE_ENTITY_COLUMNS = ("cancer_type", "ajcc_stage", "regimen", "ecog")


def _load_edge_gold(repo_root: Path) -> dict[tuple[str, str], str]:
    """gold_standard_edge_cases.csv -> {(patient_id, entity_type): gold_value}.

    Normalizes the wide-row edge-case schema to the same long shape as
    ``_load_gold`` in ``test_extraction_eval.py``.
    """
    gold: dict[tuple[str, str], str] = {}
    path = repo_root / "data" / "gold_standard_edge_cases.csv"
    with open(path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            patient_id = row["patient_id"].strip()
            for entity_type in _EDGE_ENTITY_COLUMNS:
                if entity_type not in ENTITY_TYPES:
                    continue
                gold[(patient_id, entity_type)] = row[entity_type].strip()
    return gold


def _edge_note_path(repo_root: Path, patient_id: str) -> Path | None:
    """Resolve ``data/edge_case_notes/{patient_id}_*.txt`` (expects one match)."""
    matches = sorted((repo_root / "data" / "edge_case_notes").glob(f"{patient_id}_*.txt"))
    if len(matches) == 1:
        return matches[0]
    return None


def _edge_patient_pairs(repo_root: Path) -> list[tuple[str, str]]:
    return list(_load_edge_gold(repo_root).keys())


@pytest.fixture(scope="module", autouse=True)
def _register_edge_case_mocks(repo_root: Path) -> None:
    """Seed MOCK_RESPONSES for EC-* keys; base gold only loads SYN-* at import."""
    for (patient_id, entity_type), value in _load_edge_gold(repo_root).items():
        key = f"extract:{patient_id.lower()}:{entity_type}"
        MOCK_RESPONSES[key] = json.dumps(
            {"value": value, "confidence": 0.95, "rationale": "mock-from-edge-gold"}
        )


@pytest.fixture(scope="module")
def edge_gold(repo_root: Path) -> dict[tuple[str, str], str]:
    return _load_edge_gold(repo_root)


@pytest.mark.parametrize(
    ("patient_id", "entity_type"),
    _edge_patient_pairs(Path(__file__).resolve().parents[2]),
)
def test_edge_case_extraction_matches_gold(
    patient_id: str,
    entity_type: str,
    edge_gold: dict[tuple[str, str], str],
    repo_root: Path,
    mock_client: OllamaClient,
    settings,
):
    """Score each edge-case note against wide-CSV gold for extractable entities."""
    note_path = _edge_note_path(repo_root, patient_id)
    if note_path is None:
        pytest.skip(f"Edge-case note for {patient_id} not found or ambiguous")

    note_text = note_path.read_text(encoding="utf-8")
    request = ExtractionRequest(patient_id=patient_id, note_text=note_text)
    result = extract_entity(request, entity_type, client=mock_client, settings=settings)

    expected = edge_gold[(patient_id, entity_type)].lower()
    actual = result.value.lower()

    if entity_type in {"cancer_type", "ajcc_stage", "ecog"}:
        assert actual == expected, (
            f"{patient_id}/{entity_type}: expected '{expected}', got '{actual}'"
        )
    else:
        # regimen: GEval rubric (blinded-trial placeholders, dose-modified FOLFOX, etc.)
        test_case = LLMTestCase(
            input=f"What is the {entity_type} for {patient_id}?",
            actual_output=result.value,
            expected_output=expected,
        )
        metric = regimen_histology_geval(threshold=settings.answer_relevancy_threshold)
        assert_test(test_case, [metric])
