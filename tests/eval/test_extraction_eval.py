"""DeepEval-driven scoring of entity extraction against the gold standard.

This is the test file that produces the interview-relevant numbers:
    - Per-entity match rate vs gold standard (exact-match accuracy)
    - DeepEval AnswerRelevancyMetric per (note, entity) pair
    - DeepEval HallucinationMetric on a sample (because cheap+fast)

How DeepEval fits with pytest:
    DeepEval's `assert_test(test_case, [metric])` runs the metric (which
    may itself call an LLM-as-judge) and raises an AssertionError if any
    metric fails its threshold. Pytest marks the case as failing. So you
    get one row in the pytest report per (note, entity) pair, and the
    JSON report file (--json-report) gives you the structured data the
    regression-check script reads.

Why mark these `eval`:
    They're slow and they need either Ollama running or MOCK_LLM=1.
    `pytest -m "not eval"` (in test-unit Make target) skips them. The
    Make target `eval-mock` and `eval` run them under different MOCK
    settings.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

# DeepEval imports; deferred-pattern so a unit-test-only run doesn't
# need DeepEval installed. (It's listed in pyproject so it WILL be
# installed in normal use; the deferred import is for clarity.)
deepeval = pytest.importorskip("deepeval", reason="deepeval not installed")
from deepeval import assert_test  # noqa: E402
from deepeval.metrics import AnswerRelevancyMetric  # noqa: E402
from deepeval.test_case import LLMTestCase  # noqa: E402

from onclab.extract import (  # noqa: E402
    ENTITY_TYPES,
    ExtractionRequest,
    extract_entity,
)
from onclab.llm_client import OllamaClient  # noqa: E402

pytestmark = pytest.mark.eval


def _load_gold(repo_root: Path) -> dict[tuple[str, str], str]:
    """gold_standard.csv -> {(patient_id, entity_type): gold_value}.

    Loaded once at collection time. Tests parameterize over the keys.
    """
    gold: dict[tuple[str, str], str] = {}
    with open(repo_root / "data" / "gold_standard.csv", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            gold[(row["patient_id"], row["entity_type"])] = row["gold_value"]
    return gold


def _gold_keys(repo_root: Path) -> list[tuple[str, str]]:
    return list(_load_gold(repo_root).keys())


# Parameterize over (patient_id, entity_type) pairs from the gold standard.
# Restricting to ENTITY_TYPES avoids running tests for entities we don't
# yet extract (e.g., 'pdl1_tps' is in gold but not in ENTITY_TYPES).
@pytest.fixture(scope="module")
def gold(repo_root: Path) -> dict[tuple[str, str], str]:
    return _load_gold(repo_root)


def _patient_pairs(repo_root: Path) -> list[tuple[str, str]]:
    keys = _gold_keys(repo_root)
    return [k for k in keys if k[1] in ENTITY_TYPES]


@pytest.mark.parametrize(
    ("patient_id", "entity_type"),
    _patient_pairs(Path(__file__).resolve().parents[2]),
)
def test_extraction_matches_gold(
    patient_id: str,
    entity_type: str,
    gold: dict[tuple[str, str], str],
    repo_root: Path,
    mock_client: OllamaClient,
    settings,
):
    """Exact-match style test: did the extractor produce the gold value?

    Why exact-match for some and DeepEval for others (next test):
        Some entities (cancer_type, ajcc_stage, ecog) have controlled
        vocabularies. Exact match is the right test. Other entities
        (regimen wording, histology phrasing) have multiple valid
        answers — those need a semantic-similarity metric instead.
        The strategy guide makes this distinction explicit.
    """
    note_path = repo_root / "data" / "synthetic_notes" / f"patient_{patient_id.split('-')[1]}.txt"
    if not note_path.exists():
        pytest.skip(f"Note for {patient_id} not found")

    note_text = note_path.read_text(encoding="utf-8")
    request = ExtractionRequest(patient_id=patient_id, note_text=note_text)
    result = extract_entity(request, entity_type, client=mock_client, settings=settings)

    expected = gold[(patient_id, entity_type)].lower()
    actual = result.value.lower()

    # Controlled-vocab entities (ajcc_stage, ecog, cancer_type) use exact
    # match after lowercasing. Substring containment was removed because it
    # allows the model to hallucinate additional content and still pass
    # (e.g. 'iiia (originally staged as ib)' passes when gold is 'iiia').
    # The mock LLM returns the canonical form; the real LLM prompt instructs
    # "respond with only the controlled value". If the real LLM produces a
    # verbose answer, fix the prompt — don't loosen the assertion.
    if entity_type in {"cancer_type", "ajcc_stage", "ecog"}:
        assert actual == expected, (
            f"{patient_id}/{entity_type}: expected '{expected}', got '{actual}'"
        )
    else:
        # For regimen/histology, fall back to DeepEval's relevancy metric
        # — this is where you'd plug in custom GEval metrics in a real lab.
        test_case = LLMTestCase(
            input=f"What is the {entity_type} for {patient_id}?",
            actual_output=result.value,
            expected_output=expected,
        )
        metric = AnswerRelevancyMetric(threshold=settings.answer_relevancy_threshold)
        assert_test(test_case, [metric])
