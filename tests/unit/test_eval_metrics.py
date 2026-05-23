"""Unit tests for shared DeepEval metric factories."""

from __future__ import annotations

import pytest

deepeval = pytest.importorskip("deepeval", reason="deepeval not installed")

from onclab.eval_metrics import REGIMEN_HISTOLOGY_GEVAL_CRITERIA, regimen_histology_geval  # noqa: E402

REQUIRED_GUARD_PHRASES = (
    "FOLFOX",
    "TMZ",
    "temozolomide",
    "abbreviation",
    "line of therapy",
    "drug class",
    "histology",
    "unrelated",
)


def test_regimen_histology_criteria_includes_clinical_guards():
    criteria = REGIMEN_HISTOLOGY_GEVAL_CRITERIA.lower()
    for phrase in REQUIRED_GUARD_PHRASES:
        assert phrase.lower() in criteria, f"missing guard phrase: {phrase!r}"


def test_regimen_histology_geval_smoke_instantiates(settings):
    """MOCK_LLM=1 (set in conftest) avoids live OpenAI/Ollama judge calls."""
    metric = regimen_histology_geval(threshold=settings.answer_relevancy_threshold)
    assert metric.threshold == settings.answer_relevancy_threshold
    assert metric.criteria == REGIMEN_HISTOLOGY_GEVAL_CRITERIA
    assert metric.name == "Regimen/Histology Clinical Plausibility"
