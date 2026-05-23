"""Shared DeepEval metrics for oncology extraction eval."""

from __future__ import annotations

import os
from typing import Any

from deepeval.metrics import GEval
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.models.llms.ollama_model import OllamaModel
from deepeval.test_case import LLMTestCaseParams

from onclab.config import Settings, load_settings

REGIMEN_HISTOLOGY_GEVAL_CRITERIA = """
Compare ACTUAL_OUTPUT to EXPECTED_OUTPUT for regimen or histology extraction.
Score 1.0 only when the clinical meaning matches; surface wording may differ.
Treat common oncology abbreviations as equivalent (e.g., FOLFOX, ABVD, EBRT, TMZ).
Accept TMZ as temozolomide and Stupp-style phrasing when the same concurrent/adjuvant plan is described.
Allow reordering of drug names and optional detail (dose, cycles) when the same regimen is named.
For histology, accept subtype qualifiers when the same tumor lineage is identified
(e.g., adenocarcinoma vs invasive ductal carcinoma when equivalent).
Require the same line of therapy and intent (curative, adjuvant, neoadjuvant, palliative)
when inferable from the strings.
Reject unrelated regimens, wrong drug class swaps, or histology from a different family (e.g., lymphoma vs carcinoma).
Reject answers that add or drop a major systemic agent that changes the treatment plan.
Minor punctuation, casing, and connector differences (+, and, with) should not fail an otherwise correct pair.
""".strip()


class _MockGEvalJudge(DeepEvalBaseLLM):
    """Deterministic GEval judge when MOCK_LLM=1 — no OpenAI or Ollama judge call."""

    _MOCK_STEPS = (
        "Compare ACTUAL_OUTPUT and EXPECTED_OUTPUT for clinical equivalence.",
        "Allow oncology abbreviations and synonym regimens when meaning matches.",
        "Reject unrelated regimens, wrong drug classes, or wrong histology families.",
    )

    def load_model(self, *args: Any, **kwargs: Any) -> _MockGEvalJudge:
        return self

    def generate(self, prompt: str, **kwargs: Any) -> str:
        return '{"score": 10, "reason": "MOCK_LLM=1: deterministic oracle pass"}'

    async def a_generate(self, prompt: str, **kwargs: Any) -> str:
        return self.generate(prompt, **kwargs)

    def get_model_name(self, *args: Any, **kwargs: Any) -> str:
        return "mock-geval-judge"

    def _mock_schema_response(self, schema: Any) -> Any:
        if schema is None:
            return self.generate("")
        schema_name = getattr(schema, "__name__", "")
        if schema_name == "Steps":
            return schema(steps=list(self._MOCK_STEPS))
        return schema(score=10.0, reason="MOCK_LLM=1: deterministic oracle pass")

    def generate_with_schema(self, prompt: str, schema: Any = None, **kwargs: Any) -> Any:
        return self._mock_schema_response(schema)

    async def a_generate_with_schema(self, prompt: str, schema: Any = None, **kwargs: Any) -> Any:
        return self._mock_schema_response(schema)


def _geval_judge_model(settings: Settings) -> DeepEvalBaseLLM | OllamaModel:
    if os.environ.get("MOCK_LLM", "0") == "1" or settings.mock_llm:
        return _MockGEvalJudge(model="mock-geval-judge")
    return OllamaModel(model=settings.llm_model, base_url=settings.ollama_host)


def regimen_histology_geval(threshold: float) -> GEval:
    """Clinical-plausibility GEval for regimen and histology vs gold labels."""
    settings = load_settings()
    return GEval(
        name="Regimen/Histology Clinical Plausibility",
        criteria=REGIMEN_HISTOLOGY_GEVAL_CRITERIA,
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        threshold=threshold,
        model=_geval_judge_model(settings),
    )
