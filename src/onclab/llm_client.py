"""Ollama wrapper with a mock mode for tests.

Why a wrapper instead of using the `ollama` Python lib directly:
    Two reasons.
    1. **MOCK_LLM toggle.** Unit tests need to run in CI without a GPU
       box and without the 9 GB of model weights pulled. The wrapper
       reads MOCK_LLM=1 and returns canned responses keyed by prompt
       fingerprint. The same calling code works in both modes.
    2. **Single chokepoint for observability and retries.** When you want
       to add a `tenacity` retry, a request-time logger, or a Phoenix span,
       you do it in one place instead of every call site.

Mock-as-perfect-oracle pattern:
    The mock responses are auto-derived from data/gold_standard.csv at
    module import time. In MOCK_LLM=1 mode, every extract:patient:entity
    call returns the gold value (formatted as JSON to match the real
    response shape). That makes MOCK_LLM=1 a "perfect oracle" smoke test:
    if eval-mock fails, the bug is in the harness, not the model. If
    the real eval fails on cases eval-mock passed, the model is the
    suspect. The two modes triangulate where the failure lives.

Pattern note: Wei's other portfolio projects (network-scanner,
wcag-auditor, no-log-rsvp, android-hardener) use this same shape.
The shared convention makes the portfolio readable as a coherent set.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import ollama

# ---------------------------------------------------------------------------
# Mock layer
# ---------------------------------------------------------------------------
# Read MOCK_LLM at import time. Setting MOCK_LLM after the module is imported
# does NOT switch the wrapper into mock mode — that's intentional, so a single
# pytest run has consistent behavior throughout.
MOCK_LLM: bool = os.environ.get("MOCK_LLM", "0") == "1"


def _build_mocks_from_gold() -> dict[str, str]:
    """Read data/gold_standard.csv and synthesize JSON mock responses.

    Why JSON and not plain strings: the real LLM call returns text that
    extract._parse_response runs through a regex looking for `{...}`. A
    plain-string mock would fail that regex and the extractor would
    fall through to the "parse_failure" path — which masks bugs in
    everything downstream of parsing. Returning the JSON shape the
    real model is asked to produce gives MOCK_LLM=1 a fighting chance
    to exercise the full happy path.

    Why "perfect oracle": the mock returns the gold value verbatim. So
    an eval failure under MOCK_LLM=1 means a harness bug (CSV missing,
    parametrize wrong, fixture path broken) — never a model bug. The
    harness/model failure modes get separated cleanly.
    """
    repo_root = Path(__file__).resolve().parents[2]
    gold_path = repo_root / "data" / "gold_standard.csv"
    out: dict[str, str] = {}

    if not gold_path.exists():
        # Bootstrapping: gold standard not yet created. Fall back to a
        # single default. Tests that need real fixtures will fail loudly,
        # but `import onclab` won't crash.
        return {"default": json.dumps({"value": "unknown", "confidence": 0.0, "rationale": ""})}

    with open(gold_path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            patient_id = row["patient_id"].strip().lower()
            entity_type = row["entity_type"].strip()
            value = row["gold_value"].strip()
            key = f"extract:{patient_id}:{entity_type}"
            out[key] = json.dumps(
                {"value": value, "confidence": 0.95, "rationale": "mock-from-gold"}
            )

    out["default"] = json.dumps({"value": "unknown", "confidence": 0.0, "rationale": "mock-default"})
    return out


# Built once at import. Stable across the whole test session.
MOCK_RESPONSES: dict[str, str] = _build_mocks_from_gold()


def _fingerprint(*parts: str) -> str:
    """Stable short key from arbitrary string parts.

    We don't currently need this for the gold-derived mocks (their keys
    are human-readable strings) but the helper exists for the next step:
    when prompt fingerprints become the cache key for response logging
    or retry replay, you swap callers to this without touching mocks.
    """
    joined = "|".join(parts)
    return hashlib.sha256(joined.encode()).hexdigest()[:8]


# ---------------------------------------------------------------------------
# Real client
# ---------------------------------------------------------------------------
class OllamaClient:
    """Stateful wrapper. Holds host + default model, exposes generate()."""

    def __init__(self, *, host: str, model: str) -> None:
        # The ollama lib's Client takes a host URL. If host is None, it
        # falls back to OLLAMA_HOST or http://localhost:11434.
        self._client = ollama.Client(host=host)
        self._model = model
        self._host = host

    def generate(self, prompt: str, *, mock_key: str | None = None, **opts: Any) -> str:
        """Run a single completion. Returns the raw text.

        `mock_key` is the fixture lookup string for MOCK_LLM mode. Pass
        something stable (e.g., "extract:syn-001:cancer_type"). If you
        omit it, mock mode falls back to the "default" entry.

        `**opts` is forwarded to ollama (e.g., temperature, top_p).
        We're not setting `format='json'` here because (a) Ollama's JSON
        mode quality is model-dependent and (b) we use Pydantic
        downstream to parse a freer-form response. If a model
        hallucinates outside the schema we'd rather catch that with a
        parse error than get silently-coerced wrong values.
        """
        if MOCK_LLM:
            if mock_key and mock_key in MOCK_RESPONSES:
                return MOCK_RESPONSES[mock_key]
            return MOCK_RESPONSES["default"]

        # Real call — up to 3 attempts with 2-second backoff.
        # Ollama can drop connections mid-model-load or under heavy load;
        # a single timeout aborting the whole extraction run is a bad outcome
        # for a 400-second pipeline that's 90% done. Three tries with short
        # sleeps handles the "Ollama restarted mid-eval" case without masking
        # genuine model failures (the exception re-raises on the final attempt).
        _max_retries = 3
        _backoff_seconds = 2
        last_exc: Exception | None = None
        for attempt in range(_max_retries):
            try:
                response = self._client.generate(model=self._model, prompt=prompt, options=opts)
                return response["response"].strip()
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt < _max_retries - 1:
                    time.sleep(_backoff_seconds)
        raise RuntimeError(
            f"Ollama generate failed after {_max_retries} attempts on model {self._model!r}."
        ) from last_exc

    def chat(self, messages: list[dict[str, str]], *, mock_key: str | None = None) -> str:
        """Multi-turn variant. Same MOCK_LLM contract as generate()."""
        if MOCK_LLM:
            return MOCK_RESPONSES.get(mock_key or "default", MOCK_RESPONSES["default"])
        response = self._client.chat(model=self._model, messages=messages)
        return response["message"]["content"].strip()
