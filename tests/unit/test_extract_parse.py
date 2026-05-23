"""Unit tests for the JSON parser inside extract.py.

These tests intentionally do NOT call the LLM. They feed canned response
strings to `_parse_response` and check the parser's behavior in three
regimes: clean JSON, JSON wrapped in chatter, and unparseable garbage.

Why unit-test the parser separately:
    The parser is the boundary between "what the LLM emitted" and "what
    the rest of our pipeline trusts as structured data". If it ever
    silently returned wrong values, every downstream metric would be
    measuring the wrong thing. Catching parser regressions in milliseconds
    means an eval failure later is unambiguously a model-quality issue,
    not a parsing issue.
"""

from __future__ import annotations

from onclab.extract import _parse_response


def test_parse_clean_json():
    raw = '{"value": "non-small cell lung cancer", "confidence": 0.92, "rationale": "stated explicitly"}'
    parsed = _parse_response(raw)
    assert parsed is not None
    assert parsed.value == "non-small cell lung cancer"
    assert parsed.confidence == 0.92
    assert "explicitly" in parsed.rationale


def test_parse_json_with_leading_chatter():
    """Small models love to preface JSON with 'Here is the answer:'.
    The parser must tolerate that without manual cleanup."""
    raw = 'Sure! Here is the answer:\n{"value": "IIIA", "confidence": 0.81, "rationale": "T2bN2M0"}'
    parsed = _parse_response(raw)
    assert parsed is not None
    assert parsed.value == "IIIA"


def test_parse_unparseable_returns_none():
    """When the model goes off the rails entirely, parser returns None
    so the caller can synthesize an 'unknown' answer rather than crashing
    the whole eval run."""
    raw = "I'm sorry, I cannot extract that information."
    assert _parse_response(raw) is None


def test_parse_partial_json_is_lenient():
    """If the model omits 'rationale', we still parse the value out."""
    raw = '{"value": "ABVD", "confidence": 0.7}'
    parsed = _parse_response(raw)
    assert parsed is not None
    assert parsed.value == "ABVD"
    assert parsed.rationale == ""  # default


def test_parse_handles_braces_in_string_values():
    """Regression: the old r"\\{[^{}]*\\}" regex broke on rationales like
    'FOLFOX {6 cycles}' or staging codes 'T2bN2M0 {per CT}'. The brace-aware
    scanner must walk past braces that live inside string values."""
    raw = (
        '{"value": "FOLFOX", "confidence": 0.88, '
        '"rationale": "regimen recorded as FOLFOX {6 cycles} per oncology note"}'
    )
    parsed = _parse_response(raw)
    assert parsed is not None
    assert parsed.value == "FOLFOX"
    assert "{6 cycles}" in parsed.rationale


def test_parse_clamps_out_of_range_confidence():
    """Small models occasionally emit confidence > 1.0 ('confidence: 95'
    meaning 95%). _parse_response should clamp into [0, 1] instead of
    letting an absurd value poison downstream regression gates."""
    raw = '{"value": "IIIA", "confidence": 95, "rationale": "model returned a percent"}'
    parsed = _parse_response(raw)
    assert parsed is not None
    assert parsed.confidence == 1.0


def test_parse_rejects_non_numeric_confidence():
    """If confidence is unparseable, return None rather than synthesizing
    a fake 0.0 that downstream code might trust."""
    raw = '{"value": "IIIA", "confidence": "high", "rationale": "string instead of number"}'
    assert _parse_response(raw) is None
