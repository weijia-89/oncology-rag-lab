"""Unit tests for embedding compare helpers and live A/B report schema."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from onclab.embedding_compare import (
    _model_base,
    agreement_rate,
    build_embedding_live_ab_report,
    pick_live_candidate_embed_model,
    validate_embedding_live_ab_report,
)


def test_agreement_rate_empty_returns_one():
    assert agreement_rate([]) == 1.0


def test_agreement_rate_mixed():
    rows = [
        ("q1", "a", "a", True),
        ("q2", "a", "b", False),
        ("q3", "x", "x", True),
    ]
    assert agreement_rate(rows) == pytest.approx(2 / 3)


def test_model_base_strips_tag():
    assert _model_base("nomic-embed-text:latest") == "nomic-embed-text"
    assert _model_base("mxbai-embed-large") == "mxbai-embed-large"


@pytest.mark.parametrize(
    ("available", "baseline", "expected_model", "expected_note_fragment"),
    [
        (
            ["nomic-embed-text", "mxbai-embed-large:latest"],
            "nomic-embed-text",
            "mxbai-embed-large:latest",
            "mxbai-embed-large",
        ),
        (
            ["nomic-embed-text", "all-minilm"],
            "nomic-embed-text",
            "all-minilm",
            "all-minilm",
        ),
        (
            ["nomic-embed-text", "some-custom-embed-v2"],
            "nomic-embed-text",
            "some-custom-embed-v2",
            "embed",
        ),
        (
            ["nomic-embed-text", "qwen3:14b"],
            "nomic-embed-text",
            None,
            "embedding model",
        ),
    ],
)
def test_pick_live_candidate_embed_model(
    available,
    baseline,
    expected_model,
    expected_note_fragment,
):
    with patch("onclab.embedding_compare.list_ollama_model_names", return_value=available):
        model, note = pick_live_candidate_embed_model(baseline, "http://localhost:11434")
    assert model == expected_model
    assert expected_note_fragment in note


def test_validate_embedding_live_ab_report_accepts_minimal():
    report = build_embedding_live_ab_report(
        baseline_model="nomic-embed-text",
        candidate_model="mxbai-embed-large",
        rows=[("q", "id1", "id2", False)],
        ollama_reachable_flag=True,
    )
    assert validate_embedding_live_ab_report(report) == []


def test_validate_embedding_live_ab_report_rejects_missing_keys():
    report = build_embedding_live_ab_report(
        baseline_model="nomic-embed-text",
        candidate_model="mxbai-embed-large",
        rows=[],
        ollama_reachable_flag=True,
    )
    del report["baseline_model"]
    errors = validate_embedding_live_ab_report(report)
    assert any("baseline_model" in e for e in errors)
