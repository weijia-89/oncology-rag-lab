"""Unit tests for stress eval JSON report schema (no live LLM)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from onclab.eval_report import (
    build_stress_eval_report,
    default_stress_report_path,
    validate_stress_eval_report,
    write_stress_eval_report,
)


class _FakeSettings:
    llm_model = "qwen3:14b"
    notes_dir = Path("data/synthetic_notes")
    top_k = 5


def test_build_and_validate_stress_eval_report_roundtrip():
    payload = build_stress_eval_report(
        repo_root=Path("/tmp/onclab"),
        settings=_FakeSettings(),
        corpus_size=100,
        queries=("stage lung cancer", "glioblastoma patient"),
        per_query_ms=[12.5, 8.1],
        chunks_per_query=[3, 5],
        total_elapsed_s=42.0,
        mock_llm=True,
        run_id="test-run-001",
    )
    validate_stress_eval_report(payload)
    assert payload["eval_kind"] == "corpus_scale_stress"
    assert payload["rag_metrics"] == "unavailable"
    assert payload["token_usage"] is None
    assert len(payload["cases"]) == 2
    assert payload["summary"]["pass_rate"] == 1.0


def test_write_stress_eval_report_creates_file(tmp_path: Path):
    payload = build_stress_eval_report(
        repo_root=tmp_path,
        settings=_FakeSettings(),
        corpus_size=10,
        queries=("one",),
        per_query_ms=[1.0],
        chunks_per_query=[2],
        total_elapsed_s=3.0,
        mock_llm=True,
    )
    out = write_stress_eval_report(tmp_path, payload)
    assert out == default_stress_report_path(tmp_path)
    loaded = json.loads(out.read_text(encoding="utf-8"))
    validate_stress_eval_report(loaded)


def test_validate_rejects_missing_keys():
    with pytest.raises(ValueError, match="missing keys"):
        validate_stress_eval_report({"eval_kind": "corpus_scale_stress"})
