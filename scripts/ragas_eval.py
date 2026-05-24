#!/usr/bin/env python3
"""Run Ragas-style RAG eval and write reports/eval_report.json.

Uses DeepEval native metrics (see src/onclab/ragas_eval.py). MOCK_LLM=1 runs
lexical overlap proxies without Ollama judge calls.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from onclab.config import load_settings  # noqa: E402
from onclab.eval_embedders import BagOfWordsEmbedding, build_in_memory_eval_index  # noqa: E402
from onclab.ragas_eval import run_rag_eval  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run RAG eval and write eval_report.json")
    parser.add_argument(
        "--output",
        type=Path,
        default=_REPO_ROOT / "reports" / "eval_report.json",
        help="Path for JSON report (default: reports/eval_report.json)",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=_REPO_ROOT / "reports" / "ragas_baseline.json",
        help="Baseline report for regression pass_rate (placeholder until first seed)",
    )
    args = parser.parse_args()

    settings = load_settings(
        notes_dir=_REPO_ROOT / "data" / "synthetic_notes",
        persist_dir=_REPO_ROOT / "data" / "chroma_db",
    )
    # sdk-review F1: shared embedder/index builder — same retrieval as pytest eval_index
    index = build_in_memory_eval_index(
        settings,
        BagOfWordsEmbedding(),
        collection_name="ragas_eval_notes",
    )
    report = run_rag_eval(
        _REPO_ROOT,
        eval_index=index,
        output_path=args.output,
        baseline_path=args.baseline,
        settings=settings,
    )
    print(f"wrote {args.output} pass_rate={report['pass_rate']:.3f}")
    return 0 if report["regression_status"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
