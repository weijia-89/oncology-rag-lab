"""Live Ollama embedding A/B: compare top-1 retrieval ids between two embedders.

Writes ``reports/embedding_live_ab_report.json`` with per-query baseline vs
candidate top-1 chunk ids and an agreement rate. Uses in-memory Chroma only —
no ``make ingest``, no persisted ``data/chroma_db/``.

Exit codes:
    0 — Ollama reachable; report written
    2 — Ollama unreachable (skipped)
"""

from __future__ import annotations

import sys
from pathlib import Path

from onclab.config import load_settings
from onclab.embedding_compare import (
    ollama_reachable,
    run_live_embedding_ab,
    write_embedding_live_ab_report,
)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    settings = load_settings()

    if not ollama_reachable(settings.ollama_host):
        print(
            f"Ollama not reachable at {settings.ollama_host}; skipping live embedding A/B",
            file=sys.stderr,
        )
        return 2

    print(
        f"Running live embedding A/B (baseline={settings.embed_model})...",
        file=sys.stderr,
    )
    payload = run_live_embedding_ab(settings)
    out = write_embedding_live_ab_report(repo_root, payload)

    rate = payload["agreement_rate"]
    compared = payload["queries_compared"]
    print(
        f"Agreement rate: {rate:.3f} ({compared} queries); "
        f"candidate={payload['candidate_model']!r}",
        file=sys.stderr,
    )
    if payload.get("candidate_selection_note"):
        print(f"Candidate selection: {payload['candidate_selection_note']}", file=sys.stderr)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
