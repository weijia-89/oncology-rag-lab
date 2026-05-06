"""Drift comparison: run the same eval against two model versions and diff.

Why this script exists separately from the eval suite:
    The eval suite measures "is the current pipeline good enough?"
    The drift script measures "is the new pipeline different from the
    old one, and where?". Different question, different output.
    Production teams run drift checks on every model deploy candidate;
    that's the workflow Ontada needs in their FDA-regulated pipeline,
    and this script is the lab-scale demonstration of that pattern.

What it produces:
    drift_report.csv with columns: patient_id, entity_type, baseline_value,
    candidate_value, agreement (1 if same, 0 if different).

The interview line:
    "Pipeline-version drift detection runs as a side-by-side diff over
    the same gold-standard set. The output is a CSV the QA Lead can
    review before approving a model swap. Same pattern Ontada needs
    for their Azure OpenAI version updates."

Implementation note (post-review):
    An earlier version mutated `os.environ` to switch models. That had two
    problems: (a) the side-effect leaked into any subsequent process that
    re-read settings, and (b) it relied on `load_settings()` being called
    after the mutation. The current version uses `dataclasses.replace`
    on a frozen Settings instance — no global mutation, no ordering
    dependency.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import sys
from pathlib import Path

from onclab.config import Settings, load_settings
from onclab.extract import ENTITY_TYPES, ExtractionRequest, extract_entity
from onclab.llm_client import OllamaClient


def run_for_model(*, model_name: str, base_settings: Settings) -> dict[tuple[str, str], str]:
    """Run extraction across all notes with one model.

    Returns {(patient_id, entity_type): predicted_value}.

    `base_settings` is treated as immutable. We derive a per-model copy
    via dataclasses.replace, which works because Settings is frozen.
    No env vars touched, no shared state mutated.
    """
    settings = dataclasses.replace(base_settings, llm_model=model_name)
    client = OllamaClient(host=settings.ollama_host, model=settings.llm_model)

    notes_dir = settings.notes_dir
    results: dict[tuple[str, str], str] = {}

    for note_path in sorted(notes_dir.glob("*.txt")):
        text = note_path.read_text(encoding="utf-8")

        # Extract patient_id from header (matches cli.py's parser).
        patient_id: str | None = None
        for line in text.splitlines()[:5]:
            if line.upper().startswith("PATIENT ID:"):
                patient_id = line.split(":", 1)[1].strip()
                break
        if patient_id is None:
            continue

        for entity_type in ENTITY_TYPES:
            request = ExtractionRequest(patient_id=patient_id, note_text=text)
            ent = extract_entity(request, entity_type, client=client, settings=settings)
            results[(patient_id, entity_type)] = ent.value

    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-model", required=True, help="Reference model (e.g., qwen3:14b)")
    parser.add_argument("--candidate-model", required=True, help="New model under evaluation")
    parser.add_argument("--out", required=True, type=Path, help="Output CSV path")
    args = parser.parse_args()

    base_settings = load_settings()

    print(f"Running baseline ({args.baseline_model})...", file=sys.stderr)
    baseline = run_for_model(model_name=args.baseline_model, base_settings=base_settings)

    print(f"Running candidate ({args.candidate_model})...", file=sys.stderr)
    candidate = run_for_model(model_name=args.candidate_model, base_settings=base_settings)

    rows: list[list[str]] = []
    keys = sorted(set(baseline.keys()) | set(candidate.keys()))
    agreements = 0
    for patient_id, entity_type in keys:
        b = baseline.get((patient_id, entity_type), "")
        c = candidate.get((patient_id, entity_type), "")
        # Case-insensitive comparison: a model that emits "IIIA" vs "iiia"
        # is the same answer for our purposes. If you ever need strict
        # casing (e.g., for proper-noun gene names), drop the `.lower()`.
        agree = 1 if b.strip().lower() == c.strip().lower() else 0
        agreements += agree
        rows.append([patient_id, entity_type, b, c, str(agree)])

    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["patient_id", "entity_type", "baseline", "candidate", "agree"])
        writer.writerows(rows)

    if rows:
        rate = agreements / len(rows)
        print(f"Agreement rate: {rate:.3f} ({agreements}/{len(rows)})")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
