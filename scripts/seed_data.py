"""Generate additional synthetic clinical notes from a template.

The committed lab ships with 8 hand-written notes (data/synthetic_notes/).
That's enough to demonstrate the pipeline. If you ever want to scale to
50 or 500 to stress-test retrieval, this script generates them from a
schema + template.

Why hand-write the first 8 instead of all-template:
    Hand-written notes have realistic prose variation, fragment sentences,
    abbreviations, the kind of stuff a real clinician writes. Templated
    notes are too regular — a model can game the eval by latching onto
    the template structure. Mixing some hand-written + some templated
    is a more honest distribution.

Important: ALL data here is invented. There is no PHI, ever. The point
of synthetic data is documented in the strategy guide — ASCO's Feb 2026
HHS RFI submission specifically calls out synthetic data + digital twins
as the recommended approach for clinical AI evaluation.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

# Tiny templating. A real synthetic-data pipeline would use Faker or
# Synthea; this is the lab-grade version.

TEMPLATE = """PATIENT ID: SYN-{patient_num:03d}
NOTE DATE: 2025-{month:02d}-{day:02d}
PROVIDER: Synthetic Provider {provider}

HISTORY OF PRESENT ILLNESS:
{age}-year-old {sex} presenting with {symptoms}. Imaging revealed {imaging_finding}.

DIAGNOSIS:
Stage {stage} {cancer}, {histology}. {staging_code} per AJCC 8th edition. {biomarkers}.

PERFORMANCE STATUS:
ECOG {ecog}.

TREATMENT PLAN:
{regimen}.

LABS:
Hemoglobin {hgb} g/dL. {extra_labs}.
"""

CASES = [
    {
        "cancer": "ovarian cancer",
        "histology": "high-grade serous carcinoma",
        "stage": "IIIC",
        "staging_code": "T3cN0M0",
        "biomarkers": "BRCA1 germline mutation",
        "regimen": "Cytoreductive surgery followed by carboplatin + paclitaxel + bevacizumab + olaparib maintenance",
        "imaging_finding": "ascites and omental caking",
        "symptoms": "abdominal distension and early satiety",
        "extra_labs": "CA-125 980 U/mL. Albumin 3.2 g/dL",
    },
    {
        "cancer": "renal cell carcinoma",
        "histology": "clear cell type",
        "stage": "IV",
        "staging_code": "T3aN0M1",
        "biomarkers": "intermediate IMDC risk",
        "regimen": "Pembrolizumab + axitinib first-line",
        "imaging_finding": "8 cm renal mass with three pulmonary nodules",
        "symptoms": "gross hematuria and fatigue",
        "extra_labs": "Creatinine 1.4 mg/dL. Calcium 10.1 mg/dL",
    },
]


def render(idx: int, case: dict) -> str:
    return TEMPLATE.format(
        patient_num=100 + idx,
        month=random.randint(1, 12),
        day=random.randint(1, 28),
        provider=random.choice(["A", "B", "C", "D"]),
        age=random.randint(45, 80),
        sex=random.choice(["man", "woman"]),
        ecog=random.choice([0, 1, 1, 2]),  # weighted toward 1
        hgb=round(random.uniform(10.5, 14.0), 1),
        **case,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "synthetic_notes",
    )
    parser.add_argument("--count", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(args.count):
        case = CASES[i % len(CASES)]
        text = render(i, case)
        (args.out_dir / f"patient_{100 + i:03d}.txt").write_text(text, encoding="utf-8")

    print(f"Generated {args.count} synthetic notes in {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
