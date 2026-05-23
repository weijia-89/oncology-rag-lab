# onclab, local oncology RAG testbed

[![CI](https://github.com/weijia-89/oncology-rag-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/weijia-89/oncology-rag-lab/actions/workflows/ci.yml)

**Recent (2026-05-23):** Injection guard merged ([PR #2](https://github.com/weijia-89/oncology-rag-lab/pull/2) → `dbacc60`). GEval regimen/histology rubric on branch `feat/geval-regimen-histology-sdk` (open PR).

I built onclab to study the testing infrastructure a production oncology RAG pipeline would have to ship before it could safely touch real clinical data. The pipeline itself is intentionally ordinary, an extractor for structured oncology entities (cancer type, AJCC stage, regimen, ECOG) running over synthetic clinical notes via LlamaIndex on Chroma with Ollama for inference. The part I spent the time on is the wrap around it, the DeepEval suite scored against gold-standard labels, the regression gate that fails CI if pass rate drops more than 5% versus baseline, A/B drift detection between model versions, and the Arize Phoenix traces that record every retrieval and extraction so the audit trail exists in the artifact rather than the design doc.

The corpus is small on purpose. I wrote 8 base synthetic notes and 12 adversarial edge-case notes. Each adversarial note targets a specific failure mode a real clinical pipeline hits:

- copy-forward staleness from old chart entries
- unfilled SmartPhrase templates
- Dragon transcription errors
- staging-system collisions
- negation traps in radiology language
- blinded trial regimens that obscure the actual drug
- unit ambiguity

An extractor that passes on the textbook-clean base notes is not the same thing as an extractor that passes on the edge cases, and the gap between those two pass rates is what the testing wrap is measuring.

## What's in the box

| Layer | Tool | Where it lives |
|---|---|---|
| Inference | Ollama (qwen3:14b) | local server at :11434 |
| Embedding | nomic-embed-text via Ollama | `src/onclab/ingest.py` |
| Vector store | ChromaDB (file-based) | `data/chroma_db/` |
| RAG framework | LlamaIndex | `src/onclab/rag.py` |
| Entity extraction | pydantic schema + per-entity prompts | `src/onclab/extract.py` |
| Eval | DeepEval (pytest-native) | `tests/eval/` |
| Observability | Arize Phoenix | `src/onclab/phoenix_setup.py`, UI at :6006 |
| Drift detection | A/B model comparison | `scripts/drift_compare.py` |

## Quickstart (fresh box)

```bash
# 1. Install Ollama (https://ollama.com), then start it.
# 2. From this directory:
scripts/bootstrap.sh # uv sync + ollama pull qwen3:14b + nomic-embed-text
make ingest # chunk + embed + index the synthetic corpus (8 base + 12 edge-case notes)
make extract # see the structured outputs for each note
```

To run the eval suite end-to-end:

```bash
make eval # DeepEval-driven tests with real Ollama calls (slow)
make check-regression # gate: did pass rate drop >5% vs baseline?
```

For unit tests with no Ollama dependency:

```bash
make test-unit # MOCK_LLM=1; runs in seconds; CI-friendly
```

## Why the testing wrap matters more than the pipeline

The honest answer to "is this a production-ready extractor" is no, and `data/FIDELITY_REVIEW.md` walks through why. I compared the 8 base synthetic notes against 3 real de-identified MTSamples transcriptions and catalogued 12 ways the synthetic corpus diverges from clinical reality, things like the hyper-structured template format that real notes do not share, the absence of comorbidities and hedge language, an ECOG distribution skewed toward 0 and 1, and biomarker results landing at the wrong visit type. The 12 edge-case notes in `data/edge_case_notes/` push past the textbook-clean format the base notes share, but a corpus of 20 synthetic notes still cannot simulate the long tail of real clinical narrative.

What the corpus can do is give the testing infrastructure something to score, and that infrastructure is the part I'd want to see in any production-leaning oncology pipeline. DeepEval scores per-entity precision against gold-standard labels in `tests/eval/gold_standard.py`. The regression gate in `scripts/check_regression.py` reads the latest run, compares against a pinned baseline (with explicit handling for the first-run placeholder case so an empty baseline cannot silently pass, a bug that was a P1 audit finding I had to fix), and fails CI if pass rate drops more than 5%. `scripts/drift_compare.py` runs the same notes through two model versions and reports per-entity divergence. Phoenix traces every retrieval call and every entity extraction so the audit trail exists rather than getting written into the design doc and forgotten.

The unit test layer keeps this useful in CI without burning 2000 Ollama calls on every PR. Every LLM call carries a `mock_key` like `extract:patient_001:cancer_type`, so under `MOCK_LLM=1` the test suite gets deterministic responses from `tests/fixtures/mock_llm_responses.py`, runs in seconds, and exercises the parse-and-validate logic without touching the model. The integration tests under `make eval` hit real Ollama for end-to-end behavior and live behind a separate make target so contributors don't accidentally run them.

## Read this in order

If you want to learn the layers, the data flows through the files in this order:

1. `src/onclab/config.py`: every knob in one place.
2. `src/onclab/llm_client.py`: Ollama wrapper plus the MOCK_LLM toggle.
3. `src/onclab/ingest.py`: load, chunk, embed, store.
4. `src/onclab/rag.py`: retrieval and the query engine.
5. `src/onclab/extract.py`: the actual oncology task.
6. `tests/eval/test_retrieval.py`: deterministic-layer tests.
7. `tests/eval/test_extraction_eval.py`: DeepEval scoring.
8. `scripts/drift_compare.py`: side-by-side model diff.

Each file has a top-of-module comment explaining the design decision and what I tried first.

## Hardware notes

Sized for an RTX 4070 Ti Super (16 GB VRAM) with 32 GB system RAM. Qwen3 14B Q4_K_M lands at around 8.7 GB and runs at roughly 30-40 tokens/sec on that card; if you've got 8 GB VRAM, swap to llama3.1:8b in `.env` and the eval suite will still pass.

vLLM is intentionally absent. For single-user batch eval, Ollama is faster to set up and gives you comparable throughput. vLLM's advantages are at concurrent-request scale, which a one-person eval harness never hits.

## Design questions worth thinking through

- What changes if the corpus is 150M documents instead of 20? (Chunking strategy, embedding batch size, ChromaDB → managed vector store, async ingest pipeline.)
- How does test design change when FDA sensitivity/specificity replaces internal SLAs? (Weighted precision/recall metrics in GEval; stricter false-negative budget for critical entities like `ajcc_stage`.)
- Where does the audit trail live in this design? (Every retrieval call traces to Phoenix; every entity carries a rationale string.)

## Files you might add later

- `data/synthetic_notes/patient_009.txt` and onward: `scripts/seed_data.py` shows the template pattern.
- A second embedding model (BGE-M3) for retrieval quality A/B: `OllamaEmbedding` -> `HuggingFaceEmbedding`, one swap in `ingest.py`.
- A custom GEval metric for clinical-plausibility scoring: DeepEval supports it; goes alongside `test_extraction_eval.py`.

## Related portfolio repos

onclab sits in a portfolio of QA-for-AI work. The two repos that pair most directly:

- **[`weijia-89/vibe-check`](https://github.com/weijia-89/vibe-check)**: a reviewer evidence surfacer for PRs that may contain LLM-generated code. Same evaluation-discipline shape (baseline-pinned, audit-trail-honest, regression-test-guarded) applied to code review instead of LLM evaluation.
- **[`weijia-89/playwrighter`](https://github.com/weijia-89/playwrighter)**: production Playwright pattern library plus a test-quality scorer. Same "the wrap matters more than the suite" stance, applied to E2E testing rather than RAG eval.

Three more in the same ethos:

- **[`weijia-89/northwind-qa`](https://github.com/weijia-89/northwind-qa)**: 50-test Playwright suite that exercises playwrighter's patterns end-to-end against a React 19 SUT and ships seven real bug reports.
- **[`weijia-89/palamedes`](https://github.com/weijia-89/palamedes)**: rigorous-research skill plus a multi-agent synthesis prompt. Companion artifact when the eval target is research output rather than code or a model.
- **[`weijia-89/wcag-auditor`](https://github.com/weijia-89/wcag-auditor)**: accessibility audit tool that replaced its LLM-based fix engine with deterministic per-rule templates in v0.3, because the templates were already accurate enough.

## License / Provenance

All clinical content in `data/synthetic_notes/` and `data/edge_case_notes/` is invented. No PHI. The 3 reference transcriptions in `data/real_notes/` (used only for the fidelity review) come from MTSamples, are already de-identified at the source, and stay git-ignored so this repo never re-publishes them. See `data/FIDELITY_REVIEW.md` for the methodology and the 12 fidelity gaps the synthetic notes do not cover.

Code is MIT-licensed (see `LICENSE`).
