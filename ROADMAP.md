# Roadmap

Items are ordered loosely by effort. None are committed beyond what shipped; this is a personal lab.

Last updated: 2026-05-24 (embedding drift eval SDK).

## Near-term

**Adversarial eval for prompt injection** — *open*  
`extract.py` notes this gap. Add `tests/eval/test_injection.py` with a small corpus of injection-pattern notes and assert extraction output stays within the closed vocabulary. The 12 edge-case notes in `data/edge_case_notes/` are a starting point; add injection-specific cases alongside them because the existing set targets parsing failures rather than adversarial inputs.

**Tighter eval assertions for regimen and histology** — *shipped 2026-05-23*  
[PR #3](https://github.com/weijia-89/oncology-rag-lab/pull/3) merged `regimen_histology_geval` in `src/onclab/eval_metrics.py` and wired it in `tests/eval/test_extraction_eval.py`. Custom GEval rubric replaces generic relevancy fallback for those fields.

**Lock dependencies** — *shipped 2026-05-23*  
[PR #4](https://github.com/weijia-89/oncology-rag-lab/pull/4) committed `uv.lock`; `.gitignore` entry removed. Intel Mac x86_64 uses `[tool.uv] override-dependencies` for `onnxruntime==1.23.2` ([PR #5](https://github.com/weijia-89/oncology-rag-lab/pull/5) context).

**In-memory index for eval-mock** — *shipped on `main`*  
`tests/conftest.py` `eval_index` fixture backs retrieval eval without persisted `data/chroma_db/`. Do **not** add a duplicate `oncology_chroma_ci` script; CI runs the fixture path.

**Second embedding model A/B** — *shipped 2026-05-24*  
`tests/eval/test_embedding_drift.py` runs deterministic in-memory top-1 chunk id A/B between two `MockEmbedding` configs on the synthetic corpus (no Ollama/HF). Follow-on: swap live `OllamaEmbedding` vs `HuggingFaceEmbedding` (BGE-M3) using the same compare helper shape.

## Longer-term

**Edge-case gold standard** — *shipped 2026-05-23*  
[PR #5](https://github.com/weijia-89/oncology-rag-lab/pull/5) wired `data/gold_standard_edge_cases.csv` into the extraction eval suite.

**Scale stress test** — *partial (2026-05-24)*  
CI-safe harness: `tests/eval/test_corpus_scale_stress.py` ingests 100 templated notes into in-memory Chroma (no Ollama, no persisted `data/chroma_db/`). Full 500-note file-backed run via `scripts/seed_data.py --count 500` + `make ingest` remains manual to surface ChromaDB disk limits before production scale.

**Structured output via function-calling** — *open*  
`extract.py` uses regex JSON fallback when Ollama `format='json'` is model-dependent. Qwen3 function-calling would make `rationale` part of the native output contract and remove parse-failure noise.
