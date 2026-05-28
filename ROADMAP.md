# Roadmap

Items are ordered loosely by effort. None are committed beyond what shipped; this is a personal lab.

Last updated: 2026-05-28.

## Near-term

**Adversarial eval for prompt injection** — *shipped 2026-05-25*  
`tests/eval/test_injection.py` covers `data/injection_notes/` (instruction override, fake JSON/code fences, role-play system prompts, delimiter escape) with closed-vocabulary assertions and simulated hijack rejection under `MOCK_LLM=1`. Guard lives in `extract.py` (`value_within_closed_vocabulary`).

**Tighter eval assertions for regimen and histology** — *shipped 2026-05-23*  
[PR #3](https://github.com/weijia-89/oncology-rag-lab/pull/3) merged `regimen_histology_geval` in `src/onclab/eval_metrics.py` and wired it in `tests/eval/test_extraction_eval.py`. Custom GEval rubric replaces generic relevancy fallback for those fields.

**Lock dependencies** — *shipped 2026-05-23*  
[PR #4](https://github.com/weijia-89/oncology-rag-lab/pull/4) committed `uv.lock`; `.gitignore` entry removed. Intel Mac x86_64 uses `[tool.uv] override-dependencies` for `onnxruntime==1.23.2` ([PR #5](https://github.com/weijia-89/oncology-rag-lab/pull/5) context).

**In-memory index for eval-mock** — *shipped on `main`*  
`tests/conftest.py` `eval_index` fixture backs retrieval eval without persisted `data/chroma_db/`. Do **not** add a duplicate `oncology_chroma_ci` script; CI runs the fixture path.

**Second embedding model A/B** — *shipped 2026-05-24*  
`tests/eval/test_embedding_drift.py` runs deterministic in-memory top-1 chunk id A/B between two `MockEmbedding` configs on the synthetic corpus (no Ollama/HF).

**Live Ollama embedding A/B** — *shipped 2026-05-25*  
`tests/eval/test_live_embedding_ab.py` and `scripts/embedding_live_ab_compare.py` reuse `src/onclab/embedding_compare.py` with real `OllamaEmbedding` models over in-memory Chroma (`ONCLAB_RUN_LIVE_EMBEDDING_AB=1`). Optional next human job: HuggingFace `HuggingFaceEmbedding` (BGE-M3) third arm — defer unless HF deps stay lightweight and tests skip without HF installed.

## Longer-term

**Edge-case gold standard** — *shipped 2026-05-23*  
[PR #5](https://github.com/weijia-89/oncology-rag-lab/pull/5) wired `data/gold_standard_edge_cases.csv` into the extraction eval suite.

**Scale stress test** — *partial (2026-05-24)*  
CI-safe harness: `tests/eval/test_corpus_scale_stress.py` ingests 100 templated notes into in-memory Chroma (no Ollama, no persisted `data/chroma_db/`). Full 500-note file-backed run via `scripts/seed_data.py --count 500` + `make ingest` remains manual to surface ChromaDB disk limits before production scale.

**Structured output via function-calling** — *partial (2026-05-28)*  
[#12](https://github.com/weijia-89/oncology-rag-lab/pull/12) adds `LlmExtractPayload` Pydantic validation on LLM JSON blobs before entity mapping. Native Qwen3 function-calling / `format='json'` hardening remains open.
