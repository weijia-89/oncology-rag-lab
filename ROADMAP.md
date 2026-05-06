# Roadmap

Items are ordered loosely by effort. None are committed; this is a personal lab.

## Near-term

**Adversarial eval for prompt injection**
`extract.py` notes this gap explicitly. Add `tests/eval/test_injection.py` with a small corpus of injection-pattern notes and assert that extraction output stays within the closed vocabulary. The 12 edge-case notes (`data/edge_case_notes/`) are a starting point; injection-specific ones would be added alongside.

**Tighter eval assertions for regimen and histology**
`test_extraction_eval.py` falls back to `AnswerRelevancyMetric` for `regimen` and `histology` because those fields have multiple valid surface forms. A custom `GEval` metric with a clinical-plausibility rubric would be more meaningful than a generic relevancy score. DeepEval supports this; it would live alongside the existing eval file.

**Lock dependencies**
`pyproject.toml` pins minimum versions only. Committing `uv.lock` would make installs fully reproducible across machines. Currently excluded by `.gitignore`; removing that entry and committing the lock file is a one-step change.

**In-memory index for eval-mock**
`test_retrieval.py` currently skips if `data/chroma_db/` doesn't exist. Building an in-memory ChromaDB fixture in `conftest.py` from the synthetic notes would let retrieval tests run in CI without a prior `make ingest` step.

**Second embedding model A/B**
The README describes swapping `OllamaEmbedding` for `HuggingFaceEmbedding` (BGE-M3) as a one-line change in `ingest.py`. Running `drift_compare.py` between the two embeddings, not just two LLMs, would demonstrate retrieval-quality drift detection, which is a separate concern from generation-quality drift.

## Longer-term

**Edge-case gold standard**
`data/gold_standard_edge_cases.csv` exists but `test_extraction_eval.py` only parameterizes over `data/gold_standard.csv`. Wiring the edge-case set into the eval suite would quantify how the pipeline handles copy-forward staleness, negation traps, staging conflicts, and similar failure modes.

**Scale stress test**
The design questions in the README identify what changes at 150M documents: chunking strategy, embedding batch size, async ingest, managed vector store. `scripts/seed_data.py` can generate hundreds of templated notes. Running ingest + retrieval against a 500-note corpus would surface the practical limits of file-backed ChromaDB and `SentenceSplitter` at this chunk size.

**Structured output via function-calling**
`extract.py` uses regex-based JSON parsing as a fallback because Ollama's `format='json'` mode quality is model-dependent. Qwen3 supports function-calling; switching to a structured-output call would eliminate the parse-failure path entirely and make the `rationale` field part of the model's native output contract.
