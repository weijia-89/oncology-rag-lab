# Roadmap

Items are ordered loosely by effort. None are committed; this is a personal lab.

## Near-term

**Adversarial eval for prompt injection**
`extract.py` notes this gap explicitly. Add `tests/eval/test_injection.py` with a small corpus of injection-pattern notes and assert that extraction output stays within the closed vocabulary. The 12 edge-case notes in `data/edge_case_notes/` are a starting point, though injection-specific ones would be added alongside them because the existing set targets parsing failures rather than adversarial inputs.

**Tighter eval assertions for regimen and histology**
`test_extraction_eval.py` falls back to `AnswerRelevancyMetric` for `regimen` and `histology` because those fields have multiple valid surface forms. A generic relevancy score isn't very meaningful here. A custom `GEval` metric with a clinical-plausibility rubric would be more precise, DeepEval supports this natively, and it would live alongside the existing eval file without restructuring anything.

**Lock dependencies**
`pyproject.toml` pins minimum versions only. Committing `uv.lock` would make installs fully reproducible across machines; currently it's excluded by `.gitignore`, so removing that entry and committing the lock file is a one-step change that costs nothing and gains a lot.

**In-memory index for eval-mock**
`test_retrieval.py` currently skips if `data/chroma_db/` doesn't exist. Building an in-memory ChromaDB fixture in `conftest.py` from the synthetic notes would let retrieval tests run in CI without a prior `make ingest` step, which matters because CI can't run `make ingest` and the skip means that test contributes zero signal on every push.

**Second embedding model A/B**
The README describes swapping `OllamaEmbedding` for `HuggingFaceEmbedding` (BGE-M3) as a one-line change in `ingest.py`. Running `drift_compare.py` between the two embeddings rather than just two LLMs would demonstrate retrieval-quality drift detection, which is a separate concern from generation-quality drift and worth treating separately.

## Longer-term

**Edge-case gold standard**
`data/gold_standard_edge_cases.csv` exists but `test_extraction_eval.py` only parameterizes over `data/gold_standard.csv`. Wiring the edge-case set into the eval suite would quantify how the pipeline handles copy-forward staleness, negation traps, staging conflicts, and similar failure modes that the synthetic notes don't cover.

**Scale stress test**
The design questions in the README identify what changes at 150M documents: chunking strategy, embedding batch size, async ingest, managed vector store. `scripts/seed_data.py` can generate hundreds of templated notes, so running ingest and retrieval against a 500-note corpus would surface the practical limits of file-backed ChromaDB and `SentenceSplitter` at this chunk size before those limits matter in a real workload.

**Structured output via function-calling**
`extract.py` uses regex-based JSON parsing as a fallback because Ollama's `format='json'` mode quality is model-dependent. That parse-failure path is noise. Qwen3 supports function-calling, and switching to a structured-output call would eliminate it entirely, making the `rationale` field part of the model's native output contract rather than something scraped out of free text.
