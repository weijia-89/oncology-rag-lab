# Changelog

## [0.1.1] - 2026-05-06

### Fixed

- **Retrieval tests always skipped in CI.** Noticed that `tests/eval/test_retrieval.py` did a module-scope `pytest.skip` whenever `data/chroma_db/` didn't exist — which is always in CI because `make ingest` is never run. Added `tests/unit/test_retrieval_unit.py` with an in-memory ChromaDB fixture using `MockEmbedding`, so the wiring between `VectorStoreIndex`, `ChromaVectorStore`, and `retrieve_only()` is covered without Ollama or a persisted index. Also added `importorskip` guards to `test_chunking.py` so missing `llama_index` skips cleanly instead of aborting the collection phase.

- **SSRF via OLLAMA_HOST and PHOENIX_COLLECTOR_ENDPOINT.** Found that both env vars were passed straight to their respective clients with no validation — setting `OLLAMA_HOST=http://169.254.169.254/...` would silently redirect all LLM calls to the cloud metadata endpoint. Added `validate_service_url()` in `config.py` that resolves the hostname to an IP, blocks the `169.254.0.0/16` link-local range explicitly (before the `is_private` check, which is important because Python 3.10 classifies link-local as private), and allows loopback and RFC-1918 addresses for normal localhost/LAN use. Learned that the order of `is_private` vs. the blocked-networks check matters a lot depending on the Python minor version.

- **No retry on Ollama calls in `extract.py`.** Traced through the call path and confirmed that `OllamaClient.generate()` had no error handling on the real path — a single `ConnectionError` during a 400-second eval run would abort everything with no partial results. Added a three-attempt retry loop with a two-second backoff using only `time.sleep`, no external library. The loop re-raises on the final failure so genuine model outages still surface cleanly.

- **No lint step in CI.** `Makefile` had a `lint` target but `.github/workflows/ci.yml` never called it. Added a `ruff check src/ tests/` step before the unit test step so lint regressions are caught on every push, not just locally.

### Added

- `tests/unit/test_retrieval_unit.py` — unit-level retrieval wiring test with in-memory ChromaDB and `MockEmbedding`
- `tests/unit/test_url_validator.py` — 11 tests for `validate_service_url`, covering loopback, RFC-1918, `169.254.x.x` blocking, bad schemes, and unresolvable hostnames
- `tests/unit/test_retry.py` — 4 tests verifying retry-on-transient-failure, success-on-first-try, exhausted-retries error, and mock-mode bypass

## [0.1.0] - 2025-04-30

Initial implementation.

### Added

- RAG pipeline: LlamaIndex + ChromaDB (file-backed) + Ollama `qwen3:14b` for generation, `nomic-embed-text` for embeddings
- `src/onclab/` library with modules for config, ingest, retrieval, extraction, observability, and a Typer CLI (`onclab ingest`, `onclab extract`, `onclab query`)
- Structured entity extraction for five oncology fields: `cancer_type`, `ajcc_stage`, `histology`, `regimen`, `ecog` — per-entity prompts, Pydantic output schema, regex-based JSON parsing with graceful fallback to `unknown`
- `MOCK_LLM=1` mode in `llm_client.py`: canned responses derived from `gold_standard.csv` at import time; lets unit tests run without Ollama
- Eval suite (`tests/eval/`): `test_retrieval.py` for deterministic retrieval-layer assertions; `test_extraction_eval.py` for DeepEval `AnswerRelevancyMetric` gates
- Unit tests (`tests/unit/`): cover chunking, extraction parsing, and the LLM client mock contract; run in CI under `MOCK_LLM=1`
- Arize Phoenix integration (`phoenix_setup.py`): opt-in tracing via `--trace` flag on `extract` and `query` commands; LlamaIndex auto-instrumented via `LlamaIndexInstrumentor`
- `scripts/drift_compare.py`: side-by-side extraction diff across two Ollama model versions; writes `drift_report.csv`
- `scripts/check_regression.py`: compares pytest JSON report pass rate against a committed baseline; exits non-zero if drop exceeds 5%
- `scripts/seed_data.py`: template-based synthetic note generator for expanding the corpus
- `scripts/bootstrap.sh`: one-command setup (uv sync + Ollama model pulls)
- GitHub Actions CI workflow: unit tests only, Python 3.11, `ubuntu-latest`; eval and drift markers excluded (require Ollama)
- `data/synthetic_notes/`: 8 hand-written fictional clinical notes covering lung, breast, colorectal, prostate, lymphoma, pancreatic, ovarian, and glioblastoma cases
- `data/edge_case_notes/`: 12 edge-case notes covering copy-forward staleness, unfilled smartphrase templates, Dragon transcription errors, addendum stage corrections, staging system collisions, dual primaries, negation traps, dose reductions, blinded trials, unstructured dictation, problem-list conflicts, and unit ambiguity
- `data/gold_standard.csv` and `data/gold_standard_edge_cases.csv`
- `data/real_notes/` added to `.gitignore` to prevent accidental commit of any real clinical content
- `eval_baseline.json`: committed placeholder; `check_regression.py` seeds it from the first real eval run
