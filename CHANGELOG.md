# Changelog

## [0.1.2] - 2026-05-06

### Fixed

- **`response["response"]` and `response["message"]["content"]` dict access broken on modern ollama-python.** Found this during an adversarial review of the Ollama client contract. The `generate()` return type changed from a plain dict to a response object somewhere around ollama-python 0.2; the dict key still evaluates without error because the object implements `__getitem__` via some internal shim, but it will silently return wrong data on certain builds. Switched both callsites (`llm_client.py:148` and `llm_client.py:162`) to attribute access (`response.response.strip()` and `response.message.content.strip()`), which is what the official SDK docs show and what all the example code in the ollama-python README uses. The dict form works today. It won't always.

- **SSRF guard passed public routable IPs silently.** This one was subtle. `validate_service_url()` in `config.py` correctly blocked cloud-metadata CIDRs (169.254.0.0/16, etc.) and then `continue`-d through loopback and RFC-1918. But the loop had no `else` or trailing `raise`, so a public IP like `93.184.216.34` just fell off the end and returned `url` unchanged. Added an explicit `raise ValueError(...)` for anything that reaches the bottom of the loop, which is everything that isn't loopback, private, or already-blocked. The original intent was clearly to allow only local/LAN services; the guard was just missing the last case.

## [0.1.1] - 2026-05-06

### Fixed

- **Retrieval tests always skipped in CI.** Noticed that `tests/eval/test_retrieval.py` did a module-scope `pytest.skip` whenever `data/chroma_db/` didn't exist, which is always in CI because `make ingest` is never run. That's a dead test. Added `tests/unit/test_retrieval_unit.py` with an in-memory ChromaDB fixture using `MockEmbedding`, so the wiring between `VectorStoreIndex`, `ChromaVectorStore`, and `retrieve_only()` is covered without Ollama or a persisted index. Also added `importorskip` guards to `test_chunking.py` so missing `llama_index` skips cleanly instead of aborting the collection phase.

- **SSRF via OLLAMA_HOST and PHOENIX_COLLECTOR_ENDPOINT.** Found that both env vars were passed straight to their respective clients with no validation, which meant setting `OLLAMA_HOST=http://169.254.169.254/...` would silently redirect all LLM calls to the cloud metadata endpoint. Worth noting: this is a real class of attack, not a theoretical one. Added `validate_service_url()` in `config.py` that resolves the hostname to an IP, blocks the `169.254.0.0/16` link-local range explicitly before the `is_private` check (important because Python 3.10 classifies link-local as private), and allows loopback and RFC-1918 addresses for normal localhost/LAN use. The order of `is_private` vs. the blocked-networks check matters a lot depending on the Python minor version in use.

- **No retry on Ollama calls in `extract.py`.** Traced through the call path and confirmed that `OllamaClient.generate()` had no error handling on the real path; a single `ConnectionError` during a 400-second eval run would abort everything with no partial results. Brutal. Added a three-attempt retry loop with a two-second backoff using only `time.sleep` and no external library, because adding a dep for three lines of retry logic isn't worth it. The loop re-raises on the final failure so genuine model outages still surface cleanly.

- **No lint step in CI.** `Makefile` had a `lint` target but `.github/workflows/ci.yml` never called it. Added a `ruff check src/ tests/` step before the unit test step so lint regressions are caught on every push, not just locally.

### Added

- `tests/unit/test_retrieval_unit.py`: unit-level retrieval wiring test with in-memory ChromaDB and `MockEmbedding`
- `tests/unit/test_url_validator.py`: 11 tests for `validate_service_url`, covering loopback, RFC-1918, `169.254.x.x` blocking, bad schemes, and unresolvable hostnames
- `tests/unit/test_retry.py`: 4 tests verifying retry-on-transient-failure, success-on-first-try, exhausted-retries error, and mock-mode bypass

## [0.1.0] - 2025-04-30

Initial implementation.

### Added

- RAG pipeline: LlamaIndex + ChromaDB (file-backed) + Ollama `qwen3:14b` for generation, `nomic-embed-text` for embeddings
- `src/onclab/` library with modules for config, ingest, retrieval, extraction, observability, and a Typer CLI (`onclab ingest`, `onclab extract`, `onclab query`)
- Structured entity extraction for five oncology fields: `cancer_type`, `ajcc_stage`, `histology`, `regimen`, `ecog`: per-entity prompts, Pydantic output schema, regex-based JSON parsing with graceful fallback to `unknown`
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
