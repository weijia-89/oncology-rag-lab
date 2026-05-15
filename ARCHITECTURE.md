# Architecture

## Data flow

```
data/synthetic_notes/*.txt
 │
 
 ingest.py chunk → embed → store
 (make ingest) SentenceSplitter (500 chars, 50-char overlap)
 nomic-embed-text via Ollama
 ChromaDB (file-backed, data/chroma_db/)
 │
 
 rag.py retrieve top-k chunks for any query
 (retrieve_only VectorIndexRetriever, top_k=3
 or query engine) LlamaIndex RetrieverQueryEngine
 Ollama qwen3:14b for generation
 │
 
 extract.py per-entity prompts → Pydantic-validated output
 (make extract) 5 entities: cancer_type, ajcc_stage, histology,
 regimen, ecog
 Returns ExtractedEntity(value, confidence, rationale)
 │
 
 tests/eval/ DeepEval metrics as pytest assertions
 (make eval) test_extraction_eval.py: exact-match + AnswerRelevancyMetric
 test_retrieval.py: classical chunk-content assertions
 │
 
 scripts/ regression gate and drift detection
 check_regression compares eval_results.json against eval_baseline.json
 drift_compare side-by-side extraction diff across two model versions
```

## Module responsibilities

| Module | Responsibility |
|---|---|
| `config.py` | All env vars and defaults in one frozen dataclass (`Settings`). No other module hard-codes model names or paths. |
| `llm_client.py` | Thin wrapper over the `ollama` Python lib. Handles `MOCK_LLM=1` mode: returns gold-standard-derived canned JSON so unit tests run without Ollama. |
| `ingest.py` | Reads `.txt` files, runs `SentenceSplitter`, embeds with `OllamaEmbedding`, writes to ChromaDB. Two public functions: `build_index` (slow, writes to disk) and `load_index` (fast, reads existing). |
| `rag.py` | Exposes `build_retriever` (deterministic, no LLM), `build_query_engine` (retrieval + LLM generation), and `retrieve_only` (helper for retrieval-layer tests). |
| `extract.py` | Defines `ENTITY_TYPES`, per-entity prompt strings, `ExtractedEntity` Pydantic schema, and `extract_entity` / `extract_all`. Parsing uses regex to find the first `{...}` block in the LLM response; falls back to an `unknown` entity rather than raising. |
| `phoenix_setup.py` | Idempotent `setup_phoenix()` that registers an Arize Phoenix tracer provider and auto-instruments LlamaIndex via `LlamaIndexInstrumentor`. Called only in CLI commands that want tracing (`--trace` flag); skipped in all tests. |
| `cli.py` | Typer app with three commands: `ingest`, `extract`, `query`. `extract` and `query` accept `--trace/--no-trace`. |

## Models

| Role | Default | Override |
|---|---|---|
| LLM (generation + extraction) | `qwen3:14b` via Ollama | `ONCLAB_LLM_MODEL` env var |
| Embedding | `nomic-embed-text` via Ollama | `ONCLAB_EMBED_MODEL` env var |

Both models are pulled locally; no external API calls are made anywhere in the codebase. `OLLAMA_HOST` defaults to `http://localhost:11434`, and `validate_service_url()` in `config.py` rejects anything that resolves to a public IP so a misconfigured env var can't silently route traffic to an external server.

## Configuration surface

All knobs live in `config.py`. Runtime values come from environment variables, and the `.env.example` file lists them all.

| Env var | Default | Purpose |
|---|---|---|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `ONCLAB_LLM_MODEL` | `qwen3:14b` | LLM for generation and extraction |
| `ONCLAB_EMBED_MODEL` | `nomic-embed-text` | Embedding model |
| `MOCK_LLM` | `0` | Set to `1` to use canned responses (CI, unit tests) |
| `PHOENIX_PROJECT_NAME` | `oncology-rag-lab` | Arize Phoenix project label |

Retrieval knobs (`chunk_size`, `chunk_overlap`, `top_k`) and eval thresholds (`hallucination_threshold`, `faithfulness_threshold`, `answer_relevancy_threshold`) have hard-coded defaults in `Settings`, but they can be overridden by constructing a custom `Settings` instance directly if you need non-default behavior in a specific test scenario.

## Eval harness

Tests live in two directories:

- `tests/unit/`: no Ollama dependency; run under `MOCK_LLM=1`. Covers chunking behavior, extraction parsing, and the LLM client mock contract — these run on every CI push.
- `tests/eval/`: require Ollama (or `MOCK_LLM=1`). Marked `eval` so `pytest -m "not eval"` skips them. Includes `test_retrieval.py` (classical chunk-content assertions against the live ChromaDB index) and `test_extraction_eval.py` (DeepEval `AnswerRelevancyMetric` as a pytest gate).

The `eval` marker split keeps CI fast. The full DeepEval suite runs locally or in a nightly job, because running it in CI would require Ollama, which would require a GPU — slower and more expensive for a suite that doesn't need to fire on every push.

`scripts/check_regression.py` reads `eval_results.json` (written by `pytest --json-report`) and compares the pass rate against `eval_baseline.json` (committed). Exit code 1 if the drop exceeds `--max-drop` (default 5%).

## Drift detection

`scripts/drift_compare.py` runs the same extraction pass against two Ollama model names and writes `drift_report.csv` with per-`(patient_id, entity_type)` agreement. Model swapping goes through `dataclasses.replace` on the frozen `Settings` object, so neither env vars nor global state are touched between runs. Agreement is case-insensitive.

## Observability

Arize Phoenix (`arize-phoenix`, `openinference-instrumentation-llama-index`) traces every retrieval and generation call, publishing spans to `http://localhost:6006` (configurable via `PHOENIX_COLLECTOR_ENDPOINT`). Start the UI separately with `make phoenix`. Tracing is opt-in per CLI invocation via `--trace`; the instrumentor connects to whatever endpoint is reachable at that point.

## Build and CI

- Package manager: `uv`. Build backend: `hatchling`.
- `onclab` installs as a console script pointing at `onclab.cli:app`.
- Dev dependencies (`pytest`, `ruff`, `pytest-json-report`) are in the `dev` dependency group.
- CI (`.github/workflows/ci.yml`): single job on `ubuntu-latest`, Python 3.11, runs `pytest tests/unit -m "not eval and not drift"`. No Ollama in CI; unit tests run under `MOCK_LLM=1` via the mock fixtures in `tests/conftest.py`.
- Linting: ruff with `E`, `F`, `I`, `UP`, `B` rules, line length 120, `B008` suppressed for Typer's option-default idiom.

## Directory layout

```
src/onclab/ library code (config, llm_client, ingest, rag, extract, phoenix_setup, cli)
tests/unit/ fast, no Ollama (test_chunking, test_extract_parse, test_llm_client, test_retrieval_unit, test_retry, test_url_validator)
tests/eval/ slow, Ollama or MOCK_LLM (test_retrieval, test_extraction_eval)
scripts/ standalone scripts (bootstrap.sh, seed_data.py, drift_compare.py, check_regression.py)
data/synthetic_notes/ 8 hand-written fictional clinical notes
data/edge_case_notes/ 12 edge-case notes (copy-forward staleness, negation traps, staging conflicts, etc.)
data/gold_standard.csv ground truth for the eval suite
data/chroma_db/ generated by make ingest (git-ignored)
data/real_notes/ git-ignored; never committed
```
