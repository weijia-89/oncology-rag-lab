# CLAUDE.md — agent context for this project

This file is for any Claude Code / Cowork agent that opens this directory. It exists so future-you (or another agent) doesn't have to re-derive context from a cold read.

## What this is
Local oncology RAG testbed. Mirror of the Ontada/McKesson Azure OpenAI entity extraction problem at lab scale. Built as Wei's portfolio project for the McKesson P3 SAE interview prep cycle.

## What it is not
- Not a HIPAA-compliant pipeline. All data is synthetic.
- Not a fine-tuning project.
- Not multi-user / production-grade.

## Stack (don't change without reason)
- Python 3.11+ via `uv`
- Ollama (qwen3:14b primary, nomic-embed-text for embeddings)
- LlamaIndex for the RAG layer (chosen over LangChain — strategy guide explains)
- ChromaDB for the vector store (file-based, persisted to `data/chroma_db/`)
- DeepEval for LLM-as-judge metrics (pytest-native)
- Arize Phoenix for tracing (UI at :6006)

## File map
See `src/onclab/__init__.py` for the package layout map. See `applications/mckesson/strategy_mckesson.md` § Personal LLM testing lab for the concept-to-file curriculum (8 modules, retrieval prompts, spacing schedule).

## Conventions
- `MOCK_LLM=1` env var bypasses Ollama with canned responses. `tests/conftest.py` sets this for the entire test session by default.
- `make` targets are the canonical command surface. Don't add commands outside the Makefile unless they need flags Make can't pass cleanly.
- `eval_baseline.json` is the regression floor. `scripts/check_regression.py` compares against it. Update via `cp eval_results.json eval_baseline.json` when intentionally raising the bar.
- Don't put model names or paths inline. Add to `src/onclab/config.py` and read via `Settings`.

## Where the strategic context lives
- `applications/mckesson/strategy_mckesson.md` — full role strategy + lab curriculum (single source of truth).
- `README.md` — user-facing quickstart, points back to the strategy doc for the curriculum.

## Cursor Cloud specific instructions

The update script runs `uv sync` in this repo.

- **Unit tests:** `uv run pytest tests/unit -v -m "not eval and not drift"` (30 tests, no Ollama needed)
- **Lint:** `uv run ruff check src/ tests/ scripts/` (1 pre-existing E501 in `tests/unit/test_extract_parse.py:60`)
- **Mock eval:** `MOCK_LLM=1 uv run pytest tests/eval -v -m eval` (tests eval pipeline scaffolding without Ollama)
- **Real eval/ingest/extract** require Ollama running locally with `qwen3:14b` and `nomic-embed-text` models. These are not available in the Cloud Agent VM by default.
- `MOCK_LLM=1` is set automatically by `tests/conftest.py` for the entire test session; unit tests never need a real LLM.
