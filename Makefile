# Why a Makefile when this is a Python project:
#   Make targets are the lab's command-line API. `make eval` is shorter
#   than remembering `uv run pytest tests/eval -m eval --json-report`.
#   It also documents the workflow: read this file top-to-bottom and
#   you have the build/test/run lifecycle.
#
# Cross-platform note: `make` on Windows requires WSL or a tool like
# `chocolatey install make`. If you're on the 4070 Ti Super box and
# don't want WSL, the equivalent commands are in the README.

.PHONY: install bootstrap ingest extract eval eval-mock test-unit lint check-regression phoenix clean audit drift

# ---- Setup ----

install:
	# uv sync = read pyproject.toml, build a locked venv, install everything.
	# Idempotent — safe to re-run after editing deps.
	uv sync

bootstrap: install
	# `bootstrap` = "everything `install` does PLUS pull the Ollama models".
	# Split from `install` because pulling models is ~9 GB on first run
	# and you only want to do it once. Subsequent `install` should be fast.
	@echo "Pulling Ollama models (this is the slow first-time step)..."
	ollama pull qwen3:14b           # primary LLM, ~8.7GB Q4_K_M
	ollama pull nomic-embed-text    # embedding model, ~270MB
	@echo "Models ready. Run: make ingest"

# ---- Pipeline commands ----

ingest:
	# Reads data/synthetic_notes/*.txt, chunks, embeds, stores in ChromaDB.
	# Idempotent: re-running rebuilds the index from scratch.
	# This is the "build the haystack" step before retrieval.
	uv run onclab ingest --notes-dir data/synthetic_notes --persist-dir data/chroma_db

extract:
	# Runs the entity extraction pipeline against ingested notes.
	# Prints structured output (cancer_type, stage, regimen, ...) per note.
	# Useful for eyeballing the pipeline before throwing eval at it.
	uv run onclab extract --persist-dir data/chroma_db

phoenix:
	# Starts the Phoenix observability server in the foreground.
	# Run this in a second terminal *before* the eval/extract commands
	# if you want to see the traces. `localhost:6006`.
	uv run python -c "import phoenix as px; px.launch_app(); input('Phoenix at http://localhost:6006 — Ctrl-C to stop\n')"

# ---- Tests + eval ----

test-unit:
	# Fast (~seconds). No Ollama required. Run on every commit.
	# Exclude the `eval` marker so we skip the slow LLM-judged tests here.
	uv run pytest tests/unit -v -m "not eval and not drift"

eval-mock:
	# Eval pass with MOCK_LLM=1 — uses canned responses instead of calling Ollama.
	# This is the "smoke test" for the eval pipeline itself: did our
	# DeepEval test cases parse, did the metrics initialize, did the
	# scaffolding hold together. Fast and CI-friendly.
	MOCK_LLM=1 uv run pytest tests/eval -v -m eval \
	  --json-report --json-report-file=eval_results_mock.json

eval:
	# Real eval. Calls Ollama. Takes minutes. The number that matters
	# for the interview lives here: pass rate per metric on the gold
	# standard set.
	@echo "Running real eval — needs `ollama serve` running."
	uv run pytest tests/eval -v -m eval \
	  --json-report --json-report-file=eval_results.json

drift:
	# Drift = run the same test cases against two pipeline versions
	# (e.g., qwen3:14b vs llama3.1:8b) and compare metric scores.
	# This is the pattern Ontada needs in production for model updates.
	uv run python scripts/drift_compare.py \
	  --baseline-model qwen3:14b \
	  --candidate-model llama3.1:8b \
	  --out drift_report.csv

# ---- Quality gates ----

lint:
	uv run ruff check src/ tests/ scripts/

audit:
	# Supply-chain audit. Catches CVEs in installed deps before they
	# bite. Pinning + auditing is the discipline the slopsquatting
	# research community keeps flagging — 5–22% of LLM-suggested package
	# names are hallucinated, so anything we install gets audited.
	# Install once: `uv tool install pip-audit`.
	pip-audit

check-regression:
	# Compare today's eval_results.json against eval_baseline.json.
	# Fails if any metric dropped more than 5%. This is the interview-grade
	# story: "we don't merge if the eval suite regresses, automated."
	uv run python scripts/check_regression.py \
	  --results eval_results.json \
	  --baseline eval_baseline.json \
	  --max-drop 0.05

# ---- Housekeeping ----

clean:
	# Remove the vector index, eval outputs, and pytest caches.
	# Useful before a clean re-run / before committing.
	rm -rf data/chroma_db eval_results*.json drift_report.csv
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
