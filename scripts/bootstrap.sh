#!/usr/bin/env bash
# Idempotent bootstrap: install + pull models + seed Phoenix.
# Run once per machine. After that, `make eval` is enough.
set -euo pipefail

# Resolve the project root regardless of where this is invoked from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

# 1. Python deps via uv. Falls back to pip if uv isn't installed yet.
if command -v uv >/dev/null 2>&1; then
  uv sync
else
  echo "uv not found; installing via pip + venv as a fallback."
  python3 -m venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  pip install -e .
fi

# 2. Verify Ollama is running. We don't auto-start it because the user
# may have it as a system service (macOS), a manual `ollama serve` (Linux),
# or a desktop app (Windows). Whatever they prefer.
if ! curl -fsS http://localhost:11434/api/tags >/dev/null 2>&1; then
  echo "WARNING: Ollama is not reachable at localhost:11434."
  echo "  - macOS:   brew services start ollama"
  echo "  - Linux:   ollama serve (in a separate terminal)"
  echo "  - Windows: launch the Ollama desktop app"
  echo "  Then re-run: scripts/bootstrap.sh"
  exit 1
fi

# 3. Pull the models we depend on. ollama pull is idempotent; if the
# model is already local with the same digest, it returns immediately.
echo "Pulling qwen3:14b (8.7 GB)..."
ollama pull qwen3:14b
echo "Pulling nomic-embed-text (270 MB)..."
ollama pull nomic-embed-text

cat <<'EOF'

Bootstrap complete.

Next steps:
  make ingest   # build ChromaDB index from synthetic notes
  make extract  # show structured entities for every note
  make eval     # run the DeepEval suite (slow; needs Ollama)

For tracing:
  make phoenix  # start the Phoenix UI in a separate terminal, then re-run with --trace
EOF
